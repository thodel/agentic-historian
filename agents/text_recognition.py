"""
agents/text_recognition.py – Agent A: Text Recognition (HTR)

Pipeline:
  A-a  Group files by document stem → treat as "one document"
  A-b  Run HTR via Claude Vision (simulates model selection)
  A-c  Save transcription as .txt + upload to HuggingFace
  A-d  Quality-check with a second Claude call
  A-e  If quality < threshold → retry with adjusted prompt
"""
import asyncio
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from loguru import logger

import config
from utils.claude_client import ask, ask_structured

# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_HTR = """You are an expert palaeographer specialising in medieval German
and Latin manuscripts (13th–15th century). Your task is to transcribe
handwritten historical documents as faithfully as possible.

Rules:
- Preserve original spelling, punctuation and line breaks.
- Mark unclear readings with [?] immediately after the word.
- Mark illegible passages as [illegible: ~N words].
- Do NOT modernise or correct orthography.
- If the document is Latin, transcribe as Latin.
- Wrap your transcription in <transcription>…</transcription> tags.
- After the transcription, on a new line, write a confidence score 0.0–1.0 in
  the format: CONFIDENCE:0.XX"""

SYSTEM_QA = """You are a quality-assurance agent for historical transcriptions.
You will receive the original image description AND a transcription draft.
Evaluate the transcription and respond with JSON only:
{
  "score": 0.0–1.0,
  "issues": ["list of specific problems found"],
  "corrected_passages": {"original": "corrected"},
  "verdict": "ACCEPT | REVISE | REJECT"
}"""

SYSTEM_HTR_REFINED = """You are an expert palaeographer. A previous transcription
attempt scored below acceptable quality. You will receive the image and the
previous attempt with its issues. Produce an improved transcription.
Wrap your transcription in <transcription>…</transcription> tags and include
CONFIDENCE:0.XX at the end."""

# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class HTRResult:
    doc_id: str
    transcription: str
    confidence: float
    model_used: str
    attempts: int
    qa_verdict: str
    qa_issues: list[str] = field(default_factory=list)
    output_path: Optional[Path] = None
    preview_image_path: Optional[Path] = None


# ── Helper ─────────────────────────────────────────────────────────────────

def _extract_transcription(text: str) -> str:
    match = re.search(r"<transcription>(.*?)</transcription>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_confidence(text: str) -> float:
    match = re.search(r"CONFIDENCE:([\d.]+)", text)
    if match:
        return min(1.0, max(0.0, float(match.group(1))))
    return 0.5


def _group_files(paths: list[Path]) -> dict[str, list[Path]]:
    """Group files by document stem (everything before last numeric suffix)."""
    groups: dict[str, list[Path]] = {}
    for p in paths:
        # e.g. "missive_001_p1.jpg" → stem "missive_001"
        stem = re.sub(r"[_-]p\d+$", "", p.stem, flags=re.IGNORECASE)
        groups.setdefault(stem, []).append(p)
    for g in groups.values():
        g.sort()
    return groups


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() in config.PDF_EXTENSIONS


# ── Main Agent ─────────────────────────────────────────────────────────────

class TextRecognitionAgent:
    """Agent A – Handwriting Text Recognition."""

    name = "TextRecognitionAgent"

    def group_files(self, paths: list[Path]) -> dict[str, list[Path]]:
        return _group_files(paths)

    def _pdf_to_images(self, pdf_path: Path) -> list[Path]:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF (fitz) is required for PDF processing.") from exc

        out_dir = config.TMP_DIR / "pdf_pages" / pdf_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.png"):
            old.unlink()

        images: list[Path] = []
        with fitz.open(pdf_path) as doc:
            for idx, page in enumerate(doc):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                out_path = out_dir / f"{pdf_path.stem}_p{idx + 1}.png"
                pix.save(out_path)
                images.append(out_path)
        return images

    async def _transcribe_image(
        self,
        image_path: Path,
    ) -> tuple[str, float, int, str, list[str]]:
        attempt = 0
        last_issues: list[str] = []
        transcription = ""
        confidence = 0.0

        while attempt < config.MAX_RETRIES:
            attempt += 1

            if attempt == 1:
                user_msg = (
                    "Transcribe this historical manuscript page. "
                    "It is a medieval German or Latin administrative document."
                )
                raw = await ask(
                    SYSTEM_HTR,
                    user_msg,
                    image_path=image_path,
                    model=config.GEMINI_VISION_MODEL,
                    agent_name=self.name,
                )
            else:
                user_msg = (
                    f"Previous transcription attempt had these issues: "
                    f"{json.dumps(last_issues)}. "
                    "Please produce a corrected transcription of this manuscript page."
                )
                raw = await ask(
                    SYSTEM_HTR_REFINED,
                    user_msg,
                    image_path=image_path,
                    model=config.GEMINI_VISION_MODEL,
                    agent_name=self.name,
                )

            transcription = _extract_transcription(raw)
            confidence = _extract_confidence(raw)

            qa_prompt = (
                f"Image: {image_path.name}\n\n"
                f"Transcription draft (attempt {attempt}):\n{transcription}"
            )
            qa_raw = await ask_structured(
                SYSTEM_QA,
                qa_prompt,
                image_path=image_path,
                model=config.GEMINI_VISION_MODEL,
                agent_name=f"{self.name}:QA",
            )

            try:
                qa = json.loads(qa_raw)
            except json.JSONDecodeError:
                qa = {"score": confidence, "issues": [], "verdict": "ACCEPT",
                      "corrected_passages": {}}

            qa_score = float(qa.get("score", confidence))
            qa_verdict = qa.get("verdict", "ACCEPT")
            last_issues = qa.get("issues", [])

            for orig, corrected in qa.get("corrected_passages", {}).items():
                transcription = transcription.replace(orig, corrected)

            logger.info(
                f"[AgentA] Attempt {attempt}: confidence={confidence:.2f}, "
                f"QA score={qa_score:.2f}, verdict={qa_verdict}"
            )

            if qa_score >= config.HTR_QUALITY_THRESHOLD or qa_verdict == "ACCEPT":
                break

        return transcription, confidence, attempt, qa_verdict, last_issues

    async def process_pages(self, doc_id: str, pages: list[Path]) -> HTRResult:
        logger.info(f"[AgentA] Processing document '{doc_id}' ({len(pages)} page(s))")
        combined_parts: list[str] = []
        total_confidence = 0.0
        total_attempts = 0
        qa_verdict = "ACCEPT"
        all_issues: list[str] = []

        for page in pages:
            transcription, confidence, attempts, verdict, issues = await self._transcribe_image(page)
            combined_parts.append(f"--- Page: {page.name} ---\n\n{transcription}")
            total_confidence += confidence
            total_attempts += attempts
            all_issues.extend(issues)
            if verdict != "ACCEPT":
                qa_verdict = verdict

        combined_transcription = "\n\n".join(combined_parts)
        out_path = config.TRANSCRIPTION_DIR / f"{doc_id}.txt"
        out_path.write_text(combined_transcription, encoding="utf-8")
        logger.info(f"[AgentA] Saved transcription → {out_path}")

        avg_confidence = total_confidence / max(len(pages), 1)
        return HTRResult(
            doc_id=doc_id,
            transcription=combined_transcription,
            confidence=avg_confidence,
            model_used=config.GEMINI_VISION_MODEL,
            attempts=total_attempts,
            qa_verdict=qa_verdict,
            qa_issues=all_issues,
            output_path=out_path,
            preview_image_path=pages[0] if pages else None,
        )

    async def process_file(self, image_path: Path) -> HTRResult:
        """Process a single image or PDF file through HTR + QA pipeline."""
        doc_id = image_path.stem
        logger.info(f"[AgentA] Processing: {image_path.name}")

        if _is_pdf(image_path):
            pages = self._pdf_to_images(image_path)
            if not pages:
                raise ValueError(f"No pages extracted from PDF: {image_path}")
            return await self.process_pages(doc_id, pages)

        transcription, confidence, attempt, qa_verdict, last_issues = await self._transcribe_image(image_path)

        out_path = config.TRANSCRIPTION_DIR / f"{doc_id}.txt"
        out_path.write_text(transcription, encoding="utf-8")
        logger.info(f"[AgentA] Saved transcription → {out_path}")

        return HTRResult(
            doc_id=doc_id,
            transcription=transcription,
            confidence=confidence,
            model_used=config.GEMINI_VISION_MODEL,
            attempts=attempt,
            qa_verdict=qa_verdict,
            qa_issues=last_issues,
            output_path=out_path,
            preview_image_path=image_path,
        )

    async def process_folder(self, folder: Path) -> list[HTRResult]:
        """Process all images in a hot folder, grouped by document."""
        files = [
            p for p in folder.iterdir()
            if p.suffix.lower() in (config.IMAGE_EXTENSIONS | config.PDF_EXTENSIONS)
        ]
        if not files:
            logger.warning(f"[AgentA] No images or PDFs found in {folder}")
            return []

        pdfs = [p for p in files if _is_pdf(p)]
        images = [p for p in files if p.suffix.lower() in config.IMAGE_EXTENSIONS]
        groups = _group_files(images)
        results = []

        for doc_stem, pages in groups.items():
            if len(pages) > 1:
                results.append(await self.process_pages(doc_stem, pages))
            else:
                results.append(await self.process_file(pages[0]))

        for pdf in pdfs:
            results.append(await self.process_file(pdf))

        return results

    def format_discord_summary(self, result: HTRResult) -> str:
        preview = result.transcription[:300].replace("\n", " ")
        issues_str = ""
        if result.qa_issues:
            issues_str = "\n> ⚠️ " + "\n> ⚠️ ".join(result.qa_issues[:3])

        return (
            f"## 📜 Agent A – Text Recognition Complete\n"
            f"**Document:** `{result.doc_id}`\n"
            f"**Confidence:** {result.confidence:.0%} "
            f"| **QA Verdict:** `{result.qa_verdict}` "
            f"| **Attempts:** {result.attempts}\n"
            f"**Output:** `{result.output_path}`\n"
            f"{issues_str}\n"
            f"```\n{preview}…\n```"
        )
