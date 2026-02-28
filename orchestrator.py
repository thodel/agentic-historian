"""
orchestrator.py – Orchestrates all agents for a full document pipeline.
Can run the full pipeline or individual agents on demand.
"""
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

from loguru import logger

import config
from agents.text_recognition import TextRecognitionAgent, HTRResult
from agents.source_description import SourceDescriptionAgent, SourceDescription
from agents.entity_agent import EntityAgent, EntityResult
from agents.corpus_analysis import CorpusAnalysisAgent, CorpusAnalysisResult
from agents.meta_agent import MetaAgent, MetaReport
from knowledge_hub.hub import KnowledgeHub


@dataclass
class PipelineResult:
    doc_id: str
    htr: Optional[HTRResult] = None
    description: Optional[SourceDescription] = None
    entities: Optional[EntityResult] = None
    corpus: Optional[CorpusAnalysisResult] = None
    errors: list[str] = field(default_factory=list)


class Orchestrator:
    """Runs agents individually or as a full pipeline."""

    def __init__(self, status_callback: Optional[Callable] = None):
        self.hub        = KnowledgeHub()
        self.agent_a    = TextRecognitionAgent()
        self.agent_b    = SourceDescriptionAgent(self.hub)
        self.agent_c    = EntityAgent(self.hub)
        self.agent_d    = CorpusAnalysisAgent()
        self.agent_e    = MetaAgent()
        self._cb        = status_callback  # called with (str) for live updates

    async def _status(self, msg: str):
        logger.info(msg)
        if self._cb:
            await self._cb(msg)

    # ── Single-document pipeline ───────────────────────────────────────────

    async def _run_full_pipeline(
        self,
        doc_id: str,
        image_path: Optional[Path],
        pages: Optional[list[Path]],
        run_corpus: bool,
    ) -> PipelineResult:
        result = PipelineResult(doc_id=doc_id)
        label = image_path.name if image_path else doc_id

        await self._status(f"🔄 Starting full pipeline for `{label}`…")

        # Agent A
        t0 = time.monotonic()
        try:
            await self._status("📜 **Agent A** – Running HTR…")
            if pages:
                result.htr = await self.agent_a.process_pages(doc_id, pages)
            elif image_path:
                result.htr = await self.agent_a.process_file(image_path)
            else:
                raise ValueError("No input provided for Agent A.")
            self.agent_e.record(
                "TextRecognitionAgent", doc_id,
                duration_seconds=time.monotonic() - t0,
            )
        except Exception as e:
            result.errors.append(f"Agent A failed: {e}")
            await self._status(f"❌ Agent A error: {e}")
            return result

        preview_image = None
        if result.htr and result.htr.preview_image_path:
            preview_image = result.htr.preview_image_path
        elif image_path and image_path.suffix.lower() in config.IMAGE_EXTENSIONS:
            preview_image = image_path

        # Agent B
        t0 = time.monotonic()
        try:
            await self._status("📋 **Agent B** – Generating source description…")
            result.description = await self.agent_b.process(doc_id, preview_image)
            self.agent_e.record(
                "SourceDescriptionAgent", doc_id,
                duration_seconds=time.monotonic() - t0,
            )
        except Exception as e:
            result.errors.append(f"Agent B failed: {e}")
            await self._status(f"⚠️ Agent B error: {e}")

        # Agent C
        t0 = time.monotonic()
        try:
            await self._status("🔗 **Agent C** – Extracting and linking entities…")
            result.entities = await self.agent_c.process(doc_id)
            self.agent_e.record(
                "EntityAgent", doc_id,
                duration_seconds=time.monotonic() - t0,
            )
        except Exception as e:
            result.errors.append(f"Agent C failed: {e}")
            await self._status(f"⚠️ Agent C error: {e}")

        # Agent D (optional, corpus-level)
        if run_corpus:
            t0 = time.monotonic()
            try:
                await self._status("📊 **Agent D** – Running corpus analysis…")
                result.corpus = await self.agent_d.process()
                self.agent_e.record(
                    "CorpusAnalysisAgent", "corpus",
                    duration_seconds=time.monotonic() - t0,
                )
            except Exception as e:
                result.errors.append(f"Agent D failed: {e}")
                await self._status(f"⚠️ Agent D error: {e}")

        await self._status("✅ Pipeline complete!")
        return result

    async def run_full_pipeline(
        self,
        image_path: Path,
        run_corpus: bool = False,
    ) -> PipelineResult:
        return await self._run_full_pipeline(
            doc_id=image_path.stem,
            image_path=image_path,
            pages=None,
            run_corpus=run_corpus,
        )

    async def run_full_pipeline_group(
        self,
        doc_id: str,
        pages: list[Path],
        run_corpus: bool = False,
    ) -> PipelineResult:
        preview = pages[0] if pages else None
        return await self._run_full_pipeline(
            doc_id=doc_id,
            image_path=preview,
            pages=pages,
            run_corpus=run_corpus,
        )

    # ── Individual agents ──────────────────────────────────────────────────

    async def run_agent_a(self, image_path: Path) -> Optional[HTRResult]:
        return await self.agent_a.process_file(image_path)

    async def run_agent_b(self, doc_id: str, image_path: Optional[Path] = None):
        return await self.agent_b.process(doc_id, image_path)

    async def run_agent_c(self, doc_id: str):
        return await self.agent_c.process(doc_id)

    async def run_agent_d(
        self,
        corpus_name: str = "main",
        doc_ids: Optional[list[str]] = None,
    ):
        return await self.agent_d.process(corpus_name, doc_ids)

    async def run_agent_e(self) -> MetaReport:
        return await self.agent_e.generate_report()

    # ── Folder pipeline ────────────────────────────────────────────────────

    async def process_hot_folder(self) -> list[PipelineResult]:
        files = [
            p for p in config.HOT_FOLDER.iterdir()
            if p.is_file()
            and p.suffix.lower() in (config.IMAGE_EXTENSIONS | config.PDF_EXTENSIONS)
        ]
        if not files:
            await self._status("📂 Hot folder is empty.")
            return []

        results = []
        done_dir = config.HOT_FOLDER / "processed"
        done_dir.mkdir(exist_ok=True)

        pdfs = [p for p in files if p.suffix.lower() in config.PDF_EXTENSIONS]
        images = [p for p in files if p.suffix.lower() in config.IMAGE_EXTENSIONS]
        groups = self.agent_a.group_files(images)

        for doc_id, pages in groups.items():
            result = await self.run_full_pipeline_group(doc_id, pages)
            results.append(result)
            for page in pages:
                page.rename(done_dir / page.name)

        for pdf in pdfs:
            result = await self.run_full_pipeline(pdf)
            results.append(result)
            pdf.rename(done_dir / pdf.name)

        return results

    # ── Knowledge Hub helpers ──────────────────────────────────────────────

    def hub_summary(self) -> str:
        return self.hub.summary()

    def hub_add_person(self, person: dict):
        self.hub.add_person(person)

    def hub_add_place(self, place: dict):
        self.hub.add_place(place)

    def hub_add_keyword(self, keyword: str):
        self.hub.add_keyword(keyword)

    def hub_add_document_type(self, dtype: str):
        self.hub.add_document_type(dtype)
