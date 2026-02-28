"""
agents/source_description.py – Agent B: Source Description

Pipeline:
  B-a  Check for transcription in A-c output
  B-b  Classify document type via Knowledge Hub
  B-c  Generate content description (keywords + summary)
  B-d  Describe visual appearance via vision model (script type)
  B-e  Save full Markdown metadata record
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

import config
from utils.claude_client import ask, ask_structured
from knowledge_hub.hub import KnowledgeHub

# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_CLASSIFIER = """You are an archivist specialising in medieval administrative
documents (13th–15th century German-speaking regions). Given a transcription and
the available document types from a Knowledge Hub, classify the document.

Respond ONLY with JSON:
{
  "document_type": "one of the types from the hub",
  "language": "German | Latin | Mixed | French | Other",
  "script_period": "early 14th c. | mid 14th c. | late 14th c. | early 15th c. | mid 15th c. | other",
  "confidence": 0.0–1.0,
  "reasoning": "brief explanation"
}"""

SYSTEM_CONTENT = """You are a historian specialising in late medieval German
administrative culture. Analyse the transcription and generate structured metadata.

Respond ONLY with JSON:
{
  "summary": "2–4 sentence summary in German",
  "keywords": ["list", "of", "controlled", "vocabulary", "terms"],
  "persons_mentioned": ["name1", "name2"],
  "places_mentioned": ["place1", "place2"],
  "dates_mentioned": ["date1"],
  "administrative_acts": ["type of administrative action described"],
  "social_taxonomy_terms": ["social group terms used in the document"],
  "care_related": true | false,
  "care_context": "brief description if care-related, else null"
}"""

SYSTEM_VISUAL = """You are a palaeographer examining a medieval manuscript image.
Describe the visual features of this document.

Respond ONLY with JSON:
{
  "script_type": "e.g. Kanzleikursive | Textualis | Bastarda | Humanistica",
  "ink_condition": "good | faded | damaged | stained",
  "layout": "single column | double column | table | letter format | other",
  "seal_present": true | false,
  "marginalia_present": true | false,
  "estimated_hands": 1,
  "special_features": ["any notable features like rubrics, initials, stamps"]
}"""

SYSTEM_FORMAL = """You describe the formal/linguistic properties of a historical
text. Respond ONLY with JSON:
{
  "primary_language": "German | Latin | Mixed",
  "dialect_region": "e.g. Alemannic | Bavarian | Central German | Latin",
  "formulaic_openings": ["opening phrases used"],
  "formulaic_closings": ["closing phrases used"],
  "notable_vocabulary": ["unusual or domain-specific terms worth noting"]
}"""

# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class SourceDescription:
    doc_id: str
    document_type: str
    language: str
    script_period: str
    summary: str
    keywords: list[str] = field(default_factory=list)
    persons_mentioned: list[str] = field(default_factory=list)
    places_mentioned: list[str] = field(default_factory=list)
    dates_mentioned: list[str] = field(default_factory=list)
    administrative_acts: list[str] = field(default_factory=list)
    social_taxonomy_terms: list[str] = field(default_factory=list)
    care_related: bool = False
    care_context: Optional[str] = None
    script_type: str = "unknown"
    ink_condition: str = "unknown"
    layout: str = "unknown"
    seal_present: bool = False
    primary_language: str = "unknown"
    dialect_region: str = "unknown"
    notable_vocabulary: list[str] = field(default_factory=list)
    output_path: Optional[Path] = None


# ── Main Agent ─────────────────────────────────────────────────────────────

class SourceDescriptionAgent:
    """Agent B – Source Description and Metadata Generation."""

    name = "SourceDescriptionAgent"

    def __init__(self, knowledge_hub: KnowledgeHub):
        self.hub = knowledge_hub

    def _get_transcription(self, doc_id: str) -> Optional[str]:
        path = config.TRANSCRIPTION_DIR / f"{doc_id}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    async def process(
        self,
        doc_id: str,
        image_path: Optional[Path] = None,
    ) -> Optional[SourceDescription]:
        transcription = self._get_transcription(doc_id)
        if not transcription:
            logger.warning(f"[AgentB] No transcription found for '{doc_id}'")
            return None

        hub_types = self.hub.get_document_types()
        hub_keywords = self.hub.get_controlled_vocabulary()

        # ── Step B-b: Classify ─────────────────────────────────────────────
        classify_prompt = (
            f"Available document types from Knowledge Hub:\n"
            f"{json.dumps(hub_types, ensure_ascii=False)}\n\n"
            f"Transcription:\n{transcription[:3000]}"
        )
        cls_raw = await ask_structured(
            SYSTEM_CLASSIFIER, classify_prompt, agent_name=f"{self.name}:classify"
        )
        cls = json.loads(cls_raw)

        # ── Step B-c: Content description ──────────────────────────────────
        content_prompt = (
            f"Controlled vocabulary from Knowledge Hub:\n"
            f"{json.dumps(hub_keywords, ensure_ascii=False)}\n\n"
            f"Document type: {cls.get('document_type')}\n\n"
            f"Transcription:\n{transcription[:4000]}"
        )
        content_raw = await ask_structured(
            SYSTEM_CONTENT, content_prompt, agent_name=f"{self.name}:content"
        )
        content = json.loads(content_raw)

        # ── Step B-d: Visual description (requires image) ──────────────────
        visual = {}
        formal = {}
        if (
            image_path
            and image_path.exists()
            and image_path.suffix.lower() in config.IMAGE_EXTENSIONS
        ):
            visual_raw = await ask_structured(
                SYSTEM_VISUAL, "Describe the visual properties of this manuscript.",
                image_path=image_path,
                model=config.CLAUDE_VISION_MODEL,
                agent_name=f"{self.name}:visual",
            )
            visual = json.loads(visual_raw)

        # ── Formal / linguistic analysis ───────────────────────────────────
        formal_raw = await ask_structured(
            SYSTEM_FORMAL,
            f"Transcription:\n{transcription[:3000]}",
            agent_name=f"{self.name}:formal",
        )
        formal = json.loads(formal_raw)

        # ── Assemble description ───────────────────────────────────────────
        desc = SourceDescription(
            doc_id=doc_id,
            document_type=cls.get("document_type", "unknown"),
            language=cls.get("language", "unknown"),
            script_period=cls.get("script_period", "unknown"),
            summary=content.get("summary", ""),
            keywords=content.get("keywords", []),
            persons_mentioned=content.get("persons_mentioned", []),
            places_mentioned=content.get("places_mentioned", []),
            dates_mentioned=content.get("dates_mentioned", []),
            administrative_acts=content.get("administrative_acts", []),
            social_taxonomy_terms=content.get("social_taxonomy_terms", []),
            care_related=content.get("care_related", False),
            care_context=content.get("care_context"),
            script_type=visual.get("script_type", "unknown"),
            ink_condition=visual.get("ink_condition", "unknown"),
            layout=visual.get("layout", "unknown"),
            seal_present=visual.get("seal_present", False),
            primary_language=formal.get("primary_language", "unknown"),
            dialect_region=formal.get("dialect_region", "unknown"),
            notable_vocabulary=formal.get("notable_vocabulary", []),
        )

        # ── Step B-e: Save Markdown ────────────────────────────────────────
        desc.output_path = self._save_markdown(desc, transcription)
        logger.info(f"[AgentB] Saved description → {desc.output_path}")
        return desc

    def _save_markdown(self, desc: SourceDescription, transcription: str) -> Path:
        kw_str  = ", ".join(desc.keywords)
        per_str = ", ".join(desc.persons_mentioned)
        pla_str = ", ".join(desc.places_mentioned)
        voc_str = ", ".join(desc.notable_vocabulary)
        soc_str = ", ".join(desc.social_taxonomy_terms)
        act_str = ", ".join(desc.administrative_acts)

        care_section = ""
        if desc.care_related:
            care_section = (
                f"\n## Care-Relevanz\n"
                f"**Care-Kontext:** {desc.care_context or 'Ja'}\n"
            )

        md = f"""# Quellenerschliessung: {desc.doc_id}

## Formale Beschreibung
| Eigenschaft | Wert |
|---|---|
| Dokumenttyp | {desc.document_type} |
| Sprache | {desc.language} / {desc.primary_language} |
| Dialekt | {desc.dialect_region} |
| Schriftperiode | {desc.script_period} |
| Schrifttyp | {desc.script_type} |
| Layout | {desc.layout} |
| Tintenqualität | {desc.ink_condition} |
| Siegel vorhanden | {'Ja' if desc.seal_present else 'Nein'} |

## Inhaltliche Beschreibung
**Zusammenfassung:**
{desc.summary}

**Schlagworte:** {kw_str}

**Erwähnte Personen:** {per_str or '–'}

**Erwähnte Orte:** {pla_str or '–'}

**Erwähnte Daten:** {', '.join(desc.dates_mentioned) or '–'}

**Verwaltungshandlungen:** {act_str or '–'}

## Sprache und Diskurs
**Soziale Taxonomien:** {soc_str or '–'}

**Bemerkenswertes Vokabular:** {voc_str or '–'}
{care_section}
## Transkription (Auszug)
```
{transcription[:500]}…
```

---
*Generiert durch Agentic Historian – Agent B*
"""
        out_path = config.DESCRIPTION_DIR / f"{desc.doc_id}.md"
        out_path.write_text(md, encoding="utf-8")
        return out_path

    def format_discord_summary(self, desc: SourceDescription) -> str:
        care_flag = "🏥 Care-relevant!" if desc.care_related else ""
        return (
            f"## 📋 Agent B – Source Description Complete {care_flag}\n"
            f"**Document:** `{desc.doc_id}`\n"
            f"**Type:** {desc.document_type} | **Lang:** {desc.language} | "
            f"**Script:** {desc.script_type}\n"
            f"**Summary:** {desc.summary[:200]}\n"
            f"**Keywords:** {', '.join(desc.keywords[:8])}\n"
            f"**Persons:** {', '.join(desc.persons_mentioned[:5]) or '–'}\n"
            f"**Social taxonomy terms:** {', '.join(desc.social_taxonomy_terms[:5]) or '–'}\n"
            f"📁 `{desc.output_path}`"
        )
