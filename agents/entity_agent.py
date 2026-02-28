"""
agents/entity_agent.py – Agent C: Mentioned Entity Agent

Pipeline:
  C-a  Check for transcription
  C-b  Extract named + related entities
  C-c  Link entities to Knowledge Hub (GND, Wikidata, HLS, custom)
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp
from loguru import logger

import config
from utils.claude_client import ask_structured
from knowledge_hub.hub import KnowledgeHub

# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_ENTITY_EXTRACT = """You are a named entity recognition system specialising
in medieval German and Latin administrative documents.

Extract ALL of the following entity types:
- PERSON: individuals (also fragmentary names like "Hans von X")
- PLACE: cities, villages, regions, buildings, geographic features
- ORG: organisations, institutions, offices, guilds
- DATE: explicit dates and temporal references
- ROLE: social roles, offices, occupations (Vogt, Ritter, Dienstbot, etc.)
- SOCIAL_GROUP: collective designations (arme lüt, Juden, Zigeuner, Bürger, etc.)
- CARE_ACTOR: persons involved in care relationships (master, servant, guardian, ward)
- CARE_ACTION: care activities (versorgung, pflege, dienst, erziehung, etc.)

Respond ONLY with JSON:
{
  "entities": [
    {
      "text": "exact text as in document",
      "type": "PERSON | PLACE | ORG | DATE | ROLE | SOCIAL_GROUP | CARE_ACTOR | CARE_ACTION",
      "normalised": "modern normalised form if applicable",
      "context": "short surrounding sentence for disambiguation",
      "confidence": 0.0–1.0
    }
  ]
}"""

SYSTEM_ENTITY_LINK = """You are a historical knowledge linker. Given an entity
extracted from a medieval document and data from external knowledge bases,
propose the best match.

Respond ONLY with JSON:
{
  "gnd_id": "GND identifier or null",
  "wikidata_id": "Q-number or null",
  "hls_id": "HLS identifier or null",
  "custom_hub_id": "local KB identifier or null",
  "link_confidence": 0.0–1.0,
  "disambiguation_note": "brief note on why this match was chosen or why uncertain"
}"""

# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class Entity:
    text: str
    type: str
    normalised: str = ""
    context: str = ""
    confidence: float = 0.0
    # Linked identifiers
    gnd_id: Optional[str] = None
    wikidata_id: Optional[str] = None
    hls_id: Optional[str] = None
    custom_hub_id: Optional[str] = None
    link_confidence: float = 0.0
    disambiguation_note: str = ""


@dataclass
class EntityResult:
    doc_id: str
    entities: list[Entity] = field(default_factory=list)
    output_path: Optional[Path] = None

    @property
    def persons(self): return [e for e in self.entities if e.type == "PERSON"]
    @property
    def places(self): return [e for e in self.entities if e.type == "PLACE"]
    @property
    def social_groups(self): return [e for e in self.entities if e.type == "SOCIAL_GROUP"]
    @property
    def care_actors(self): return [e for e in self.entities if e.type == "CARE_ACTOR"]


# ── Wikidata Lookup ────────────────────────────────────────────────────────

async def _wikidata_search(name: str, entity_type: str) -> Optional[dict]:
    """Quick Wikidata search for an entity name."""
    type_filter = {
        "PERSON": "Q5",       # human
        "PLACE": "Q515",      # city (broad)
        "ORG": "Q43229",      # organisation
    }.get(entity_type)

    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "de",
        "format": "json",
        "limit": 3,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as r:
                data = await r.json()
                results = data.get("search", [])
                if results:
                    return {"id": results[0]["id"], "label": results[0].get("label", "")}
    except Exception as e:
        logger.debug(f"Wikidata lookup failed for '{name}': {e}")
    return None


# ── Main Agent ─────────────────────────────────────────────────────────────

class EntityAgent:
    """Agent C – Entity Extraction and Linking."""

    name = "EntityAgent"

    def __init__(self, knowledge_hub: KnowledgeHub):
        self.hub = knowledge_hub

    def _get_transcription(self, doc_id: str) -> Optional[str]:
        path = config.TRANSCRIPTION_DIR / f"{doc_id}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    async def process(self, doc_id: str) -> Optional[EntityResult]:
        transcription = self._get_transcription(doc_id)
        if not transcription:
            logger.warning(f"[AgentC] No transcription for '{doc_id}'")
            return None

        # ── C-b: Extract entities ──────────────────────────────────────────
        extract_prompt = (
            f"Document ID: {doc_id}\n\n"
            f"Transcription:\n{transcription[:5000]}"
        )
        raw = await ask_structured(
            SYSTEM_ENTITY_EXTRACT, extract_prompt, agent_name=self.name
        )
        data = json.loads(raw)
        entities_raw = data.get("entities", [])

        entities: list[Entity] = []
        for e in entities_raw:
            entity = Entity(
                text=e.get("text", ""),
                type=e.get("type", "UNKNOWN"),
                normalised=e.get("normalised", e.get("text", "")),
                context=e.get("context", ""),
                confidence=float(e.get("confidence", 0.5)),
            )
            entities.append(entity)

        # ── C-c: Link entities ─────────────────────────────────────────────
        hub_persons = self.hub.get_persons()
        hub_places  = self.hub.get_places()

        for entity in entities:
            # Check local Knowledge Hub first
            if entity.type == "PERSON":
                match = self.hub.find_person(entity.normalised or entity.text)
                if match:
                    entity.custom_hub_id = match.get("id")
                    entity.link_confidence = 0.9
                    continue

            if entity.type == "PLACE":
                match = self.hub.find_place(entity.normalised or entity.text)
                if match:
                    entity.custom_hub_id = match.get("id")
                    entity.link_confidence = 0.9
                    continue

            # Wikidata lookup for persons/places/orgs
            if entity.type in {"PERSON", "PLACE", "ORG"} and entity.confidence > 0.6:
                wd = await _wikidata_search(entity.normalised or entity.text, entity.type)
                if wd:
                    entity.wikidata_id = wd["id"]
                    entity.link_confidence = 0.7

            # Ask Claude for disambiguation if needed
            if entity.confidence > 0.7 and not entity.wikidata_id and not entity.custom_hub_id:
                link_prompt = (
                    f"Entity: {entity.text} (type: {entity.type})\n"
                    f"Normalised: {entity.normalised}\n"
                    f"Context: {entity.context}\n\n"
                    f"Available Hub persons: {json.dumps(hub_persons[:20], ensure_ascii=False)}\n"
                    f"Available Hub places: {json.dumps(hub_places[:20], ensure_ascii=False)}"
                )
                link_raw = await ask_structured(
                    SYSTEM_ENTITY_LINK, link_prompt,
                    agent_name=f"{self.name}:link"
                )
                try:
                    link = json.loads(link_raw)
                    entity.gnd_id           = link.get("gnd_id")
                    entity.wikidata_id      = link.get("wikidata_id") or entity.wikidata_id
                    entity.hls_id           = link.get("hls_id")
                    entity.custom_hub_id    = link.get("custom_hub_id")
                    entity.link_confidence  = float(link.get("link_confidence", 0.0))
                    entity.disambiguation_note = link.get("disambiguation_note", "")
                except Exception:
                    pass

        result = EntityResult(doc_id=doc_id, entities=entities)
        result.output_path = self._save(result, transcription)
        logger.info(
            f"[AgentC] {len(entities)} entities extracted, "
            f"saved → {result.output_path}"
        )
        return result

    def _save(self, result: EntityResult, transcription: str) -> Path:
        """Save entity data as JSON + Markdown."""
        json_path = config.OUTPUT_DIR / f"{result.doc_id}_entities.json"
        entities_dicts = [
            {
                "text": e.text, "type": e.type, "normalised": e.normalised,
                "context": e.context, "confidence": e.confidence,
                "gnd_id": e.gnd_id, "wikidata_id": e.wikidata_id,
                "hls_id": e.hls_id, "custom_hub_id": e.custom_hub_id,
                "link_confidence": e.link_confidence,
                "disambiguation_note": e.disambiguation_note,
            }
            for e in result.entities
        ]
        json_path.write_text(
            json.dumps({"doc_id": result.doc_id, "entities": entities_dicts},
                       ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # Markdown summary
        md_path = config.OUTPUT_DIR / f"{result.doc_id}_entities.md"
        lines = [f"# Entitäten: {result.doc_id}\n"]
        for etype in ["PERSON", "PLACE", "ORG", "SOCIAL_GROUP", "CARE_ACTOR", "CARE_ACTION"]:
            group = [e for e in result.entities if e.type == etype]
            if group:
                lines.append(f"\n## {etype}")
                for e in group:
                    ids = " | ".join(filter(None, [
                        f"WD:{e.wikidata_id}" if e.wikidata_id else "",
                        f"GND:{e.gnd_id}" if e.gnd_id else "",
                        f"HUB:{e.custom_hub_id}" if e.custom_hub_id else "",
                    ]))
                    lines.append(
                        f"- **{e.text}** → `{e.normalised}` "
                        f"[conf:{e.confidence:.0%}] {ids}"
                    )
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return json_path

    def format_discord_summary(self, result: EntityResult) -> str:
        persons_str = ", ".join(e.text for e in result.persons[:6])
        places_str  = ", ".join(e.text for e in result.places[:6])
        social_str  = ", ".join(e.text for e in result.social_groups[:6])
        care_str    = ", ".join(e.text for e in result.care_actors[:4])
        linked = sum(
            1 for e in result.entities
            if e.wikidata_id or e.gnd_id or e.custom_hub_id
        )
        return (
            f"## 🔗 Agent C – Entity Extraction Complete\n"
            f"**Document:** `{result.doc_id}` | "
            f"**Total entities:** {len(result.entities)} | **Linked:** {linked}\n"
            f"👤 **Persons:** {persons_str or '–'}\n"
            f"📍 **Places:** {places_str or '–'}\n"
            f"👥 **Social groups:** {social_str or '–'}\n"
            f"🏥 **Care actors:** {care_str or '–'}\n"
            f"📁 `{result.output_path}`"
        )
