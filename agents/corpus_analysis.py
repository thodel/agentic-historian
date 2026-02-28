"""
agents/corpus_analysis.py – Agent D: Corpus Analysis Agent

Pipeline:
  D-a  Check for transcription(s)
  D-b  Run corpus analysis → static output (freq, topics, taxonomy, care terms)
  D-c  Build Voyant Tools link from GitHub-hosted corpus
"""
import json
import re
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

import config
from utils.claude_client import ask, ask_structured

# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_TOPIC_MODEL = """You are a digital humanities specialist performing
topic modelling on medieval administrative documents (corpus).
Given the corpus text, identify 5–10 coherent topics.

Respond ONLY with JSON:
{
  "topics": [
    {
      "label": "topic name in German",
      "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
      "doc_ids": ["doc_ids where this topic appears"],
      "description": "brief description of this topic"
    }
  ]
}"""

SYSTEM_TAXONOMY_ANALYSIS = """You are a historian analysing social taxonomies in
medieval documents. Given the corpus, identify patterns in how social groups
are categorised and treated.

Respond ONLY with JSON:
{
  "taxonomy_patterns": [
    {
      "term": "social term (e.g. arme lüt, Juden, Bürger)",
      "frequency": 0,
      "associated_administrative_acts": ["actions linked to this group"],
      "treatment_pattern": "description of how this group is addressed by authorities",
      "intersectionalities": ["other categories that co-occur with this term"]
    }
  ],
  "dominant_taxonomies": ["most prominent categorisation principles"],
  "temporal_trends": "any observable changes if multiple time periods present"
}"""

SYSTEM_CARE_ANALYSIS = """You are a historian specialising in pre-modern care
practices. Analyse the corpus for care-related content.

Respond ONLY with JSON:
{
  "care_instances": [
    {
      "doc_id": "document identifier",
      "care_type": "domestic | institutional | familial | wage-based | community",
      "actors": {"provider": "name/role", "recipient": "name/role"},
      "arrangements": "description of the care arrangement",
      "compensation": "how care was remunerated or exchanged",
      "conflict_involved": true
    }
  ],
  "care_economy_patterns": "overall patterns in care as economic exchange",
  "gender_patterns": "observations about gendered dimensions of care"
}"""

# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class CorpusStats:
    total_docs: int
    total_tokens: int
    unique_tokens: int
    top_terms: list[tuple[str, int]] = field(default_factory=list)
    doc_lengths: dict[str, int] = field(default_factory=dict)


@dataclass
class CorpusAnalysisResult:
    corpus_name: str
    stats: CorpusStats
    topics: list[dict] = field(default_factory=list)
    taxonomy_patterns: list[dict] = field(default_factory=list)
    care_instances: list[dict] = field(default_factory=list)
    care_economy_patterns: str = ""
    voyant_url: Optional[str] = None
    output_dir: Optional[Path] = None


# ── Helpers ────────────────────────────────────────────────────────────────

_STOPWORDS_DE = {
    "und", "die", "der", "das", "ist", "in", "zu", "von", "den", "dem",
    "des", "ein", "eine", "auf", "mit", "er", "sie", "wir", "daz", "daz",
    "vnd", "als", "sich", "ir", "an", "ze", "so", "ouch", "aber", "noch",
    "sin", "hat", "sol", "han", "wol", "also", "umb", "nach", "bi", "vor",
    "et", "est", "non", "in", "ad", "de", "per", "qui", "quod", "ut",
}

def _tokenise(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-ZäöüÄÖÜß]{3,}\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS_DE]

def _compute_stats(texts: dict[str, str]) -> CorpusStats:
    all_tokens = []
    doc_lengths = {}
    for doc_id, text in texts.items():
        tokens = _tokenise(text)
        doc_lengths[doc_id] = len(tokens)
        all_tokens.extend(tokens)
    counter = Counter(all_tokens)
    return CorpusStats(
        total_docs=len(texts),
        total_tokens=len(all_tokens),
        unique_tokens=len(counter),
        top_terms=counter.most_common(30),
        doc_lengths=doc_lengths,
    )

def _build_voyant_url(corpus_text: str, corpus_name: str) -> str:
    """Build a Voyant Tools URL with corpus text embedded."""
    encoded = urllib.parse.quote(corpus_text[:50000])  # Voyant limit
    base = config.VOYANT_URL.rstrip("/")
    return f"{base}/?input={encoded}&corpus={urllib.parse.quote(corpus_name)}"


# ── Main Agent ─────────────────────────────────────────────────────────────

class CorpusAnalysisAgent:
    """Agent D – Corpus Analysis."""

    name = "CorpusAnalysisAgent"

    def _load_corpus(self, doc_ids: Optional[list[str]] = None) -> dict[str, str]:
        """Load transcriptions. If doc_ids is None, load all available."""
        texts = {}
        paths = list(config.TRANSCRIPTION_DIR.glob("*.txt"))
        for p in paths:
            doc_id = p.stem
            if doc_ids and doc_id not in doc_ids:
                continue
            texts[doc_id] = p.read_text(encoding="utf-8")
        return texts

    async def process(
        self,
        corpus_name: str = "main",
        doc_ids: Optional[list[str]] = None,
    ) -> Optional[CorpusAnalysisResult]:
        texts = self._load_corpus(doc_ids)
        if not texts:
            logger.warning("[AgentD] No transcriptions available for corpus analysis")
            return None

        logger.info(f"[AgentD] Analysing corpus '{corpus_name}' ({len(texts)} docs)")

        # ── D-b: Statistical analysis ──────────────────────────────────────
        stats = _compute_stats(texts)

        combined = "\n\n---\n\n".join(
            f"[{doc_id}]\n{text}" for doc_id, text in list(texts.items())[:20]
        )

        # Topic modelling
        topic_raw = await ask_structured(
            SYSTEM_TOPIC_MODEL,
            f"Corpus '{corpus_name}' ({len(texts)} documents):\n\n{combined[:8000]}",
            agent_name=f"{self.name}:topics",
        )
        topics = json.loads(topic_raw).get("topics", [])

        # Social taxonomy analysis
        taxonomy_raw = await ask_structured(
            SYSTEM_TAXONOMY_ANALYSIS,
            f"Corpus '{corpus_name}':\n\n{combined[:8000]}",
            agent_name=f"{self.name}:taxonomy",
        )
        taxonomy_data = json.loads(taxonomy_raw)

        # Care analysis
        care_raw = await ask_structured(
            SYSTEM_CARE_ANALYSIS,
            f"Corpus '{corpus_name}':\n\n{combined[:8000]}",
            agent_name=f"{self.name}:care",
        )
        care_data = json.loads(care_raw)

        # ── D-c: Voyant Tools link ─────────────────────────────────────────
        voyant_url = _build_voyant_url(combined, corpus_name)

        result = CorpusAnalysisResult(
            corpus_name=corpus_name,
            stats=stats,
            topics=topics,
            taxonomy_patterns=taxonomy_data.get("taxonomy_patterns", []),
            care_instances=care_data.get("care_instances", []),
            care_economy_patterns=care_data.get("care_economy_patterns", ""),
            voyant_url=voyant_url,
        )

        result.output_dir = self._save(result, texts)
        logger.info(f"[AgentD] Corpus analysis saved → {result.output_dir}")
        return result

    def _save(self, result: CorpusAnalysisResult, texts: dict) -> Path:
        out_dir = config.OUTPUT_DIR / f"corpus_{result.corpus_name}"
        out_dir.mkdir(exist_ok=True)

        # Save raw corpus for GitHub/Voyant
        corpus_path = out_dir / "corpus.txt"
        corpus_path.write_text(
            "\n\n---\n\n".join(f"[{k}]\n{v}" for k, v in texts.items()),
            encoding="utf-8"
        )

        # Stats JSON
        stats_path = out_dir / "stats.json"
        stats_path.write_text(json.dumps({
            "total_docs": result.stats.total_docs,
            "total_tokens": result.stats.total_tokens,
            "unique_tokens": result.stats.unique_tokens,
            "top_terms": result.stats.top_terms,
            "doc_lengths": result.stats.doc_lengths,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        # Topics JSON
        (out_dir / "topics.json").write_text(
            json.dumps(result.topics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Taxonomy JSON
        (out_dir / "taxonomy.json").write_text(
            json.dumps(result.taxonomy_patterns, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # Care JSON
        (out_dir / "care_analysis.json").write_text(
            json.dumps({
                "instances": result.care_instances,
                "economy_patterns": result.care_economy_patterns,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # Markdown report
        top_terms_str = " | ".join(f"{t} ({n})" for t, n in result.stats.top_terms[:15])
        topics_str = "\n".join(
            f"- **{t['label']}**: {', '.join(t['keywords'][:5])}"
            for t in result.topics
        )
        taxonomy_str = "\n".join(
            f"- **{p['term']}** (n={p.get('frequency', '?')}): "
            f"{p.get('treatment_pattern', '')[:100]}"
            for p in result.taxonomy_patterns[:10]
        )

        md = f"""# Korpusanalyse: {result.corpus_name}

## Statistik
| | |
|---|---|
| Dokumente | {result.stats.total_docs} |
| Token gesamt | {result.stats.total_tokens:,} |
| Unique Token | {result.stats.unique_tokens:,} |

**Häufigste Begriffe:** {top_terms_str}

## Topics
{topics_str}

## Soziale Taxonomien
{taxonomy_str}

## Care-Analyse
**Instanzen:** {len(result.care_instances)}
**Muster:** {result.care_economy_patterns[:300]}

## Voyant Tools
[→ Corpus in Voyant öffnen]({result.voyant_url})

---
*Generiert durch Agentic Historian – Agent D*
"""
        (out_dir / "report.md").write_text(md, encoding="utf-8")
        result.output_dir = out_dir
        return out_dir

    def format_discord_summary(self, result: CorpusAnalysisResult) -> str:
        topics_str = "\n".join(
            f"> • **{t['label']}**: {', '.join(t['keywords'][:4])}"
            for t in result.topics[:5]
        )
        return (
            f"## 📊 Agent D – Corpus Analysis Complete\n"
            f"**Corpus:** `{result.corpus_name}` | "
            f"**Docs:** {result.stats.total_docs} | "
            f"**Tokens:** {result.stats.total_tokens:,}\n"
            f"\n**Topics identified:**\n{topics_str}\n\n"
            f"**Care instances:** {len(result.care_instances)}\n"
            f"🔗 [Open in Voyant Tools]({result.voyant_url})\n"
            f"📁 `{result.output_dir}`"
        )
