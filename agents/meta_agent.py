"""
agents/meta_agent.py – Agent E: Agent-Agent (Meta Agent)

Tasks:
  E-a  Monitor agents A–D: resource usage (tokens, cost, time, storage)
  E-b  Evaluate input/output ratios per agent
  E-c  Periodically review setup and suggest code improvements
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

import config
from utils.llm_client import ask

# ── Cost estimates (rough, Gemini Pro) ────────────────────────────────────
# Prices per 1M tokens (adjust to your actual model/tier)
COST_PER_1M_INPUT  = 15.00   # USD
COST_PER_1M_OUTPUT = 75.00   # USD


# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class AgentRun:
    agent_name: str
    doc_id: str
    timestamp: float = field(default_factory=time.time)
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * COST_PER_1M_INPUT / 1_000_000
            + self.output_tokens * COST_PER_1M_OUTPUT / 1_000_000
        )


@dataclass
class MetaReport:
    generated_at: str
    total_runs: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    total_duration_seconds: float
    per_agent_stats: dict
    storage_mb: float
    suggestions: list[str] = field(default_factory=list)
    improvement_plan: str = ""


# ── Meta Agent ─────────────────────────────────────────────────────────────

class MetaAgent:
    """Agent E – Monitors all other agents, tracks resources, suggests improvements."""

    name = "MetaAgent"
    _log_path = config.OUTPUT_DIR / "meta_agent_log.json"

    def __init__(self):
        self._runs: list[AgentRun] = self._load_log()

    def _load_log(self) -> list[AgentRun]:
        if self._log_path.exists():
            try:
                data = json.loads(self._log_path.read_text(encoding="utf-8"))
                return [AgentRun(**r) for r in data]
            except Exception:
                pass
        return []

    def _save_log(self):
        entries = [
            {
                "agent_name": r.agent_name,
                "doc_id": r.doc_id,
                "timestamp": r.timestamp,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "duration_seconds": r.duration_seconds,
                "success": r.success,
                "error": r.error,
            }
            for r in self._runs[-500:]  # keep last 500 runs
        ]
        self._log_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def record(
        self,
        agent_name: str,
        doc_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_seconds: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ):
        run = AgentRun(
            agent_name=agent_name,
            doc_id=doc_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_seconds=duration_seconds,
            success=success,
            error=error,
        )
        self._runs.append(run)
        self._save_log()

    def _compute_storage_mb(self) -> float:
        total = 0
        for d in [config.TRANSCRIPTION_DIR, config.DESCRIPTION_DIR,
                  config.OUTPUT_DIR, config.CORPUS_DIR]:
            if d.exists():
                total += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        return total / (1024 * 1024)

    def _per_agent_stats(self) -> dict:
        stats = {}
        for run in self._runs:
            if run.agent_name not in stats:
                stats[run.agent_name] = {
                    "runs": 0, "input_tokens": 0, "output_tokens": 0,
                    "cost_usd": 0.0, "errors": 0, "total_duration": 0.0,
                }
            s = stats[run.agent_name]
            s["runs"] += 1
            s["input_tokens"]  += run.input_tokens
            s["output_tokens"] += run.output_tokens
            s["cost_usd"]      += run.cost_usd
            s["total_duration"] += run.duration_seconds
            if not run.success:
                s["errors"] += 1
        return stats

    async def generate_report(self) -> MetaReport:
        """Generate a comprehensive status and resource report."""
        per_agent = self._per_agent_stats()
        storage   = self._compute_storage_mb()

        total_input  = sum(r.input_tokens for r in self._runs)
        total_output = sum(r.output_tokens for r in self._runs)
        total_cost   = sum(r.cost_usd for r in self._runs)
        total_dur    = sum(r.duration_seconds for r in self._runs)

        # Ask Gemini to suggest improvements
        context = (
            f"Agentic Historian system status:\n"
            f"Total agent runs: {len(self._runs)}\n"
            f"Total tokens used: {total_input + total_output:,}\n"
            f"Total cost: ${total_cost:.4f}\n"
            f"Storage used: {storage:.1f} MB\n"
            f"Per-agent stats: {json.dumps(per_agent, indent=2)}\n\n"
            f"Recent errors: {[r.error for r in self._runs[-20:] if r.error]}\n\n"
            "Based on this data, suggest 3–7 concrete improvements to the pipeline. "
            "Consider: token efficiency, error patterns, model selection, "
            "agent ordering, caching opportunities. "
            "Also suggest if any processes should run differently or less/more often. "
            "Respond in German, in prose."
        )

        suggestions_text = await ask(
            "Du bist der Meta-Agent des Agentic Historian Systems. "
            "Analysiere den Systemstatus und gib konkrete Verbesserungsvorschläge.",
            context,
            agent_name=self.name,
        )

        # Parse suggestions into list
        suggestions = [
            line.strip().lstrip("•-–*").strip()
            for line in suggestions_text.split("\n")
            if line.strip() and len(line.strip()) > 20
        ][:8]

        report = MetaReport(
            generated_at=datetime.now().isoformat(),
            total_runs=len(self._runs),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cost_usd=total_cost,
            total_duration_seconds=total_dur,
            per_agent_stats=per_agent,
            storage_mb=storage,
            suggestions=suggestions,
            improvement_plan=suggestions_text,
        )

        self._save_report(report)
        return report

    def _save_report(self, report: MetaReport):
        out = config.OUTPUT_DIR / "meta_report.md"
        agent_rows = "\n".join(
            f"| {name} | {s['runs']} | {s['input_tokens']:,} | "
            f"{s['output_tokens']:,} | ${s['cost_usd']:.4f} | {s['errors']} |"
            for name, s in report.per_agent_stats.items()
        )
        suggestions_md = "\n".join(f"- {s}" for s in report.suggestions)

        md = f"""# Meta-Agent Report
**Generiert:** {report.generated_at}

## Ressourcenübersicht
| Metrik | Wert |
|---|---|
| Agent-Runs gesamt | {report.total_runs} |
| Input-Tokens | {report.total_input_tokens:,} |
| Output-Tokens | {report.total_output_tokens:,} |
| Geschätzte Kosten | ${report.total_cost_usd:.4f} |
| Gesamtdauer | {report.total_duration_seconds:.0f}s |
| Storage | {report.storage_mb:.1f} MB |

## Agenten-Statistik
| Agent | Runs | Input-Tokens | Output-Tokens | Kosten | Fehler |
|---|---|---|---|---|---|
{agent_rows}

## Verbesserungsvorschläge
{suggestions_md}

## Ausführlicher Verbesserungsplan
{report.improvement_plan}
"""
        out.write_text(md, encoding="utf-8")
        logger.info(f"[AgentE] Meta report saved → {out}")

    def format_discord_summary(self, report: MetaReport) -> str:
        per_agent_str = "\n".join(
            f"> • **{name}**: {s['runs']} runs, "
            f"{s['input_tokens']+s['output_tokens']:,} tokens, "
            f"${s['cost_usd']:.4f}"
            for name, s in report.per_agent_stats.items()
        )
        suggestions_str = "\n".join(
            f"> {i+1}. {s[:100]}" for i, s in enumerate(report.suggestions[:4])
        )
        return (
            f"## 🤖 Agent E – Meta Report\n"
            f"**Runs:** {report.total_runs} | "
            f"**Tokens:** {report.total_input_tokens + report.total_output_tokens:,} | "
            f"**Cost:** ${report.total_cost_usd:.4f} | "
            f"**Storage:** {report.storage_mb:.1f} MB\n\n"
            f"**Per-Agent:**\n{per_agent_str}\n\n"
            f"**Improvement suggestions:**\n{suggestions_str}"
        )
