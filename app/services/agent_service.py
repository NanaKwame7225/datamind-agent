"""
DataMind Agent — Multi-Agent Analysis Engine

Each notebook cell runs a panel of specialist agents over the data, then a
synthesizer merges their findings into one elite answer.

Agents:
  • data_quality — completeness, outliers, whether the data supports the question
  • trends       — movements over time, correlations, segments
  • risk         — anomalies, concerns, what to watch
  • synthesizer  — reads the other three, resolves disagreement, writes the answer

Design notes:
  - Specialists run in PARALLEL (asyncio.gather) so 3 calls cost ~1 call of time.
  - Question-awareness: the question is classified so the most relevant specialist
    is emphasised in the synthesis (a forecasting question leans on trends, a
    "what's wrong with this data" question leans on data_quality).
  - Graceful degradation: if a specialist fails, its slot is marked unavailable
    and synthesis proceeds with whoever succeeded. A cell never hard-fails just
    because one agent errored.
"""
from __future__ import annotations
import asyncio, json, logging

logger = logging.getLogger(__name__)


AGENTS = {
    "data_quality": {
        "label": "Data Quality",
        "icon": "shield",
        "system": (
            "You are the Data Quality specialist on an analysis panel. Examine the "
            "dataset for completeness, missing values, outliers, duplicates, "
            "inconsistent types, and whether the data can actually support the user's "
            "question. Be concise and specific — cite columns and counts. Flag anything "
            "that would make downstream conclusions unreliable. 3-5 tight findings."
        ),
    },
    "trends": {
        "label": "Trends & Patterns",
        "icon": "trending",
        "system": (
            "You are the Trends & Patterns specialist on an analysis panel. Identify "
            "movements over time, growth/decline, correlations, seasonality, and "
            "meaningful segments. Quantify with real numbers from the data (deltas, "
            "percentages, rates). Focus on what is changing and how fast. 3-5 findings."
        ),
    },
    "risk": {
        "label": "Risk & Anomaly",
        "icon": "alert",
        "system": (
            "You are the Risk & Anomaly specialist on an analysis panel. Surface what is "
            "concerning: anomalies, outliers that break the pattern, concentration risk, "
            "sudden shifts, and things the user should watch or act on. Be direct about "
            "severity. Avoid false alarms — only flag what the data supports. 3-5 findings."
        ),
    },
}

SYNTH_SYSTEM = (
    "You are the lead analyst synthesising a panel of specialist findings into ONE "
    "authoritative answer for a business user. You are given the user's question, a "
    "data summary, and the specialists' findings (data quality, trends, risk). "
    "Write a clear, confident, elite answer that directly answers the question. "
    "Resolve any disagreement between specialists explicitly. Lead with the direct "
    "answer, then the key supporting points, then any caveats or risks worth noting. "
    "Do NOT mention 'agents' or 'specialists' or the panel mechanism — speak as one "
    "expert. Professional, precise, no filler."
)


def classify_question(q: str) -> str:
    """Which specialist should the synthesis lean on? Cheap keyword heuristic."""
    ql = (q or "").lower()
    if any(w in ql for w in ("missing", "clean", "quality", "complete", "valid", "duplicate", "error")):
        return "data_quality"
    if any(w in ql for w in ("risk", "anomal", "concern", "fraud", "outlier", "unusual", "watch", "problem")):
        return "risk"
    if any(w in ql for w in ("trend", "forecast", "grow", "predict", "over time", "season", "correlat", "segment", "compare")):
        return "trends"
    return "trends"  # default emphasis


class AgentService:

    def _data_summary(self, data: list, columns: list = None) -> str:
        """Compact textual summary of the data for the agents (keeps tokens sane)."""
        if not data:
            return "No data provided."
        cols = columns or (list(data[0].keys()) if isinstance(data[0], dict) else [])
        n = len(data)
        sample = data[:15]
        try:
            import statistics
            numeric_cols = {}
            for c in cols:
                vals = []
                for row in data:
                    v = row.get(c) if isinstance(row, dict) else None
                    if isinstance(v, (int, float)):
                        vals.append(v)
                if len(vals) >= 3:
                    numeric_cols[c] = {
                        "min": min(vals), "max": max(vals),
                        "mean": round(statistics.mean(vals), 2),
                        "count": len(vals),
                    }
        except Exception:
            numeric_cols = {}
        lines = [f"Rows: {n}", f"Columns ({len(cols)}): {', '.join(map(str, cols))}"]
        if numeric_cols:
            lines.append("Numeric summaries:")
            for c, s in list(numeric_cols.items())[:12]:
                lines.append(f"  {c}: min={s['min']} max={s['max']} mean={s['mean']} n={s['count']}")
        lines.append("Sample rows (first 15):")
        lines.append(json.dumps(sample, default=str)[:2500])
        return "\n".join(lines)

    async def _run_agent(self, agent_key: str, question: str, data_summary: str, industry: str) -> dict:
        """Run a single specialist. Never raises — returns a status dict."""
        spec = AGENTS[agent_key]
        messages = [{
            "role": "user",
            "content": (
                f"User question: {question}\n\n"
                f"Data summary:\n{data_summary}\n\n"
                f"Give your specialist findings."
            ),
        }]
        try:
            text, tokens, provider = await self._chat_with_system(spec["system"], messages, industry)
            return {"agent": agent_key, "label": spec["label"], "icon": spec["icon"],
                    "ok": True, "findings": text, "provider": provider, "tokens": tokens}
        except Exception as e:
            logger.warning(f"Agent {agent_key} failed: {e}")
            return {"agent": agent_key, "label": spec["label"], "icon": spec["icon"],
                    "ok": False, "findings": None, "error": str(e)[:200]}

    async def _chat_with_system(self, agent_system: str, messages: list, industry: str):
        """Call the LLM chat with the agent's persona prepended to the first message."""
        from app.services.elite_llm_service import elite_llm_service, LLMProvider
        # Prepend the agent persona to the user message so it steers this call
        framed = [{"role": "user", "content": agent_system + "\n\n" + messages[0]["content"]}]
        text, tokens, provider = await elite_llm_service.chat(
            messages=framed, industry=industry, provider=LLMProvider.anthropic,
            max_tokens=900, temperature=0.1)
        return text, tokens, provider

    async def analyze(self, question: str, data: list, columns: list = None,
                      industry: str = "general") -> dict:
        """
        Run the full panel: specialists in parallel, then synthesis.
        Returns the elite answer plus each agent's view and metadata.
        """
        if not question or not (question or "").strip():
            return {"success": False, "error": "Ask a question for this cell."}
        if not data:
            return {"success": False, "error": "This cell has no data to analyse."}

        data_summary = self._data_summary(data, columns)
        emphasis = classify_question(question)

        # 1. Run the three specialists in parallel
        specialist_keys = list(AGENTS.keys())
        results = await asyncio.gather(*[
            self._run_agent(k, question, data_summary, industry) for k in specialist_keys
        ])
        agents_out = {r["agent"]: r for r in results}
        succeeded = [r for r in results if r["ok"]]

        if not succeeded:
            return {"success": False,
                    "error": "All analysis agents were unavailable — the AI providers may be busy. Try again in a moment.",
                    "agents": results}

        # 2. Synthesise
        findings_block = "\n\n".join(
            f"### {AGENTS[r['agent']]['label']}\n{r['findings']}"
            for r in succeeded
        )
        emphasis_note = ""
        if emphasis in AGENTS and agents_out.get(emphasis, {}).get("ok"):
            emphasis_note = f"\n\nThe question most concerns {AGENTS[emphasis]['label']} — weight that lens."
        synth_msg = [{
            "role": "user",
            "content": (
                f"{SYNTH_SYSTEM}\n\n"
                f"User question: {question}\n\n"
                f"Data summary:\n{data_summary}\n\n"
                f"Specialist findings:\n{findings_block}{emphasis_note}\n\n"
                f"Write the final authoritative answer now."
            ),
        }]
        try:
            from app.services.elite_llm_service import elite_llm_service, LLMProvider
            answer, s_tokens, s_provider = await elite_llm_service.chat(
                messages=synth_msg, industry=industry, provider=LLMProvider.anthropic,
                max_tokens=1400, temperature=0.15)
        except Exception as e:
            # Fall back to the strongest single specialist if synthesis fails
            logger.warning(f"Synthesis failed, falling back: {e}")
            best = agents_out.get(emphasis) if agents_out.get(emphasis, {}).get("ok") else succeeded[0]
            answer = best["findings"]
            s_tokens, s_provider = 0, best.get("provider", "fallback")

        total_tokens = sum(r.get("tokens", 0) for r in succeeded) + (s_tokens or 0)
        return {
            "success": True,
            "answer": answer,
            "emphasis": emphasis,
            "agents": [
                {"agent": r["agent"], "label": r["label"], "icon": r["icon"],
                 "ok": r["ok"], "findings": r.get("findings"), "error": r.get("error")}
                for r in results
            ],
            "provider": s_provider,
            "tokens": total_tokens,
            "agents_ok": len(succeeded),
            "agents_total": len(results),
        }


agent_service = AgentService()
