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
            "You are the Data Quality specialist. The user asked a SPECIFIC question — "
            "your job is to assess whether THIS data can answer THAT question, and flag "
            "anything about the data that would affect the answer. Cite exact columns, "
            "counts, and values (e.g. 'revenue is null in 4 of 36 rows: Feb-North, "
            "Mar-East...'). Do NOT give a generic data-health overview — focus on what "
            "matters for THIS question. 2-4 specific findings with real numbers."
        ),
    },
    "trends": {
        "label": "Trends & Patterns",
        "icon": "trending",
        "system": (
            "You are the Trends & Patterns specialist. Answer with SPECIFIC evidence "
            "relevant to the user's exact question. Cite real values, deltas, and names "
            "from the data — 'North grew from 120k to 149k (+24%) while South fell 8%', "
            "not 'there is growth'. Identify the specific segments, periods, or categories "
            "that matter for what they asked. Do NOT summarise everything — zero in on "
            "what answers the question. 2-4 findings, every one quantified."
        ),
    },
    "risk": {
        "label": "Risk & Anomaly",
        "icon": "alert",
        "system": (
            "You are the Risk & Anomaly specialist. Surface the SPECIFIC risks and "
            "anomalies relevant to the user's exact question, named precisely: which row, "
            "which category, which value, how far from normal. 'March-North spiked to "
            "310k, 3.2x the 96k monthly average' not 'there are some outliers'. Only flag "
            "what the data actually supports, and only what bears on THIS question. Rank "
            "by severity. 2-4 concrete findings."
        ),
    },
}

SYNTH_SYSTEM = (
    "You are the lead analyst writing ONE authoritative answer for a business user. "
    "You are given the user's EXACT question, a data summary, and specialist findings.\n\n"
    "ABSOLUTE RULES:\n"
    "1. Answer the SPECIFIC question asked — nothing more, nothing less. If they ask "
    "'which region is most at risk', name the region and say why; do NOT give a general "
    "overview of all regions. If they ask 'why did March drop', explain March "
    "specifically; do NOT summarise the whole trend.\n"
    "2. Be CONCRETE. Cite the actual numbers, names, categories, dates, and rows from "
    "the data — real values, not vague direction words. 'North fell 18% (from 240k to "
    "197k)' not 'some regions declined'. Every claim must point to specific evidence.\n"
    "3. NO generic filler. Ban phrases like 'the data shows interesting patterns', "
    "'overall performance is strong', 'there are several factors'. If a sentence would "
    "be true of almost any dataset, delete it.\n"
    "4. Lead with the DIRECT answer to their exact question in the first sentence. Then "
    "the specific evidence. Then only caveats that genuinely affect THIS answer.\n"
    "5. If the data cannot answer the question, say exactly what's missing — don't pad "
    "with unrelated observations.\n"
    "Do NOT mention 'agents', 'specialists', or the panel. Speak as one expert. "
    "Precise, specific, grounded in this dataset's actual values."
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
        """Compact but detailed summary — enough for the agents to answer specifically."""
        if not data:
            return "No data provided."
        cols = columns or (list(data[0].keys()) if isinstance(data[0], dict) else [])
        n = len(data)
        sample = data[:20]
        numeric_cols, cat_cols = {}, {}
        try:
            import statistics
            for c in cols:
                vals, missing = [], 0
                raw = []
                for row in data:
                    v = row.get(c) if isinstance(row, dict) else None
                    if v is None or v == "":
                        missing += 1
                    raw.append(v)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        vals.append(v)
                if len(vals) >= 3:
                    svals = sorted(vals)
                    numeric_cols[c] = {
                        "min": min(vals), "max": max(vals),
                        "mean": round(statistics.mean(vals), 2),
                        "median": round(statistics.median(vals), 2),
                        "std": round(statistics.pstdev(vals), 2) if len(vals) > 1 else 0,
                        "count": len(vals), "missing": missing,
                    }
                else:
                    # treat as categorical — top values
                    from collections import Counter
                    non_null = [str(v) for v in raw if v is not None and v != ""]
                    if non_null and len(set(non_null)) <= max(30, n // 2):
                        top = Counter(non_null).most_common(6)
                        cat_cols[c] = {"unique": len(set(non_null)), "missing": missing,
                                       "top": top}
        except Exception:
            pass
        lines = [f"Rows: {n}", f"Columns ({len(cols)}): {', '.join(map(str, cols))}"]
        if numeric_cols:
            lines.append("Numeric columns (min/median/mean/max/std, missing):")
            for c, s in list(numeric_cols.items())[:15]:
                lines.append(f"  {c}: min={s['min']} median={s['median']} mean={s['mean']} "
                             f"max={s['max']} std={s['std']} missing={s['missing']}")
        if cat_cols:
            lines.append("Categorical columns (top values by count):")
            for c, s in list(cat_cols.items())[:10]:
                tops = ", ".join(f"{v}={cnt}" for v, cnt in s["top"])
                lines.append(f"  {c}: {s['unique']} unique, missing={s['missing']} — top: {tops}")
        lines.append(f"Sample rows (first 20 of {n}):")
        lines.append(json.dumps(sample, default=str)[:3000])
        return "\n".join(lines)

    async def _run_agent(self, agent_key: str, question: str, data_summary: str, industry: str) -> dict:
        """Run a single specialist. Never raises — returns a status dict."""
        spec = AGENTS[agent_key]
        messages = [{
            "role": "user",
            "content": (
                f"THE QUESTION TO SERVE: \"{question}\"\n\n"
                f"Data summary:\n{data_summary}\n\n"
                f"Give findings from your specialty that specifically help answer that "
                f"exact question. Cite real values from the data above. Be specific, not general."
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
                f"════════════════════════════════════════\n"
                f"THE EXACT QUESTION YOU MUST ANSWER:\n\"{question}\"\n"
                f"════════════════════════════════════════\n\n"
                f"Data summary:\n{data_summary}\n\n"
                f"Specialist findings to draw from:\n{findings_block}{emphasis_note}\n\n"
                f"Now write your answer. First sentence must directly answer "
                f"\"{question}\" with a specific claim citing real values. Then the "
                f"specific supporting evidence. No generic overview."
            ),
        }]
        try:
            from app.services.elite_llm_service import elite_llm_service, LLMProvider
            answer, s_tokens, s_provider = await elite_llm_service.chat(
                messages=synth_msg, industry=industry, provider=LLMProvider.anthropic,
                max_tokens=1400, temperature=0.05)
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
