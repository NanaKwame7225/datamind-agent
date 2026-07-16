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


def _r(v):
    """Round a numeric value for display, handling NaN/None."""
    try:
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return round(float(v), 2)
    except Exception:
        return v


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
        """Compact but detailed summary — enough for specific answers, fast on big data."""
        if not data:
            return "No data provided."
        n = len(data)
        # Use pandas for fast vectorised stats on large data (falls back to pure Python).
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            cols = columns or list(df.columns)
            lines = [f"Rows: {n}", f"Columns ({len(cols)}): {', '.join(map(str, cols))}"]
            num_lines, cat_lines = [], []
            for c in cols:
                if c not in df.columns:
                    continue
                s = df[c]
                missing = int(s.isna().sum())
                numeric = pd.to_numeric(s, errors="coerce")
                if numeric.notna().sum() >= 3 and numeric.notna().sum() >= 0.5 * s.notna().sum():
                    num_lines.append(
                        f"  {c}: min={_r(numeric.min())} median={_r(numeric.median())} "
                        f"mean={_r(numeric.mean())} max={_r(numeric.max())} "
                        f"std={_r(numeric.std())} missing={missing}")
                else:
                    nun = int(s.nunique(dropna=True))
                    if 0 < nun <= max(40, n // 2):
                        vc = s.value_counts().head(6)
                        tops = ", ".join(f"{k}={int(v)}" for k, v in vc.items())
                        cat_lines.append(f"  {c}: {nun} unique, missing={missing} — top: {tops}")
            if num_lines:
                lines.append("Numeric columns (min/median/mean/max/std, missing):")
                lines.extend(num_lines[:15])
            if cat_lines:
                lines.append("Categorical columns (top values by count):")
                lines.extend(cat_lines[:10])
            lines.append(f"Sample rows (first 20 of {n}):")
            lines.append(json.dumps(data[:20], default=str)[:3000])
            return "\n".join(lines)
        except Exception:
            return self._data_summary_fallback(data, columns)

    def _data_summary_fallback(self, data: list, columns: list = None) -> str:
        """Pure-Python summary if pandas isn't available."""
        cols = columns or (list(data[0].keys()) if isinstance(data[0], dict) else [])
        n = len(data)
        lines = [f"Rows: {n}", f"Columns ({len(cols)}): {', '.join(map(str, cols))}"]
        lines.append(f"Sample rows (first 20 of {n}):")
        lines.append(json.dumps(data[:20], default=str)[:3000])
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

    async def analyze_fast(self, question: str, data: list, columns: list = None,
                           industry: str = "general", data_summary: str = None) -> dict:
        """
        Fast mode: ONE well-prompted call that does the work of the panel — quality
        checks, trends, and risks — folded into a single specific answer. ~1/4 the
        time of the full panel. Used as the default; the panel is opt-in for depth.
        """
        if not question or not (question or "").strip():
            return {"success": False, "error": "Ask a question for this cell."}
        if not data and not data_summary:
            return {"success": False, "error": "This cell has no data to analyse."}
        if not data_summary:
            data_summary = self._data_summary(data, columns)
        emphasis = classify_question(question)

        system = (
            "You are an elite data analyst. Answer the user's EXACT question with "
            "surgical specificity. RULES: (1) Answer only what's asked — name the "
            "specific region/month/category/row, don't give a general overview. "
            "(2) Cite real values from the data ('North fell 240k→197k, -18%'), never "
            "vague direction words. (3) No generic filler — if a sentence would be true "
            "of any dataset, cut it. (4) First sentence directly answers the question. "
            "(5) Briefly note data-quality caveats or risks ONLY if they affect this "
            "answer. Be precise, specific, grounded in the actual numbers below."
        )
        msg = [{
            "role": "user",
            "content": (
                f"{system}\n\n"
                f"════════════════════════════════════════\n"
                f"THE EXACT QUESTION:\n\"{question}\"\n"
                f"════════════════════════════════════════\n\n"
                f"Data summary:\n{data_summary}\n\n"
                f"Answer now — first sentence directly answers \"{question}\" with a "
                f"specific claim citing real values. Then the specific evidence."
            ),
        }]
        try:
            from app.services.elite_llm_service import elite_llm_service, LLMProvider
            answer, tokens, provider = await elite_llm_service.chat(
                messages=msg, industry=industry, provider=LLMProvider.anthropic,
                max_tokens=1200, temperature=0.05)
            return {"success": True, "answer": answer, "emphasis": emphasis,
                    "agents": [], "provider": provider, "tokens": tokens,
                    "mode": "fast", "agents_ok": 1, "agents_total": 1}
        except Exception as e:
            logger.error(f"Fast analysis failed: {e}", exc_info=True)
            return {"success": False,
                    "error": f"Analysis failed — {str(e)[:220]}",
                    "agents": []}

    async def analyze(self, question: str, data: list, columns: list = None,
                      industry: str = "general", data_summary: str = None) -> dict:
        """
        Run the full panel: specialists in parallel, then synthesis.
        Returns the elite answer plus each agent's view and metadata.
        If data_summary is provided (precomputed at notebook creation), it's reused
        instead of recomputing stats over thousands of rows on every cell.
        """
        if not question or not (question or "").strip():
            return {"success": False, "error": "Ask a question for this cell."}
        if not data and not data_summary:
            return {"success": False, "error": "This cell has no data to analyse."}

        if not data_summary:
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
            # Surface the REAL reason — a generic "busy" message hides config
            # problems (missing keys, bad model names) that look identical.
            errs = [f"{r['label']}: {r.get('error','?')}" for r in results]
            logger.error("All agents failed: %s", " | ".join(errs))
            return {"success": False,
                    "error": "Analysis failed — " + (results[0].get("error") or "no provider responded")[:220],
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
