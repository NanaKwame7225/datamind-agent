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
        "provider": "gemini",
        "system": (
            "You are the Data Quality specialist. The user asked a SPECIFIC question — "
            "your job is to assess whether THIS data can answer THAT question, and flag "
            "anything about the data that would affect the answer. Cite exact columns, "
            "counts, and values (e.g. 'revenue is null in 4 of 36 rows: Feb-North, "
            "Mar-East...'). Do NOT give a generic data-health overview — focus on what "
            "matters for THIS question. 2-4 specific findings with real numbers.\n\n"
            "GROUNDING CHECK (do this FIRST, before anything else): compare the nouns and "
            "metrics in the question against the actual column names in the data summary. "
            "If the question asks about an entity or metric that is NOT a column (e.g. it "
            "asks about 'games' or 'profit' but the columns are value_a, value_b, period), "
            "your FIRST finding must state that plainly — name exactly which requested "
            "thing has no matching column. Do NOT assume value_a means 'profit', or that a "
            "period column means 'month', unless the data explicitly says so. A guessed "
            "mapping is a data-quality risk, not a fact — flag it as an assumption."
        ),
    },
    "trends": {
        "label": "Trends & Patterns",
        "icon": "trending",
        "provider": "openai",
        "system": (
            "You are the Trends & Patterns specialist. Answer with SPECIFIC evidence "
            "relevant to the user's exact question. Cite real values, deltas, and names "
            "from the data — 'North grew from 120k to 149k (+24%) while South fell 8%', "
            "not 'there is growth'. Identify the specific segments, periods, or categories "
            "that matter for what they asked. Do NOT summarise everything — zero in on "
            "what answers the question. 2-4 findings, every one quantified.\n\n"
            "Only reason about columns that actually exist in the data summary. If the "
            "question names something that is not a column, do not invent it or silently "
            "substitute a similarly-shaped column — refer to columns by their real names "
            "and note the mismatch."
        ),
    },
    "risk": {
        "label": "Risk & Anomaly",
        "icon": "alert",
        "provider": "groq",
        "system": (
            "You are the Risk & Anomaly specialist. Surface the SPECIFIC risks and "
            "anomalies relevant to the user's exact question, named precisely: which row, "
            "which category, which value, how far from normal. 'March-North spiked to "
            "310k, 3.2x the 96k monthly average' not 'there are some outliers'. Only flag "
            "what the data actually supports, and only what bears on THIS question. Rank "
            "by severity. 2-4 concrete findings."
        ),
    },
    "forecasting": {
        "label": "Forecasting",
        "icon": "trending",
        "provider": "openai",
        "system": (
            "You are the Forecasting specialist. Project where the key metrics are "
            "heading based on the trend in the data. Cite the actual trajectory — "
            "'revenue rose 120k\u2192149k\u2192171k over three periods, ~15%/period, implying "
            "~197k next period if the trend holds'. State your assumption (linear, "
            "seasonal, decelerating) and the main risk to the projection. Only forecast "
            "columns that exist and actually have a time/sequence order; if the data has "
            "no real time axis, say so instead of inventing a forecast. 2-3 findings."
        ),
    },
    "segmentation": {
        "label": "Segmentation",
        "icon": "grid",
        "provider": "gemini",
        "system": (
            "You are the Segmentation specialist. Break the data into the meaningful "
            "groups that answer the question, and quantify each. 'Three tiers by value: "
            "top 8 SKUs = 61% of value, next 40 = 30%, long tail 200+ = 9%'. Name the "
            "actual categories/clusters and their real shares. Use existing columns to "
            "group; do not invent segments the data can't support. 2-4 findings."
        ),
    },
    "benchmarking": {
        "label": "Benchmarking",
        "icon": "target",
        "provider": "openai",
        "system": (
            "You are the Benchmarking specialist. Compare the data's figures against "
            "sensible industry norms or internal baselines for this industry, and say "
            "whether each is above/below and by how much. 'Margin 17.5% vs ~25% industry "
            "norm \u2014 7.5pts under'. Be explicit that norms are approximate unless a "
            "benchmark is provided in the data. Flag the metrics most out of line. "
            "2-3 findings."
        ),
    },
    "recommendations": {
        "label": "Recommendations",
        "icon": "check",
        "provider": "anthropic",
        "system": (
            "You are the Recommendations specialist. Give concrete, prioritised NEXT "
            "ACTIONS grounded in the specific data \u2014 not generic advice. Each action "
            "must reference the real finding that motivates it: 'Liquidate the 209 SKUs "
            "unsold >30 days (GHS 124,605 tied up) \u2014 they are 0.8% of volume but block "
            "working capital'. Rank by impact. 2-4 actions, each tied to a number."
        ),
    },
}

SYNTH_SYSTEM = (
    "You are the lead analyst writing ONE authoritative answer for a business user. "
    "You are given the user's EXACT question, a data summary, and specialist findings.\n\n"
    "RULE 0 — GROUND THE QUESTION IN THE ACTUAL COLUMNS (do this before any other rule):\n"
    "The data summary lists the real column names. Check the question's key nouns and "
    "metrics against them. If the question asks about an entity or metric that does NOT "
    "exist as a column (e.g. asks about 'games' or 'profit' when the columns are only "
    "value_a, value_b, value_c, period), then:\n"
    "  (a) Your FIRST sentence must say so plainly — name exactly what is missing: "
    "'This dataset has no game or profit column.'\n"
    "  (b) Then ask the single clarifying question that would resolve it: 'Did you mean "
    "value_a — and is that revenue or profit?'\n"
    "  (c) Then, only if a reasonable interpretation exists, give a best-effort answer "
    "that EXPLICITLY labels the assumption: 'Treating value_a as an unlabelled metric, "
    "its highest value is...'. Never present a guessed column mapping as if it were a "
    "fact in the data, and never silently rename a column to match the question. A "
    "column called 'period' is not automatically 'month'; value_a is not automatically "
    "'profit'.\n"
    "If, and ONLY if, the question's nouns and metrics DO map cleanly onto real columns, "
    "ignore Rule 0 and answer with full confidence per the rules below.\n\n"
    "ABSOLUTE RULES (apply once the question is grounded):\n"
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
    "4. Lead with the DIRECT answer to their exact question in the first sentence "
    "(unless Rule 0 applies, in which case the missing-column statement comes first). "
    "Then the specific evidence. Then only caveats that genuinely affect THIS answer.\n"
    "5. If the data cannot answer the question, say exactly what's missing FIRST — "
    "don't pad with unrelated observations, and don't bury the gap under a confident-"
    "sounding answer.\n"
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
            lines = [f"Rows: {n}", f"Columns ({len(cols)}): {', '.join(map(str, cols))}",
                     "These are the ONLY columns in the data. If the question refers to "
                     "anything not in this list, that thing is not present — say so rather "
                     "than mapping it onto one of these columns."]
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
            # THE LEADERS — without these, "which is highest?" is unanswerable,
            # because the top row may sit anywhere in the file.
            try:
                skip=("rank","id","index","year","no","number","code")
                cands=[c for c in df.columns
                       if pd.api.types.is_numeric_dtype(df[c])
                       and not any(k in str(c).lower() for k in skip)]
                if cands:
                    whole=[c for c in cands if any(w in str(c).lower()
                           for w in ("global","total","overall","combined","gross","net"))]
                    m=(whole or [df[cands].sum().idxmax()])[0]
                    labels=[c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])][:2]
                    show=[c for c in dict.fromkeys([*labels,m]) if c in df.columns]
                    top=df.nlargest(12,m)[show]
                    lines.append(f"TOP 12 ROWS BY {str(m).upper()} (from the WHOLE dataset — "
                                 f"authoritative for 'which is highest/top/best', but ONLY "
                                 f"for the actual column {str(m)}, not for a differently-named "
                                 f"thing the user may have asked about):")
                    lines.append(top.to_string(index=False,max_colwidth=32)[:1200])
            except Exception:
                pass
            lines.append(f"Sample rows (first 20 of {n}, file order — NOT ranked):")
            lines.append(json.dumps(data[:20], default=str)[:2200])
            return "\n".join(lines)
        except Exception:
            return self._data_summary_fallback(data, columns)

    def _data_summary_fallback(self, data: list, columns: list = None) -> str:
        """Pure-Python summary if pandas isn't available."""
        cols = columns or (list(data[0].keys()) if isinstance(data[0], dict) else [])
        n = len(data)
        lines = [f"Rows: {n}", f"Columns ({len(cols)}): {', '.join(map(str, cols))}",
                 "These are the ONLY columns in the data. If the question refers to "
                 "anything not in this list, that thing is not present — say so rather "
                 "than mapping it onto one of these columns."]
        lines.append(f"Sample rows (first 20 of {n}):")
        lines.append(json.dumps(data[:20], default=str)[:3000])
        return "\n".join(lines)

    # Difficulty tier -> which provider to prefer. Each still fails over.
    TIER_PROVIDER = {"hard": "anthropic", "medium": "openai", "easy": "gemini"}

    async def _route(self, question: str, data_summary: str, industry: str) -> dict:
        """
        One fast, cheap LLM call that decides which specialists to run and at
        what difficulty tier. Returns {"agents": {key: tier, ...}}.
        Never raises: on any failure returns {} so analyze() uses the static panel.
        """
        roster = ", ".join(AGENTS.keys())
        router_prompt = (
            "You are a routing controller for a data-analysis panel. Given a user "
            "question and the dataset's columns, choose which specialist agents should "
            "run and how hard each one's job is.\n\n"
            f"AVAILABLE AGENTS: {roster}\n"
            "TIERS: hard (needs the strongest model), medium, easy.\n\n"
            f"QUESTION: \"{question}\"\n"
            f"DATA (columns + summary):\n{data_summary[:1200]}\n\n"
            "Rules: always include data_quality. Include the 2-4 agents most relevant "
            "to THIS question (e.g. forecasting only if there's a time/sequence axis; "
            "segmentation for grouping questions; benchmarking for comparison questions; "
            "recommendations if they ask what to do). Set tier by how much reasoning the "
            "question demands. Respond with ONLY a JSON object, no prose:\n"
            '{"agents": {"data_quality": "easy", "trends": "medium", ...}}'
        )
        try:
            from app.services.llm_service import llm_service
            from app.models.schemas import LLMProvider
            import json, re
            # Route on a fast/cheap model; fall over if it's down.
            provider = self._resolve_provider("gemini")
            text, _, _ = await llm_service.chat(
                messages=[{"role": "user", "content": router_prompt}],
                industry=industry, provider=provider,
                max_tokens=250, temperature=0.0)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return {}
            plan = json.loads(m.group(0))
            agents = plan.get("agents", {})
            # Keep only valid agent keys; drop anything hallucinated.
            agents = {k: v for k, v in agents.items() if k in AGENTS}
            if "data_quality" not in agents:
                agents["data_quality"] = "medium"
            return {"agents": agents} if agents else {}
        except Exception as e:
            logger.warning(f"Router failed, using static panel: {e}")
            return {}

    async def _run_agent(self, agent_key: str, question: str, data_summary: str, industry: str, provider_override: str = None) -> dict:
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
            preferred = provider_override or spec.get("provider")
            text, tokens, provider = await self._chat_with_system(spec["system"], messages, industry, preferred)
            return {"agent": agent_key, "label": spec["label"], "icon": spec["icon"],
                    "ok": True, "findings": text, "provider": provider, "tokens": tokens}
        except Exception as e:
            logger.warning(f"Agent {agent_key} failed: {e}")
            return {"agent": agent_key, "label": spec["label"], "icon": spec["icon"],
                    "ok": False, "findings": None, "error": str(e)[:200]}

    @staticmethod
    def _resolve_provider(name):
        """Map a provider name string to the LLMProvider enum, safely.
        Unknown names (e.g. 'groq', which may not be an enum member) fall back
        to openai as the requested start — chat() still fails over from there."""
        from app.models.schemas import LLMProvider
        try:
            return LLMProvider(name)
        except Exception:
            try:
                return LLMProvider.openai
            except Exception:
                return LLMProvider.anthropic

    async def _chat_with_system(self, agent_system: str, messages: list, industry: str,
                                preferred: str = None):
        """Call the LLM chat with the agent's persona prepended to the first message.
        `preferred` is the provider this specialist tries FIRST; if it fails, chat()
        falls back down the full provider chain automatically."""
        from app.services.llm_service import llm_service
        provider = self._resolve_provider(preferred or "anthropic")
        framed = [{"role": "user", "content": agent_system + "\n\n" + messages[0]["content"]}]
        text, tokens, used = await llm_service.chat(
            messages=framed, industry=industry, provider=provider,
            max_tokens=900, temperature=0.1)
        return text, tokens, used

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
            "surgical specificity.\n"
            "RULE 0 — GROUND FIRST: check the question's key nouns and metrics against the "
            "actual column names in the data summary. If the question asks about an entity "
            "or metric that is NOT a column (e.g. 'games' or 'profit' when the columns are "
            "value_a, value_b, period), then your FIRST sentence must say so plainly ('This "
            "dataset has no game or profit column'), THEN ask the one clarifying question "
            "that resolves it ('Did you mean value_a, and is it revenue or profit?'), THEN "
            "give a best-effort answer only if a reasonable reading exists, explicitly "
            "labelling the assumption ('treating value_a as an unlabelled metric...'). "
            "Never silently rename a column to fit the question; 'period' is not "
            "automatically 'month' and value_a is not automatically 'profit'. If the "
            "question DOES map cleanly onto real columns, skip Rule 0 and answer with full "
            "confidence.\n"
            "OTHER RULES: (1) Answer only what's asked — name the "
            "specific region/month/category/row, don't give a general overview. "
            "(2) Cite real values from the data ('North fell 240k→197k, -18%'), never "
            "vague direction words. (3) No generic filler — if a sentence would be true "
            "of any dataset, cut it. (4) First sentence directly answers the question "
            "(or, under Rule 0, states what's missing). "
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
                f"Answer now. First, silently check the question's nouns against the real "
                f"columns above. If they match, answer \"{question}\" directly with a "
                f"specific claim citing real values, then the evidence. If something asked "
                f"about is not a column, say what's missing first, ask the clarifying "
                f"question, then give a clearly-labelled best-effort reading."
            ),
        }]
        try:
            from app.services.llm_service import llm_service
            from app.models.schemas import LLMProvider
            answer, tokens, provider = await llm_service.chat(
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

        # 1. Route: decide which specialists run and at what tier (cheap call).
        #    On any router failure we fall back to the static core panel.
        plan = await self._route(question, data_summary, industry)
        if plan.get("agents"):
            routed = plan["agents"]  # {agent_key: tier}
        else:
            routed = {"data_quality": "medium", "trends": "medium", "risk": "medium"}
        specialist_keys = list(routed.keys())

        # Map each agent's tier to a preferred provider (still fails over).
        results = await asyncio.gather(*[
            self._run_agent(k, question, data_summary, industry,
                            self.TIER_PROVIDER.get(routed.get(k, "medium"), "openai"))
            for k in specialist_keys
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
                f"Now write your answer. First apply Rule 0: if the question names "
                f"something that is not a real column, lead with what's missing, ask the "
                f"clarifying question, then give a labelled best-effort reading. Otherwise, "
                f"the first sentence must directly answer \"{question}\" with a specific "
                f"claim citing real values, then the specific supporting evidence. No "
                f"generic overview."
            ),
        }]
        try:
            from app.services.llm_service import llm_service
            from app.models.schemas import LLMProvider
            answer, s_tokens, s_provider = await llm_service.chat(
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
