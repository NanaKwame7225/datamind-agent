"""
DataMind Agent — Elite LLM Service
5-provider failover: Claude Sonnet 4 → GPT-4o → Gemini 2.0 Flash → Command R+ → Mistral Large
Injects pre-computed statistical evidence into every prompt.
"""
from __future__ import annotations
import logging, os
from typing import Optional
from config.settings import settings
from app.models.schemas import LLMProvider

logger = logging.getLogger(__name__)

INDUSTRY_CONTEXTS = {
    "finance": """You are a CFO-level financial data analyst with 30 years experience.
Your analysis must reference: ROI, EBITDA, NPV, cash conversion cycle, working capital ratios, liquidity, leverage.
Always ground claims in specific numbers from the data. Flag FX exposure, fraud signals, covenant risks.""",
    "education": """You are a senior education data analyst and institutional researcher.
Reference: pass rates, dropout predictors, cohort retention, fee collection efficiency, teacher-student ratios.
Always ground claims in specific numbers. Identify at-risk student segments with evidence.""",
    "supply_chain": """You are a supply chain director with operations research expertise.
Reference: OTIF, DSI, reorder points, EOQ, bullwhip effect, safety stock, supplier reliability scores.
Always ground claims in specific numbers. Quantify stockout risks and carrying costs.""",
    "procurement": """You are a Chief Procurement Officer with category management expertise.
Reference: spend concentration, maverick spend %, savings rate, PO cycle time, vendor performance scores.
Always ground claims in specific numbers. Rank vendors by risk and value.""",
    "healthcare": """You are a healthcare analytics director with clinical operations expertise.
Reference: LOS, readmission %, ALOS, bed utilisation, case mix index, cost per DRG.
Always ground claims in specific numbers. Prioritise patient safety signals.""",
    "mining": """You are a mining engineering analyst with operations expertise.
Reference: head grade, recovery rate, strip ratio, TRIFR, OEE, tonnes per man-shift.
Always ground claims in specific numbers. Quantify cost per recovered unit of ore.""",
    "petroleum": """You are a petroleum engineering analyst with reservoir expertise.
Reference: BOE/day, GOR, water cut, decline curve, lifting cost, RRR, facility uptime.
Always ground claims in specific numbers. Flag production decline signals.""",
    "marketing": """You are a CMO-level marketing analytics director.
Reference: CAC, LTV, LTV:CAC ratio, ROAS, MER, CTR, CPC, CPM, conversion rate, attribution,
funnel drop-off, channel mix, campaign incrementality, brand vs performance split, cohort retention.
Always ground claims in specific numbers. Attribute performance to specific channels and campaigns,
name the winners and losers, and quantify wasted spend. Distinguish correlation from incrementality.""",
    "sales": """You are a VP of Sales with revenue operations expertise.
Reference: pipeline coverage, win rate, average deal size, sales cycle length, quota attainment,
stage conversion, lead-to-close, churn, net revenue retention, rep productivity, forecast accuracy.
Always ground claims in specific numbers. Name the specific reps, regions, segments, or stages
driving or dragging performance, and quantify the gap to target.""",
    "retail": """You are a retail analytics director with commercial expertise.
Reference: GMV, basket size, conversion rate, inventory turnover, shrinkage %, churn rate, LTV.
Always ground claims in specific numbers. Segment by customer cohort and product category.""",
    "agriculture": """You are an agricultural economist and precision farming analyst.
Reference: yield per hectare, input-output ratio, price volatility, growing degree days.
Always ground claims in specific numbers. Model profit sensitivity to weather and price.""",
    "manufacturing": """You are a manufacturing excellence director with lean/six sigma expertise.
Reference: OEE, PPM defects, MTTR, MTBF, cycle time, first pass yield, capacity utilisation.
Always ground claims in specific numbers. Rank defect types by frequency and cost.""",
    "ngo": """You are a MEAL director with UN/INGO expertise.
Reference: cost per beneficiary, outcome indicators, attribution gap, donor LTV, budget variance.
Always ground claims in specific numbers. Assess programme effectiveness with evidence.""",
    "general": """You are a senior data scientist and business intelligence director.
Adapt metrics and terminology to the specific industry and dataset.
Always ground claims in specific numbers from the data provided.""",
}

ELITE_SYSTEM = """You are DataMind Agent — an elite AI business analyst operating at the level of a seasoned McKinsey partner combined with a PhD-level data scientist.

CORE RULES — YOU MUST FOLLOW EVERY ONE:

1. EVERY CLAIM MUST BE GROUNDED IN DATA
   - Never say "likely", "probably", "may" without citing the specific evidence
   - Always quote exact numbers: "12.4% of values", "3 out of 47 records", "deviation of 2.8σ"
   - Cite the statistical method used: "Z-score analysis", "OLS regression (p=0.003)", "Pearson r=0.87"

2. ALWAYS INCLUDE CONFIDENCE SCORES
   - State your confidence for each major finding: High (>80%), Medium (60-80%), Low (<60%)
   - Explain what drives the confidence level
   - Flag where small sample size reduces reliability

3. IMPACT RANKING IS MANDATORY
   - Rank every finding by business impact (1 = highest impact)
   - Explain WHY each issue matters to KPIs specifically
   - Quantify the impact where possible: "This anomaly inflates the mean by 18%"

4. SEGMENT EVERY FINDING
   - Break down findings by available categories (region, department, product, etc.)
   - Identify which segment is driving the issue
   - Compare best vs worst performing segments with exact numbers

5. UNCERTAINTY AND SELF-AUDIT
   - State explicitly what assumptions you are making
   - Identify what you could be wrong about
   - Flag what additional data would improve the analysis

RESPONSE STRUCTURE — USE EXACTLY THIS FORMAT:

## Executive Summary
One paragraph. The single most important thing the decision-maker needs to know. One actionable sentence at the end.

## Key Findings (ranked by impact)

### Finding 1 — [Title] | Impact: HIGH | Confidence: [X]%
**Evidence:** [Exact numbers, sample rows, distributions]
**Business Impact:** [Specific KPI effect, quantified where possible]
**Affected Segments:** [Which categories/groups are most affected]
**Recommended Action:** [Specific, measurable action]

### Finding 2 — [Title] | Impact: MEDIUM | Confidence: [X]%
[Same structure]

## What the Data Shows
A structured narrative explaining the overall patterns with specific numbers throughout.

## Segment Analysis
Break down the top metric by every available categorical variable. Quote exact numbers for each segment.

## Correlations & Relationships
Which metrics move together? What is the r-value? Is it statistically significant?

## Recommendations (numbered, prioritised)
1. [Most impactful] — [Why] — [How to measure success]
2. [Second most impactful] — [Why] — [How to measure success]

## Uncertainty & Caveats
- What assumptions were made
- What this analysis cannot tell you
- What additional data would improve confidence

## Self-Audit
- What I could be wrong about
- Alternative explanations for the patterns found
- Limitations of the methods used"""


class EliteLLMService:
    """
    Elite LLM service with automatic failover across 5 providers.
    Injects pre-computed statistical evidence into every prompt.
    Priority: Claude Sonnet 4 → GPT-4o → Gemini 2.0 Flash → Command R+ → Mistral Large
    """

    async def chat(
        self,
        messages: list[dict],
        industry: str = "general",
        provider: LLMProvider = LLMProvider.openai,
        model: Optional[str] = None,
        max_tokens: int = 2500,
        temperature: float = 0.1,
        elite_context: Optional[dict] = None,
    ) -> tuple[str, int, str]:
        """
        Returns (response_text, tokens_used, provider_used).
        elite_context: pre-computed analysis dict injected into the prompt.
        """
        system = ELITE_SYSTEM + "\n\n" + INDUSTRY_CONTEXTS.get(industry, INDUSTRY_CONTEXTS["general"])

        # Inject elite pre-computed statistical evidence
        if elite_context:
            messages = self._inject_elite_context(messages, elite_context)

        chain = self._build_chain(provider, model)
        last_error = None
        for fn, name, mdl in chain:
            try:
                logger.info(f"Trying {name}")
                text, tokens = await fn(messages, system, mdl, max_tokens, temperature)
                logger.info(f"{name} succeeded — {tokens} tokens")
                return text, tokens, name
            except Exception as e:
                last_error = e
                logger.warning(f"{name} failed: {e}")
                continue
        raise Exception(f"All providers failed. Last error: {last_error}")

    def _inject_elite_context(self, messages: list[dict], ctx: dict) -> list[dict]:
        """Prepend pre-computed statistical evidence to the user message."""
        lines = [
            f"DATASET: {ctx.get('row_count', '?')} rows × {ctx.get('col_count', '?')} columns",
            f"COLUMNS: {', '.join(ctx.get('columns', [])[:12])}",
            "",
            "PRE-COMPUTED STATISTICAL EVIDENCE — USE THESE EXACT NUMBERS IN YOUR RESPONSE:",
        ]

        def g(d, *keys, default="?"):
            """Safe nested get: g(d,'a','b') == d['a']['b'] or default."""
            cur = d
            for k in keys:
                if isinstance(cur, dict) and k in cur and cur[k] is not None:
                    cur = cur[k]
                elif isinstance(cur, (list, tuple)) and isinstance(k, int) and -len(cur) <= k < len(cur):
                    cur = cur[k]
                else:
                    return default
            return cur

        # Distributions
        try:
            if ctx.get("distributions"):
                lines.append("\nDISTRIBUTIONS:")
                for col, d in list(ctx["distributions"].items())[:6]:
                    lines.append(
                        f"  {col}: n={g(d,'count')}, mean={g(d,'mean')}, median={g(d,'median')}, "
                        f"std={g(d,'std')}, min={g(d,'min')}, max={g(d,'max')}, "
                        f"missing={g(d,'missing_pct')}%, shape={g(d,'distribution_shape')}, cv={g(d,'cv_pct')}%"
                    )
        except Exception as e:
            logger.debug(f"distributions block skipped: {e}")

        # Anomalies
        try:
            anomalies = [f for f in ctx.get("findings", []) if f.get("type") == "anomaly"]
            if anomalies:
                lines.append("\nANOMALIES DETECTED (Z-score > 3\u03c3):")
                for f in anomalies[:5]:
                    ev = f.get("evidence", {}) or {}
                    nr = ev.get("normal_range", ["?", "?"])
                    lo, hi = (nr + ["?", "?"])[:2] if isinstance(nr, list) else ("?", "?")
                    conf = f.get("confidence")
                    conf_s = f"{round(conf*100)}%" if isinstance(conf, (int, float)) else "?"
                    lines.append(
                        f"  {f.get('column','?')}: {g(ev,'anomaly_count')} anomalies ({g(ev,'anomaly_pct')}% of records), "
                        f"normal range [{lo}, {hi}], "
                        f"anomalous values: {(ev.get('anomaly_values') or [])[:3]}, "
                        f"impact on mean: +{g(ev,'impact_on_mean_pct')}%, "
                        f"max Z-score: {g(ev,'z_score_max')}\u03c3, "
                        f"confidence: {conf_s}"
                    )
        except Exception as e:
            logger.debug(f"anomalies block skipped: {e}")

        # Trends
        try:
            trends = [f for f in ctx.get("findings", []) if f.get("type") == "trend"]
            if trends:
                lines.append("\nTRENDS (OLS linear regression):")
                for f in trends[:4]:
                    ev = f.get("evidence", {}) or {}
                    sig = f"p={ev.get('p_value','?')} ({'significant' if ev.get('statistically_significant') else 'not significant'})"
                    lines.append(
                        f"  {f.get('column','?')}: {g(ev,'total_change_pct')}% change over {g(ev,'period_count')} periods, "
                        f"R\u00b2={g(ev,'r_squared')}, {sig}, "
                        f"first={g(ev,'first_value')}, last={g(ev,'last_value')}, "
                        f"slope={g(ev,'slope_per_period')} per period"
                    )
        except Exception as e:
            logger.debug(f"trends block skipped: {e}")

        # Correlations (key names vary across analysis versions — read defensively)
        try:
            if ctx.get("correlations"):
                lines.append("\nCORRELATIONS (Pearson):")
                for c in ctx["correlations"][:5]:
                    pval = c.get("p_value",
                                 c.get("p_value_bonferroni_corrected",
                                       c.get("p_value_raw", "?")))
                    is_sig = c.get("significant",
                                   c.get("significant_after_correction", False))
                    lines.append(
                        f"  {c.get('col1','?')} vs {c.get('col2','?')}: r={c.get('correlation','?')} "
                        f"({c.get('strength','?')} {c.get('direction','?')}), "
                        f"p={pval} ({'significant' if is_sig else 'not significant'}), "
                        f"n={c.get('n_observations', c.get('n','?'))}. {c.get('interpretation','')}"
                    )
        except Exception as e:
            logger.debug(f"correlations block skipped: {e}")

        # Segmentation
        try:
            if ctx.get("segmentation"):
                lines.append("\nSEGMENTATION ANALYSIS:")
                for seg_col, seg_data in list(ctx["segmentation"].items())[:3]:
                    if not isinstance(seg_data, dict):
                        continue
                    lines.append(f"  By {seg_col} ({seg_data.get('unique_segments','?')} segments):")
                    for metric, segments in list((seg_data.get("metrics") or {}).items())[:2]:
                        lines.append(f"    {metric}:")
                        for sgm in (segments or [])[:5]:
                            diff = sgm.get("deviation_from_overall_pct")
                            diff_s = f"{diff:+.1f}%" if isinstance(diff, (int, float)) else "?"
                            sig = "(statistically different p<0.05)" if sgm.get("statistically_different") else ""
                            lines.append(
                                f"      #{sgm.get('rank','?')} {sgm.get('segment','?')}: mean={sgm.get('mean','?')}, "
                                f"n={sgm.get('count','?')} ({sgm.get('pct_of_total','?')}% of data), "
                                f"deviation from avg: {diff_s} {sig}"
                            )
        except Exception as e:
            logger.debug(f"segmentation block skipped: {e}")

        # Impact ranking
        try:
            if ctx.get("impact_ranking"):
                lines.append("\nIMPACT RANKING (pre-computed, use this order):")
                for i, imp in enumerate(ctx["impact_ranking"][:5], 1):
                    conf = imp.get("confidence")
                    conf_s = f"{round(conf*100)}%" if isinstance(conf, (int, float)) else "?"
                    lines.append(
                        f"  #{i} {imp.get('column','?')}: impact_score={imp.get('score','?')}/10, "
                        f"confidence={conf_s}, "
                        f"issue={imp.get('primary_issue','?')}, reason={imp.get('reason','?')}"
                    )
        except Exception as e:
            logger.debug(f"impact ranking block skipped: {e}")

        # Uncertainty flags
        try:
            if ctx.get("uncertainty"):
                lines.append("\nUNCERTAINTY FLAGS (include in self-audit):")
                for u in ctx["uncertainty"][:3]:
                    detail = str(u.get("detail", ""))[:150]
                    lines.append(
                        f"  - {u.get('issue','?')}: {detail} "
                        f"(confidence adjustment: {u.get('confidence_adjustment','?')})"
                    )
        except Exception as e:
            logger.debug(f"uncertainty block skipped: {e}")

        # Self-audit
        try:
            if ctx.get("self_audit"):
                for audit in ctx["self_audit"][:1]:
                    lines.append(f"\nPRE-COMPUTED ASSUMPTIONS:")
                    for a in (audit.get("assumptions") or [])[:4]:
                        lines.append(f"  - {a}")
        except Exception as e:
            logger.debug(f"self-audit block skipped: {e}")

        # Data grounding summary
        try:
            dg = ctx.get("data_grounding", {}) or {}
            lc = dg.get("lowest_confidence_finding", 0) or 0
            hc = dg.get("highest_confidence_finding", 0) or 0
            lines.append(f"\nDATA GROUNDING SUMMARY:")
            lines.append(
                f"  Total data points analysed: {dg.get('total_data_points', '?')} | "
                f"Anomalies found: {dg.get('total_anomalies_found', 0)} | "
                f"Correlations: {dg.get('total_correlations_found', 0)} | "
                f"Segments: {dg.get('segments_analysed', 0)} | "
                f"Confidence range: {round(lc*100)}% \u2013 {round(hc*100)}%"
            )
        except Exception as e:
            logger.debug(f"data grounding block skipped: {e}")

        evidence_block = "\n".join(lines)

        # Prepend evidence to the last user message
        augmented = []
        for i, msg in enumerate(messages):
            if msg["role"] == "user" and i == len(messages) - 1:
                augmented.append({
                    "role": "user",
                    "content": evidence_block + "\n\n" + msg["content"],
                })
            else:
                augmented.append(msg)
        return augmented

    def _build_chain(self, preferred_provider, preferred_model):
        # Keys are read from the environment with sensible aliases, so a var
        # named GEMINI_API_KEY works as well as GOOGLE_API_KEY. This avoids a
        # provider being silently dropped due to an env-var name mismatch.
        def _key(*names):
            for n in names:
                v = os.getenv(n)
                if v:
                    return v
            return None

        gemini_model = os.getenv("GEMINI_MODEL") or "gemini-2.0-flash"
        # Order: strongest AVAILABLE first. Gemini leads (fast, strong, key
        # known good), then OpenAI, then Groq and Mistral as fast backstops.
        # Anthropic/Cohere stay defined so they're used automatically if a key
        # is ever added.
        # Order: strongest AVAILABLE first. Claude leads (best analytical
        # quality, key present), then OpenAI, Gemini, and Groq/Mistral as fast
        # backstops. A provider with no key is filtered out below, so a missing
        # key costs no time.
        all_providers = [
            (LLMProvider.anthropic, self._anthropic, "Claude Sonnet 4",  "claude-sonnet-4-20250514", _key("ANTHROPIC_API_KEY")),
            (LLMProvider.openai,    self._openai,    "GPT-4o",           "gpt-4o",                   _key("OPENAI_API_KEY")),
            (LLMProvider.gemini,    self._gemini,    "Gemini 2.0 Flash", gemini_model,               _key("GOOGLE_API_KEY", "GEMINI_API_KEY")),
            ("groq",                self._groq,      "Llama 3.3 70B",    "llama-3.3-70b-versatile",  _key("GROQ_API_KEY")),
            (LLMProvider.mistral,   self._mistral,   "Mistral Large",    "mistral-large-latest",      _key("MISTRAL_API_KEY")),
            (LLMProvider.cohere,    self._cohere,    "Command R+",       "command-r-plus",            _key("COHERE_API_KEY")),
        ]
        preferred = [
            (fn, name, preferred_model or mdl)
            for p, fn, name, mdl, key in all_providers
            if p == preferred_provider and key
        ]
        fallbacks = [
            (fn, name, mdl)
            for p, fn, name, mdl, key in all_providers
            if p != preferred_provider and key
        ]
        chain = preferred + fallbacks
        if not chain:
            return [(self._no_keys_error, "no-provider", "none")]
        return chain

    async def _no_keys_error(self, *args, **kwargs):
        raise Exception(
            "No API keys configured. Add ANTHROPIC_API_KEY or OPENAI_API_KEY in Railway Variables."
        )

    async def _anthropic(self, messages, system, model, max_tokens, temperature):
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        r = await client.messages.create(
            model=model, max_tokens=max_tokens,
            temperature=temperature, system=system, messages=messages,
        )
        return r.content[0].text, r.usage.input_tokens + r.usage.output_tokens

    async def _openai(self, messages, system, model, max_tokens, temperature):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        return r.choices[0].message.content, r.usage.total_tokens

    async def _gemini(self, messages, system, model, max_tokens, temperature):
        import google.generativeai as genai
        genai.configure(api_key=(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")))
        m = genai.GenerativeModel(
            model_name=model, system_instruction=system,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        msgs = [
            {"role": "user" if x["role"] == "user" else "model", "parts": [x["content"]]}
            for x in messages
        ]
        chat = m.start_chat(history=msgs[:-1])
        r = await chat.send_message_async(msgs[-1]["parts"][0])
        tokens = r.usage_metadata.total_token_count if hasattr(r, "usage_metadata") else 0
        return r.text, tokens

    async def _cohere(self, messages, system, model, max_tokens, temperature):
        import cohere
        client = cohere.AsyncClientV2(api_key=settings.COHERE_API_KEY)
        r = await client.chat(
            model=model,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        return r.message.content[0].text, r.usage.tokens.input_tokens + r.usage.tokens.output_tokens

    async def _groq(self, messages, system, model, max_tokens, temperature):
        # Groq exposes an OpenAI-compatible endpoint.
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("GROQ_API_KEY"),
                             base_url="https://api.groq.com/openai/v1")
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        return r.choices[0].message.content, (r.usage.total_tokens if r.usage else 0)

    async def _mistral(self, messages, system, model, max_tokens, temperature):
        from mistralai import Mistral
        client = Mistral(api_key=settings.MISTRAL_API_KEY)
        r = await client.chat.complete_async(
            model=model,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        return r.choices[0].message.content, r.usage.total_tokens


llm_service = EliteLLMService()