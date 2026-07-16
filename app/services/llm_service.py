"""
DataMind Agent — Elite LLM Service
5-provider failover: Claude Sonnet 4 → GPT-4o → Gemini 2.0 Flash → Command R+ → Mistral Large
Injects pre-computed statistical evidence into every prompt.
"""
from __future__ import annotations
import logging
from typing import Optional
from config.settings import settings
from app.models.schemas import LLMProvider

logger = logging.getLogger(__name__)

INDUSTRY_CONTEXTS = {
    "finance": """You are a CFO-level financial data analyst with 20 years experience.
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
        provider: LLMProvider = LLMProvider.anthropic,
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

        # Distributions
        if ctx.get("distributions"):
            lines.append("\nDISTRIBUTIONS:")
            for col, d in list(ctx["distributions"].items())[:6]:
                lines.append(
                    f"  {col}: n={d['count']}, mean={d['mean']}, median={d['median']}, "
                    f"std={d['std']}, min={d['min']}, max={d['max']}, "
                    f"missing={d['missing_pct']}%, shape={d['distribution_shape']}, cv={d.get('cv_pct','?')}%"
                )

        # Anomalies
        anomalies = [f for f in ctx.get("findings", []) if f["type"] == "anomaly"]
        if anomalies:
            lines.append("\nANOMALIES DETECTED (Z-score > 3σ):")
            for f in anomalies[:5]:
                ev = f["evidence"]
                lines.append(
                    f"  {f['column']}: {ev['anomaly_count']} anomalies ({ev['anomaly_pct']}% of records), "
                    f"normal range [{ev['normal_range'][0]}, {ev['normal_range'][1]}], "
                    f"anomalous values: {ev['anomaly_values'][:3]}, "
                    f"impact on mean: +{ev['impact_on_mean_pct']}%, "
                    f"max Z-score: {ev['z_score_max']}σ, "
                    f"confidence: {round(f['confidence']*100)}%"
                )

        # Trends
        trends = [f for f in ctx.get("findings", []) if f["type"] == "trend"]
        if trends:
            lines.append("\nTRENDS (OLS linear regression):")
            for f in trends[:4]:
                ev = f["evidence"]
                sig = f"p={ev.get('p_value','?')} ({'significant' if ev.get('statistically_significant') else 'not significant'})"
                lines.append(
                    f"  {f['column']}: {ev['total_change_pct']}% change over {ev['period_count']} periods, "
                    f"R²={ev['r_squared']}, {sig}, "
                    f"first={ev['first_value']}, last={ev['last_value']}, "
                    f"slope={ev['slope_per_period']} per period"
                )

        # Correlations
        if ctx.get("correlations"):
            lines.append("\nCORRELATIONS (Pearson):")
            for c in ctx["correlations"][:5]:
                lines.append(
                    f"  {c['col1']} vs {c['col2']}: r={c['correlation']} "
                    f"({c['strength']} {c['direction']}), "
                    f"p={c['p_value']} ({'significant' if c['significant'] else 'not significant'}), "
                    f"n={c['n_observations']}. {c['interpretation']}"
                )

        # Segmentation
        if ctx.get("segmentation"):
            lines.append("\nSEGMENTATION ANALYSIS:")
            for seg_col, seg_data in list(ctx["segmentation"].items())[:3]:
                lines.append(f"  By {seg_col} ({seg_data['unique_segments']} segments):")
                for metric, segments in list(seg_data["metrics"].items())[:2]:
                    lines.append(f"    {metric}:")
                    for s in segments[:5]:
                        diff = s["deviation_from_overall_pct"]
                        sig = "(statistically different p<0.05)" if s["statistically_different"] else ""
                        lines.append(
                            f"      #{s['rank']} {s['segment']}: mean={s['mean']}, "
                            f"n={s['count']} ({s['pct_of_total']}% of data), "
                            f"deviation from avg: {diff:+.1f}% {sig}"
                        )

        # Impact ranking
        if ctx.get("impact_ranking"):
            lines.append("\nIMPACT RANKING (pre-computed, use this order):")
            for i, imp in enumerate(ctx["impact_ranking"][:5], 1):
                lines.append(
                    f"  #{i} {imp['column']}: impact_score={imp['score']}/10, "
                    f"confidence={round(imp['confidence']*100)}%, "
                    f"issue={imp['primary_issue']}, reason={imp['reason']}"
                )

        # Uncertainty flags
        if ctx.get("uncertainty"):
            lines.append("\nUNCERTAINTY FLAGS (include in self-audit):")
            for u in ctx["uncertainty"][:3]:
                lines.append(
                    f"  - {u['issue']}: {u['detail'][:150]} "
                    f"(confidence adjustment: {u['confidence_adjustment']})"
                )

        # Self-audit
        if ctx.get("self_audit"):
            for audit in ctx["self_audit"][:1]:
                lines.append(f"\nPRE-COMPUTED ASSUMPTIONS:")
                for a in audit.get("assumptions", [])[:4]:
                    lines.append(f"  - {a}")

        # Data grounding summary
        dg = ctx.get("data_grounding", {})
        lines.append(f"\nDATA GROUNDING SUMMARY:")
        lines.append(
            f"  Total data points analysed: {dg.get('total_data_points', '?')} | "
            f"Anomalies found: {dg.get('total_anomalies_found', 0)} | "
            f"Correlations: {dg.get('total_correlations_found', 0)} | "
            f"Segments: {dg.get('segments_analysed', 0)} | "
            f"Confidence range: {round(dg.get('lowest_confidence_finding',0)*100)}% – "
            f"{round(dg.get('highest_confidence_finding',0)*100)}%"
        )

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
        all_providers = [
            (LLMProvider.anthropic, self._anthropic, "Claude Sonnet 4",  "claude-sonnet-4-20250514", settings.ANTHROPIC_API_KEY),
            (LLMProvider.openai,    self._openai,    "GPT-4o",           "gpt-4o",                   settings.OPENAI_API_KEY),
            (LLMProvider.gemini,    self._gemini,    "Gemini 2.0 Flash", "gemini-2.0-flash",         settings.GOOGLE_API_KEY),
            (LLMProvider.cohere,    self._cohere,    "Command R+",       "command-r-plus",            settings.COHERE_API_KEY),
            (LLMProvider.mistral,   self._mistral,   "Mistral Large",    "mistral-large-latest",      settings.MISTRAL_API_KEY),
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
        genai.configure(api_key=settings.GOOGLE_API_KEY)
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
