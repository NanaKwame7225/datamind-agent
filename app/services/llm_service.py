"""
DataMind Agent — Elite LLM Service v3
Priority: Claude Sonnet 4 → GPT-4o → Gemini 2.0 Flash → Groq → Command R+
Claude leads for accuracy. GPT-4o for depth. Gemini for speed. Groq as fast fallback.
Every call injects pre-computed statistical evidence — AI cannot invent numbers.
"""
from __future__ import annotations
import logging
from typing import Optional
from config.settings import settings
from app.models.schemas import LLMProvider

logger = logging.getLogger(__name__)

INDUSTRY_CONTEXTS = {
    "finance": """You are a CFO-level financial data analyst with 20 years experience.
Focus on: revenue, profitability, cash flow, risk, ROI, EBITDA, NPV, liquidity ratios.
Ground every claim in exact numbers. Flag anomalies, fraud signals, covenant risks.
Reference causal flags, bias warnings and alternative explanations where provided.""",
    "education": """You are a senior education data analyst and institutional researcher.
Focus on: pass rates, dropout predictors, cohort retention, fee collection, assessment analytics.
Ground every claim in exact numbers. Identify at-risk segments with evidence.""",
    "supply_chain": """You are a supply chain director with operations research expertise.
Focus on: OTIF, DSI, reorder points, EOQ, supplier reliability, stockout risk.
Ground every claim in exact numbers. Quantify stockout risks and carrying costs.""",
    "procurement": """You are a Chief Procurement Officer with category management expertise.
Focus on: spend concentration, maverick spend, savings rate, PO cycle time, vendor performance.
Ground every claim in exact numbers. Rank vendors by risk and value.""",
    "healthcare": """You are a healthcare analytics director with clinical operations expertise.
Focus on: LOS, readmission %, ALOS, bed utilisation, cost per patient, diagnostic accuracy.
Ground every claim in exact numbers. Prioritise patient safety signals.""",
    "mining": """You are a mining engineering analyst with operations expertise.
Focus on: head grade, recovery rate, strip ratio, TRIFR, OEE, tonnes per man-shift.
Ground every claim in exact numbers. Quantify cost per recovered unit.""",
    "petroleum": """You are a petroleum engineering analyst with reservoir expertise.
Focus on: BOE/day, GOR, water cut, decline curves, lifting cost, RRR, facility uptime.
Ground every claim in exact numbers. Flag production decline signals.""",
    "retail": """You are a retail analytics director with commercial expertise.
Focus on: GMV, basket size, conversion rate, inventory turnover, shrinkage, churn rate, LTV.
Ground every claim in exact numbers. Segment by customer cohort and product category.""",
    "agriculture": """You are an agricultural economist and precision farming analyst.
Focus on: yield per hectare, input-output ratio, price volatility, growing degree days.
Ground every claim in exact numbers. Model profit sensitivity to weather and price.""",
    "manufacturing": """You are a manufacturing excellence director with lean/six sigma expertise.
Focus on: OEE, PPM defects, MTTR, MTBF, cycle time, first pass yield, capacity utilisation.
Ground every claim in exact numbers. Rank defect types by frequency and cost.""",
    "ngo": """You are a MEAL director with UN/INGO expertise.
Focus on: cost per beneficiary, outcome indicators, donor LTV, budget variance, programme effectiveness.
Ground every claim in exact numbers. Assess impact with evidence.""",
    "general": """You are a senior data scientist and business intelligence director.
Adapt metrics and terminology to the specific industry and dataset.
Ground every claim in exact numbers from the data provided.""",
}

ELITE_SYSTEM = """You are DataMind Agent — an elite AI business analyst at the level of a McKinsey partner combined with a PhD-level data scientist.

═══════════════════════════════════════════════════════
RULE 0 — VERIFY BEFORE YOU WRITE (MOST IMPORTANT RULE)
═══════════════════════════════════════════════════════
- Before writing any number, verify it matches the PRE-COMPUTED EVIDENCE block exactly
- If your own calculation differs from the evidence block, USE THE EVIDENCE BLOCK NUMBER
- NEVER invent, estimate, or independently calculate any number
- Every statistic you write must appear verbatim in the evidence block
- If a number is not in the evidence block, say "not available in dataset" — do not guess

═══════════════════════════════════════════════════════
RULE 1 — GROUND EVERY CLAIM IN DATA
═══════════════════════════════════════════════════════
- Never say "likely", "probably", "may" without citing specific evidence
- Quote exact numbers: "12.4% of values", "3 out of 47 records", "Z-score of 2.8σ"
- Cite the method: "Z-score analysis", "OLS regression (p=0.003)", "Pearson r=0.87"
- If the data does not support a claim, do not make the claim

═══════════════════════════════════════════════════════
RULE 2 — CONFIDENCE SCORES ON EVERY FINDING
═══════════════════════════════════════════════════════
- State confidence for each finding: High (>80%), Medium (60–80%), Low (<60%)
- Explain what drives the confidence
- Reduce confidence for small samples, high missing data, or non-significant p-values

═══════════════════════════════════════════════════════
RULE 3 — IMPACT RANKING IS MANDATORY
═══════════════════════════════════════════════════════
- Rank every finding by business impact (1 = highest)
- Explain exactly why each issue affects KPIs
- Quantify: "This anomaly inflates the mean by 18%"

═══════════════════════════════════════════════════════
RULE 4 — CAUSAL REASONING
═══════════════════════════════════════════════════════
- Reference all confounders, Granger causality signals, and discontinuity signals
- Distinguish correlation from causation explicitly
- Never claim causation from observational data alone

═══════════════════════════════════════════════════════
RULE 5 — BIAS AUDIT
═══════════════════════════════════════════════════════
- Reference survivorship bias, selection bias, and missingness mechanism warnings
- State whether imputation is valid
- Flag impossible values

═══════════════════════════════════════════════════════
RULE 6 — SEGMENTATION
═══════════════════════════════════════════════════════
- Break every finding down by available categorical variables
- Include Cohen's d effect sizes
- Identify which segment is driving each issue

═══════════════════════════════════════════════════════
RULE 7 — ALTERNATIVE EXPLANATIONS
═══════════════════════════════════════════════════════
- For every major finding, provide 2–3 competing explanations
- State which is most supported by the data and why

═══════════════════════════════════════════════════════
RULE 8 — SELF-AUDIT
═══════════════════════════════════════════════════════
- State what assumptions you are making
- State what you could be wrong about
- Reference power analysis — is the sample adequate?

═══════════════════════════════════════════════════════
RESPONSE FORMAT — USE EXACTLY THIS STRUCTURE
═══════════════════════════════════════════════════════

## EXECUTIVE SUMMARY
One paragraph. The single most important insight. End with one actionable sentence.

---

## DATA QUALITY & BIAS AUDIT
Exact counts: rows, columns, missing values, duplicates, outliers capped.
Any bias warnings from the audit.

---

## KEY FINDINGS (ranked by impact)

### Finding 1 — [Title] | Impact: HIGH | Confidence: [X]%
**Evidence:** [Exact numbers from evidence block — z-scores, p-values, effect sizes]
**Business Impact:** [Specific KPI effect, quantified]
**Affected Segments:** [Which segments, with Cohen's d]
**Causal Note:** [Causal or correlational? Confounders?]
**Alternative Explanations:** [2–3 competing theories]
**Recommended Action:** [Specific, measurable action]

### Finding 2 — [Title] | Impact: MEDIUM | Confidence: [X]%
[Same structure]

---

## CAUSAL ANALYSIS
Address all confounder flags, Granger signals, discontinuity signals.
State what can and cannot be inferred causally.

---

## SEGMENT ANALYSIS
Break down every key metric by all available categories.
Exact numbers, p-values, effect sizes per segment.

---

## CORRELATIONS
r-value, Bonferroni-corrected p-value, sample size for each pair.
Distinguish spurious from meaningful.

---

## TIME SERIES INSIGHTS
Structural breaks, seasonality, autocorrelation, regime changes.

---

## RECOMMENDATIONS (numbered, prioritised)
1. [Action] — [Why] — [How to measure] — [Confidence: X%]
2. [Action] — [Why] — [How to measure] — [Confidence: X%]

---

## UNCERTAINTY & CAVEATS
Sample size adequacy, multiple comparison corrections, limitations.

---

## SELF-AUDIT
What I could be wrong about. What I tested and rejected. Domain context needed."""


class EliteLLMService:
    """
    Elite LLM service — Claude leads, GPT-4o backs up, Gemini for speed,
    Groq as fast fallback, Command R+ as last resort.
    Statistical evidence injected into every prompt to prevent hallucination.
    """

    async def chat(
        self,
        messages: list[dict],
        industry: str = "general",
        provider: LLMProvider = LLMProvider.anthropic,
        model: Optional[str] = None,
        max_tokens: int = 3000,
        temperature: float = 0.05,
        elite_context: Optional[dict] = None,
    ) -> tuple[str, int, str]:
        system = ELITE_SYSTEM + "\n\n" + INDUSTRY_CONTEXTS.get(industry, INDUSTRY_CONTEXTS["general"])

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

    def _build_chain(self, preferred_provider, preferred_model):
        """
        Fixed priority chain — Claude leads for accuracy.
        Preference parameter only affects which model Claude/GPT uses,
        not the order of providers.
        """
        anthropic_key = getattr(settings, 'ANTHROPIC_API_KEY', None)
        openai_key    = getattr(settings, 'OPENAI_API_KEY', None)
        google_key    = getattr(settings, 'GOOGLE_API_KEY', None)
        groq_key      = getattr(settings, 'GROQ_API_KEY', None)
        cohere_key    = getattr(settings, 'COHERE_API_KEY', None)

        chain = []

        # 1. Claude Sonnet 4 — best accuracy, careful reasoning
        if anthropic_key:
            chain.append((self._anthropic, "Claude Sonnet 4", "claude-sonnet-4-20250514"))

        # 2. GPT-4o — strong structured analysis
        if openai_key:
            chain.append((self._openai, "GPT-4o", "gpt-4o"))

        # 3. Gemini 2.0 Flash — fast, good for large contexts
        if google_key:
            chain.append((self._gemini, "Gemini 2.0 Flash", "gemini-2.0-flash"))

        # 4. Groq llama-3.3-70b — very fast fallback
        if groq_key:
            chain.append((self._groq, "Groq llama-3.3-70b", "llama-3.3-70b-versatile"))

        # 5. Command R+ — last resort
        if cohere_key:
            chain.append((self._cohere, "Command R+", "command-r-plus"))

        if not chain:
            chain.append((self._no_keys_error, "no-provider", "none"))

        return chain

    def _inject_elite_context(self, messages: list[dict], ctx: dict) -> list[dict]:
        """
        Inject pre-computed statistical evidence into the user message.
        This forces the AI to reference real numbers instead of hallucinating.
        """
        lines = [
            "══════════════════════════════════════════════════",
            "PRE-COMPUTED STATISTICAL EVIDENCE",
            "YOU MUST USE ONLY THESE NUMBERS IN YOUR RESPONSE.",
            "DO NOT CALCULATE OR ESTIMATE ANY NUMBER INDEPENDENTLY.",
            "══════════════════════════════════════════════════",
            "",
            f"DATASET: {ctx.get('row_count','?')} rows × {ctx.get('col_count','?')} columns",
            f"COLUMNS AVAILABLE: {', '.join(ctx.get('columns',[])[:12])}",
            "",
        ]

        # Distributions
        if ctx.get("distributions"):
            lines.append("DISTRIBUTIONS (use these exact values):")
            for col, d in list(ctx["distributions"].items())[:6]:
                lines.append(
                    f"  {col}: n={d['count']}, mean={d['mean']}, median={d['median']}, "
                    f"std={d['std']}, min={d['min']}, max={d['max']}, "
                    f"p25={d['p25']}, p75={d['p75']}, p95={d['p95']}, "
                    f"missing={d['missing_pct']}%, cv={d.get('cv_pct','?')}%, "
                    f"shape={d['distribution_shape']}, normal={d.get('is_normal','?')}"
                )

        # Anomalies
        anomaly_findings = [f for f in ctx.get("findings",[]) if f["type"]=="anomaly"]
        if anomaly_findings:
            lines.append("\nANOMALIES DETECTED:")
            for f in anomaly_findings[:5]:
                ev = f["evidence"]
                lines.append(
                    f"  {f['column']}: {ev['anomaly_count']} anomalous records "
                    f"({ev['anomaly_pct']}% of {ev.get('total_records', ctx.get('row_count','?'))} records), "
                    f"normal range=[{ev['normal_range'][0]}, {ev['normal_range'][1]}], "
                    f"values found={ev['anomaly_values'][:3]}, "
                    f"impact on mean=+{ev['impact_on_mean_pct']}%, "
                    f"max Z-score={ev['z_score_max']}σ, "
                    f"confidence={round(f['confidence']*100)}%, "
                    f"impact={f['impact_score']}/10"
                )
                if ev.get("context_rows"):
                    lines.append(f"    Anomalous row samples: {ev['context_rows'][:2]}")

        # Trends
        trend_findings = [f for f in ctx.get("findings",[]) if f["type"]=="trend"]
        if trend_findings:
            lines.append("\nTREND ANALYSIS (OLS regression):")
            for f in trend_findings[:4]:
                ev = f["evidence"]
                sig = "SIGNIFICANT" if ev.get('statistically_significant') else "NOT significant"
                brk = f", STRUCTURAL BREAK at period {ev.get('structural_break_at_period')}" if ev.get('structural_break_detected') else ""
                lines.append(
                    f"  {f['column']}: total change={ev['total_change_pct']}% over {ev['period_count']} periods, "
                    f"slope={ev['slope_per_period']}/period, R²={ev['r_squared']}, "
                    f"p={ev.get('p_value','?')} ({sig}), "
                    f"first={ev['first_value']}, last={ev['last_value']}{brk}, "
                    f"confidence={round(f['confidence']*100)}%"
                )

        # Correlations
        if ctx.get("correlations"):
            lines.append("\nCORRELATIONS (Pearson + Bonferroni correction):")
            for c in ctx["correlations"][:5]:
                sig = "SIGNIFICANT after Bonferroni" if c.get("significant_after_correction") else "not significant after Bonferroni"
                lines.append(
                    f"  {c['col1']} vs {c['col2']}: r={c['correlation']} ({c['strength']} {c['direction']}), "
                    f"p_raw={c['p_value_raw']}, p_corrected={c.get('p_value_bonferroni_corrected','?')} ({sig}), "
                    f"n={c['n_observations']} | {c['interpretation']}"
                )

        # Segmentation
        if ctx.get("segmentation"):
            lines.append("\nSEGMENTATION (t-test + Cohen's d):")
            for seg_col, seg_data in list(ctx["segmentation"].items())[:3]:
                lines.append(f"  By {seg_col} ({seg_data['unique_segments']} segments):")
                for metric, segments in list(seg_data["metrics"].items())[:2]:
                    lines.append(f"    {metric}:")
                    for s in segments[:6]:
                        sig = " [DIFF]" if s["statistically_different"] else ""
                        lines.append(
                            f"      Rank {s['rank']} {s['segment']}: mean={s['mean']}, "
                            f"n={s['count']} ({s['pct_of_total']}%), "
                            f"deviation={s['deviation_from_overall_pct']:+.1f}%{sig}, "
                            f"p={s['p_value']}, Cohen's d={s.get('cohens_d','?')} ({s.get('effect_size','?')} effect)"
                        )

        # Causal analysis
        causal = ctx.get("causal_analysis", {})
        if causal:
            lines.append("\nCAUSAL ANALYSIS FLAGS:")
            for c in causal.get("potential_confounders", [])[:3]:
                lines.append(f"  CONFOUNDER: {c['warning']} (p={c['p_value']})")
            for g in causal.get("granger_causality_signals", [])[:3]:
                lines.append(f"  GRANGER SIGNAL: {g['note']} (lag r={g['lag_correlation']}, p={g['p_value']})")
            for r in causal.get("regression_discontinuity_signals", [])[:2]:
                lines.append(f"  DISCONTINUITY: {r['note']}")

        # Bias audit
        bias = ctx.get("bias_audit", {})
        if bias:
            lines.append("\nBIAS AUDIT:")
            for col, m in list(bias.get("missingness_mechanism", {}).items())[:3]:
                lines.append(f"  {col} missingness: {m['likely_mechanism']} — {m['recommendation']}")
            for iv in bias.get("impossible_values", [])[:2]:
                lines.append(f"  IMPOSSIBLE VALUE: {iv['issue']} — {iv['suggestion']}")
            surv = bias.get("survivorship_bias_warning", {})
            if surv:
                lines.append(f"  SURVIVORSHIP BIAS WARNING: {surv['note']}")

        # Advanced statistics
        advanced = ctx.get("advanced_statistics", {})
        if advanced:
            lines.append("\nADVANCED STATISTICS:")
            for p in advanced.get("power_analysis", [])[:2]:
                lines.append(
                    f"  Power ({p['column']}): n={p['n']}, "
                    f"min detectable effect={p['min_detectable_effect']} "
                    f"({p['min_detectable_effect_pct_of_mean']}% of mean), "
                    f"adequate power={'YES' if p['adequate_power'] else 'NO — underpowered'}"
                )
            mc = advanced.get("multiple_comparison_correction", {})
            if mc:
                lines.append(f"  Multiple comparisons: {mc['n_tests_run']} tests, Bonferroni α={mc['alpha_adjusted']}, note={mc['note']}")
            for k, v in list(advanced.get("effect_sizes", {}).items())[:2]:
                lines.append(f"  Effect size ({k}): Cohen's d={v['cohens_d']} ({v['magnitude']}) — {v['interpretation']}")

        # Time series
        ts = ctx.get("time_series_intelligence", {})
        if ts:
            lines.append("\nTIME SERIES INTELLIGENCE:")
            for col, t in list(ts.items())[:2]:
                sb = t.get("structural_break", {})
                lines.append(
                    f"  {col}: {sb.get('interpretation','?')}, "
                    f"autocorr_lag1={t.get('autocorrelation_lag1','?')}, "
                    f"seasonality={'YES' if t.get('seasonality_signal') else 'NO'}, "
                    f"trend_stability={t.get('trend_stability','?')}"
                )

        # Alternative explanations
        alt = ctx.get("alternative_explanations", {})
        if alt:
            lines.append("\nALTERNATIVE EXPLANATIONS (address these in your response):")
            for col, alts in list(alt.items())[:3]:
                lines.append(f"  {col}:")
                for a in alts[:3]:
                    lines.append(f"    - {a}")

        # Impact ranking
        if ctx.get("impact_ranking"):
            lines.append("\nIMPACT RANKING (pre-computed):")
            for i, imp in enumerate(ctx["impact_ranking"][:5], 1):
                lines.append(
                    f"  #{i} {imp['column']}: score={imp['score']}/10, "
                    f"confidence={round(imp['confidence']*100)}%, "
                    f"issue={imp['primary_issue']}"
                )

        # Uncertainty
        if ctx.get("uncertainty"):
            lines.append("\nUNCERTAINTY FLAGS:")
            for u in ctx["uncertainty"][:5]:
                lines.append(f"  - {u['issue']}: {u['detail'][:160]} (adj={u.get('confidence_adjustment',0)})")

        # Data grounding
        dg = ctx.get("data_grounding", {})
        if dg:
            lines.append(
                f"\nSUMMARY: {dg.get('total_data_points','?')} data points, "
                f"{dg.get('total_anomalies_found','?')} anomalies, "
                f"{dg.get('total_correlations_found','?')} correlations, "
                f"{dg.get('segments_analysed','?')} segments, "
                f"{dg.get('causal_flags_raised','?')} causal flags, "
                f"{dg.get('bias_flags_raised','?')} bias flags, "
                f"{dg.get('methods_applied','?')} methods applied, "
                f"top confidence={round(dg.get('highest_confidence_finding',0)*100)}%"
            )

        lines.append("\n══════════════════════════════════════════════════")
        lines.append("END OF EVIDENCE BLOCK — NOW WRITE YOUR ANALYSIS")
        lines.append("══════════════════════════════════════════════════\n")

        evidence_block = "\n".join(lines)
        augmented = []
        for msg in messages:
            if msg["role"] == "user":
                augmented.append({"role": "user", "content": evidence_block + "\n" + msg["content"]})
            else:
                augmented.append(msg)
        return augmented

    # ── PROVIDER IMPLEMENTATIONS ──────────────────────────────────────────────

    async def _no_keys_error(self, *args, **kwargs):
        raise Exception("No API keys configured. Add ANTHROPIC_API_KEY in Railway Variables.")

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
            messages=[{"role":"system","content":system}] + messages,
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
        msgs = [{"role":"user" if x["role"]=="user" else "model","parts":[x["content"]]} for x in messages]
        chat = m.start_chat(history=msgs[:-1])
        r = await chat.send_message_async(msgs[-1]["parts"][0])
        tokens = r.usage_metadata.total_token_count if hasattr(r,"usage_metadata") else 0
        return r.text, tokens

    async def _groq(self, messages, system, model, max_tokens, temperature):
        """Groq — fast fallback using OpenAI-compatible API"""
        from openai import AsyncOpenAI
        groq_key = getattr(settings, 'GROQ_API_KEY', None)
        client = AsyncOpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system}] + messages,
            max_tokens=min(max_tokens, 8000),
            temperature=temperature,
        )
        return r.choices[0].message.content, r.usage.total_tokens

    async def _cohere(self, messages, system, model, max_tokens, temperature):
        import cohere
        client = cohere.AsyncClientV2(api_key=settings.COHERE_API_KEY)
        r = await client.chat(
            model=model,
            messages=[{"role":"system","content":system}] + messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        return r.message.content[0].text, r.usage.tokens.input_tokens + r.usage.tokens.output_tokens


llm_service = EliteLLMService()
