"""
DataMind Agent — Analysis Router (v3)

Fixes over v2:
  • Elite analysis step no longer silently fails on a missing key.
    Every field is read defensively, and the step reports real timing
    whether it succeeds or fails.
  • Correlation key corrected: 'significant_after_correction' (Bonferroni),
    with a fallback to the older 'significant' key.
  • RAG industry benchmarks are injected into the LLM prompt.
  • Memory layer stores each analysis and feeds prior context back in.
  • Causal flags, bias audit and advanced statistics surface as Insights.
  • A partial elite result still yields insights rather than nothing.

All routes require a signed-in user (guest sessions count). Auth is enforced
at the router level via the shared require_user dependency.
"""
import time, logging, json
from pydantic import BaseModel as _BaseModel
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import pandas as pd

from app.models.schemas import (
    AnalysisRequest, AnalysisResponse,
    Metric, Insight, ChartData, PipelineStep, LLMProvider
)
from app.services.llm_service import llm_service
from app.services.analysis_service import analysis_service
from app.services.viz_service import viz_service
from app.routers.auth import require_user

router = APIRouter(dependencies=[Depends(require_user)])
logger = logging.getLogger(__name__)


class ClusterRequest(_BaseModel):
    features: list[str]
    n_clusters: int = 4
    data: list[dict]


def _step(name, tool, status="done", ms=0.0, preview=None):
    return PipelineStep(name=name, tool=tool, status=status,
                        duration_ms=round(ms, 1), output_preview=preview)


def _g(d, *keys, default=None):
    """Safe nested get. _g(d, 'a', 'b') == d['a']['b'] or default."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _count_segments(segmentation) -> int:
    """Count segments without assuming every entry has a 'metrics' key."""
    if not isinstance(segmentation, dict):
        return 0
    total = 0
    for v in segmentation.values():
        if isinstance(v, dict):
            m = v.get("metrics")
            if isinstance(m, dict):
                total += len(m)
            elif v.get("unique_segments"):
                total += int(v["unique_segments"])
    return total


# ── Insight builders — each is independently guarded ──────────────────────────

def _findings_to_insights(elite) -> list[Insight]:
    out = []
    for f in (elite.get("findings") or [])[:6]:
        try:
            ev = f.get("evidence", {}) or {}
            conf = float(f.get("confidence", 0.7))
            ftype = f.get("type", "")
            col = str(f.get("column", "")).replace("_", " ")

            if ftype == "anomaly":
                rng = ev.get("normal_range", ["?", "?"])
                lo, hi = (rng + ["?", "?"])[:2] if isinstance(rng, list) else ("?", "?")
                body = (
                    f"Detected {ev.get('anomaly_count','?')} anomalous records "
                    f"({ev.get('anomaly_pct','?')}% of data). "
                    f"Normal range: [{lo}, {hi}]. "
                    f"Anomalous values: {(ev.get('anomaly_values') or [])[:3]}. "
                    f"These inflate the average by {ev.get('impact_on_mean_pct','?')}%. "
                    f"Max Z-score: {ev.get('z_score_max','?')}σ."
                )
            elif ftype == "trend":
                sig = ("statistically significant" if ev.get("statistically_significant")
                       else "not yet statistically significant")
                brk = ""
                if ev.get("structural_break_detected"):
                    brk = f" A structural break was detected at period {ev.get('structural_break_at_period','?')}."
                body = (
                    f"Over {ev.get('period_count','?')} periods, {col} changed "
                    f"{ev.get('total_change_pct','?')}% (from {ev.get('first_value','?')} "
                    f"to {ev.get('last_value','?')}). R² = {ev.get('r_squared','?')} — "
                    f"trend is {sig} (p = {ev.get('p_value','?')}).{brk}"
                )
            else:
                body = f.get("body") or f.get("title", "")

            out.append(Insight(
                title=f.get("title", "Finding"),
                body=body,
                severity=f.get("severity", "info"),
                source=f"{f.get('method','Statistical analysis')} · Confidence: {round(conf*100)}%",
                confidence=conf,
            ))
        except Exception as e:
            logger.debug(f"Skipped a finding: {e}")
    return out


def _correlations_to_insights(elite) -> list[Insight]:
    out = []
    for c in (elite.get("correlations") or [])[:2]:
        try:
            # v2 used 'significant'; v3 uses 'significant_after_correction'
            sig = c.get("significant_after_correction", c.get("significant", False))
            if not sig:
                continue
            p_show = c.get("p_value_bonferroni_corrected", c.get("p_value_raw", c.get("p_value", "?")))
            out.append(Insight(
                title=(f"Strong relationship: {str(c.get('col1','')).replace('_',' ')} ↔ "
                       f"{str(c.get('col2','')).replace('_',' ')}"),
                body=(
                    f"Pearson r = {c.get('correlation','?')} "
                    f"({c.get('strength','?')} {c.get('direction','?')} correlation), "
                    f"p = {p_show} after Bonferroni correction, "
                    f"based on {c.get('n_observations','?')} observations. "
                    f"{c.get('interpretation','')}"
                ),
                severity="info",
                source="Pearson correlation · Bonferroni corrected · scipy.stats",
                confidence=0.90,
            ))
        except Exception as e:
            logger.debug(f"Skipped a correlation: {e}")
    return out


def _causal_to_insights(elite) -> list[Insight]:
    out = []
    causal = elite.get("causal_analysis") or {}
    try:
        for c in (causal.get("potential_confounders") or [])[:2]:
            out.append(Insight(
                title="Possible confounding variable",
                body=f"{c.get('warning','')} (p = {c.get('p_value','?')}). "
                     "A relationship you observe may be driven by this third variable rather than a direct effect.",
                severity="warning",
                source="Confounder detection · ANOVA",
                confidence=0.75,
            ))
        for g in (causal.get("granger_causality_signals") or [])[:1]:
            out.append(Insight(
                title="Lead-lag signal detected",
                body=f"{g.get('note','')} (lag correlation r = {g.get('lag_correlation','?')}, "
                     f"p = {g.get('p_value','?')}). This is suggestive of a lead-lag relationship, "
                     "not proof of causation.",
                severity="info",
                source="Granger-style lag analysis",
                confidence=0.65,
            ))
        for r in (causal.get("regression_discontinuity_signals") or [])[:1]:
            out.append(Insight(
                title="Discontinuity at a threshold",
                body=str(r.get("note", "")),
                severity="warning",
                source="Regression discontinuity screen",
                confidence=0.60,
            ))
    except Exception as e:
        logger.debug(f"Causal insights skipped: {e}")
    return out


def _bias_to_insights(elite) -> list[Insight]:
    out = []
    bias = elite.get("bias_audit") or {}
    try:
        for col, m in list((bias.get("missingness_mechanism") or {}).items())[:2]:
            out.append(Insight(
                title=f"Missing data in {col.replace('_',' ')} is not random",
                body=f"Likely mechanism: {m.get('likely_mechanism','?')}. {m.get('recommendation','')}",
                severity="warning",
                source="Missingness mechanism test · point-biserial",
                confidence=0.70,
            ))
        for iv in (bias.get("impossible_values") or [])[:2]:
            out.append(Insight(
                title="Impossible values present",
                body=f"{iv.get('issue','')} {iv.get('suggestion','')}",
                severity="critical",
                source="Range validation",
                confidence=0.95,
            ))
        surv = bias.get("survivorship_bias_warning") or {}
        if surv:
            out.append(Insight(
                title="Possible survivorship bias",
                body=str(surv.get("note", "")),
                severity="warning",
                source="Survivorship bias screen",
                confidence=0.65,
            ))
    except Exception as e:
        logger.debug(f"Bias insights skipped: {e}")
    return out


def _power_to_insights(elite) -> list[Insight]:
    out = []
    adv = elite.get("advanced_statistics") or {}
    try:
        for p in (adv.get("power_analysis") or [])[:1]:
            if p.get("adequate_power"):
                continue
            out.append(Insight(
                title="Sample size may be too small to detect real effects",
                body=(
                    f"With n = {p.get('n','?')}, the smallest effect detectable at 80% power "
                    f"is {p.get('min_detectable_effect','?')} "
                    f"({p.get('min_detectable_effect_pct_of_mean','?')}% of the mean). "
                    "Effects smaller than this could exist and go unseen."
                ),
                severity="warning",
                source="Power analysis · α=0.05, power=0.80",
                confidence=0.85,
            ))
    except Exception as e:
        logger.debug(f"Power insights skipped: {e}")
    return out


def _uncertainty_to_insights(elite) -> list[Insight]:
    out = []
    for u in (elite.get("uncertainty") or [])[:2]:
        try:
            out.append(Insight(
                title=f"Data limitation: {u.get('issue','')}",
                body=str(u.get("detail", "")),
                severity="warning",
                source="Self-audit",
                confidence=1.0,
            ))
        except Exception:
            pass
    return out


def _elite_metrics(elite) -> list[Metric]:
    out = []
    dg = elite.get("data_grounding") or {}
    try:
        if dg.get("total_rows_analysed") is not None:
            out.append(Metric(label="Rows analysed", value=str(dg["total_rows_analysed"])))
        out.append(Metric(label="Findings detected", value=str(len(elite.get("findings") or []))))
        if dg.get("segments_analysed") is not None:
            out.append(Metric(label="Segments checked", value=str(dg["segments_analysed"])))
        top = dg.get("highest_confidence_finding") or 0
        if top > 0:
            out.append(Metric(
                label="Top confidence",
                value=f"{round(top*100)}%",
                trend="up" if top > 0.8 else "flat",
            ))
    except Exception as e:
        logger.debug(f"Elite metrics skipped: {e}")
    return out


# ── Main endpoint ─────────────────────────────────────────────────────────────

def _data_facts(df, max_chars: int = 4500, query: str = "") -> str:
    """
    Real facts from the data so the model can cite specifics instead of
    generalising. Critically this includes the TOP and BOTTOM rows by the main
    metric — "which X is highest?" is unanswerable from a head() sample, since
    the leader may sit anywhere in the file.
    """
    import pandas as pd
    parts = []
    num = df.select_dtypes(include="number")

    # Pick the metric that matters: prefer aggregates, ignore ids/years.
    metric = None
    if not num.empty:
        skip = ("rank", "id", "index", "year", "no", "number", "code")
        cands = [c for c in num.columns if not any(s in str(c).lower() for s in skip)]
        if not cands:
            cands = list(num.columns)
        # A column named in the question wins; else prefer global/total; else biggest sum
        ql = (query or "").lower()
        named = [c for c in cands if str(c).lower().replace("_", " ") in ql or str(c).lower() in ql]
        whole = [c for c in cands if any(w in str(c).lower()
                 for w in ("global", "total", "overall", "combined", "gross", "net"))]
        metric = (named or whole or [num[cands].sum().idxmax()])[0]

    # 1. THE LEADERS — the rows that actually answer "which is highest/lowest"
    if metric is not None:
        try:
            label_cols = [c for c in df.columns if c not in num.columns][:3]
            show = ([*label_cols, metric] if label_cols else [metric])
            show = [c for c in dict.fromkeys(show) if c in df.columns]
            top = df.nlargest(15, metric)[show]
            parts.append(f"TOP 15 ROWS BY {str(metric).upper()} (the highest in the WHOLE dataset, "
                         f"not a sample — use these to answer 'which is highest/top/best'):")
            parts.append(top.to_string(index=False, max_colwidth=34)[:1600])
            bot = df.nsmallest(5, metric)[show]
            parts.append(f"\nBOTTOM 5 ROWS BY {str(metric).upper()}:")
            parts.append(bot.to_string(index=False, max_colwidth=34)[:500])
        except Exception as e:
            logger.warning(f"Top/bottom rows failed: {e}")

    # 2. A plain sample for context (file order)
    parts.append(f"\nSAMPLE ROWS (first 8 of {len(df)}, file order — NOT ranked):")
    parts.append(df.head(8).to_string(index=False, max_colwidth=22)[:800])

    # 3. Numeric summaries
    if not num.empty:
        parts.append("\nNUMERIC SUMMARY:")
        try:
            stats = num.describe().T[["min", "50%", "mean", "max"]]
            stats.columns = ["min", "median", "mean", "max"]
            parts.append(stats.to_string(max_colwidth=18)[:900])
        except Exception:
            pass

    # 3. Top categories ranked by the main metric — the "which is best/worst" material
    num_cols = set(num.columns)
    cats = [c for c in df.columns
            if c not in num_cols
            and not pd.api.types.is_datetime64_any_dtype(df[c])
            and 0 < df[c].nunique(dropna=True) <= 60]
    if cats and not num.empty:
        # Ignore id-like / index-like columns when choosing what to rank by
        skip = ("rank", "id", "index", "year", "no", "number", "code")
        candidates = [c for c in num.columns
                      if not any(s in str(c).lower() for s in skip)]
        if not candidates:
            candidates = list(num.columns)
        metric = num[candidates].sum(numeric_only=True).idxmax()
        for c in cats[:3]:
            try:
                top = (df.groupby(c)[metric].sum()
                         .sort_values(ascending=False).head(8))
                parts.append(f"\nTOP {c.upper()} BY TOTAL {metric.upper()}:")
                parts.append(top.to_string()[:600])
            except Exception:
                continue
    out = "\n".join(parts)
    return out[:max_chars]


@router.post("/analyse", response_model=AnalysisResponse)
async def analyse(req: AnalysisRequest):
    t0 = time.perf_counter()
    steps, metrics, insights, charts = [], [], [], []
    df = None
    elite_context = None
    session_id = getattr(req, "session_id", None) or "default"

    # ── STEP 1–2: Ingest and clean ────────────────────────────────────────────
    if req.inline_data:
        t = time.perf_counter()
        raw_df = pd.DataFrame(req.inline_data)

        tc = time.perf_counter()
        try:
            df, cleaning_report = analysis_service.clean_data(raw_df)
            cr_steps = cleaning_report.get("steps", [])
            imp = cr_steps[3].get("columns_imputed", {}) if len(cr_steps) > 3 else {}
            win = cr_steps[4].get("columns_winsorised", {}) if len(cr_steps) > 4 else {}
            dup = cr_steps[2].get("rows_removed", 0) if len(cr_steps) > 2 else 0
            type_issues = len(cleaning_report.get("evidence", {}))
            clean_preview = (f"Duplicates removed: {dup} · Missing filled: {len(imp)} columns · "
                             f"Outliers capped: {len(win)} columns · Type issues flagged: {type_issues}")
            clean_status = "done"
        except Exception as e:
            logger.error(f"Cleaning failed: {e}", exc_info=True)
            df = raw_df
            clean_preview = f"Cleaning skipped: {e}"
            clean_status = "error"

        steps.append(_step("Data received", "Pandas",
            ms=(time.perf_counter()-t)*1000,
            preview=f"{len(raw_df)} rows × {len(raw_df.columns)} columns ingested"))
        steps.append(_step("Data cleaning", "Pandas · NumPy · SciPy",
            status=clean_status, ms=(time.perf_counter()-tc)*1000, preview=clean_preview))

        # ── STEP 3: Elite deep analysis ───────────────────────────────────────
        te = time.perf_counter()
        try:
            elite_context = analysis_service.elite_analyse(df, req.query, req.industry.value)
            if not isinstance(elite_context, dict):
                raise ValueError(f"elite_analyse returned {type(elite_context).__name__}, expected dict")
        except Exception as e:
            logger.error(f"elite_analyse raised: {e}", exc_info=True)
            elite_context = None
            steps.append(_step("Elite analysis", "NumPy · SciPy · Scikit-learn",
                status="error", ms=(time.perf_counter()-te)*1000,
                preview=f"Failed: {e}"))

        if elite_context is not None:
            # Each builder is independently guarded, so one bad key
            # can no longer wipe out the whole step.
            insights.extend(_findings_to_insights(elite_context))
            insights.extend(_correlations_to_insights(elite_context))
            insights.extend(_causal_to_insights(elite_context))
            insights.extend(_bias_to_insights(elite_context))
            insights.extend(_power_to_insights(elite_context))
            insights.extend(_uncertainty_to_insights(elite_context))
            metrics.extend(_elite_metrics(elite_context))

            n_find = len(elite_context.get("findings") or [])
            n_corr = len(elite_context.get("correlations") or [])
            n_seg  = _count_segments(elite_context.get("segmentation"))
            n_causal = (len(_g(elite_context, "causal_analysis", "potential_confounders", default=[])) +
                        len(_g(elite_context, "causal_analysis", "granger_causality_signals", default=[])))
            n_bias = len(_g(elite_context, "bias_audit", "missingness_mechanism", default={}))

            steps.append(_step("Elite analysis", "NumPy · SciPy · Scikit-learn",
                status="done", ms=(time.perf_counter()-te)*1000,
                preview=(f"{n_find} findings · {n_corr} correlations · {n_seg} segments · "
                         f"{n_causal} causal flags · {n_bias} bias flags")))

        # ── STEP 4: Quality metrics ────────────────────────────────────────────
        t = time.perf_counter()
        try:
            quality = analysis_service.quality_report(df)
            metrics.insert(0, Metric(
                label="Data quality",
                value=f"{quality['overall_score']}%",
                trend="up" if quality["overall_score"] > 85 else "down",
            ))
            steps.append(_step("Quality scored", "Pandas",
                ms=(time.perf_counter()-t)*1000,
                preview=f"Score: {quality['overall_score']}% · Missing: {quality['missing_cells']} cells"))
        except Exception as e:
            logger.warning(f"Quality report failed: {e}")
            steps.append(_step("Quality scored", "Pandas", status="error",
                ms=(time.perf_counter()-t)*1000, preview=str(e)))
    else:
        steps.append(_step("Data received", "API", ms=1.0, preview="No dataset — AI analysis only"))

    # ── STEP 5: Forecast ──────────────────────────────────────────────────────
    if df is not None and req.enable_forecast:
        t = time.perf_counter()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols and len(df) >= 20:
            try:
                series = df[numeric_cols[0]].dropna()
                fr = analysis_service.forecast_arima(series, 12)
                metrics.append(Metric(
                    label="Next period forecast",
                    value=round(fr["forecast"][0], 2),
                    trend="up" if fr["forecast"][0] > float(series.iloc[-1]) else "down",
                ))
                fig = viz_service.plotly_forecast(
                    series.tolist()[-30:], fr["forecast"],
                    fr["lower_bound"], fr["upper_bound"],
                    f"{numeric_cols[0].replace('_',' ')} — Forecast")
                charts.append(ChartData(
                    chart_type="forecast",
                    title=f"{numeric_cols[0].replace('_',' ')} Forecast",
                    data=fig,
                    description=f"12-period ahead forecast using {fr['model']} · AIC: {fr['aic']}",
                ))
                steps.append(_step("Forecast generated", "Statsmodels ARIMA",
                    ms=(time.perf_counter()-t)*1000,
                    preview=f"Model: {fr['model']} · AIC: {fr['aic']}"))
            except Exception as e:
                logger.warning(f"Forecast failed: {e}")
                steps.append(_step("Forecast generated", "Statsmodels", status="error",
                    ms=(time.perf_counter()-t)*1000, preview=str(e)))

    # ── STEP 6: Charts ────────────────────────────────────────────────────────
    if df is not None and req.enable_viz:
        t = time.perf_counter()
        try:
            auto = viz_service.auto_chart(df)
            for c in auto:
                charts.append(ChartData(chart_type=c["type"], title=c["title"], data=c["figure"]))
            steps.append(_step("Charts built", "Plotly",
                ms=(time.perf_counter()-t)*1000,
                preview=f"{len(auto)} interactive charts generated"))
        except Exception as e:
            logger.warning(f"Viz failed: {e}")
            steps.append(_step("Charts built", "Plotly", status="error",
                ms=(time.perf_counter()-t)*1000, preview=str(e)))

    # ── STEP 7: Retrieve benchmarks + prior context (RAG + Memory) ────────────
    t = time.perf_counter()
    rag_block, memory_block = "", ""

    try:
        from app.services.rag_service import rag_service
        metric_names = list(df.columns) if df is not None else []
        rag_block = rag_service.get_industry_context(req.industry.value, metric_names)
    except Exception as e:
        logger.info(f"RAG context unavailable: {e}")

    try:
        from app.services.memory_service import memory_service
        memory_block = memory_service.get_memory_context(session_id)
    except Exception as e:
        logger.info(f"Memory context unavailable: {e}")

    if rag_block or memory_block:
        steps.append(_step("Knowledge retrieved", "RAG · Memory",
            ms=(time.perf_counter()-t)*1000,
            preview=(f"{'Benchmarks loaded' if rag_block else 'No benchmarks'} · "
                     f"{'Prior analyses recalled' if memory_block else 'No prior context'}")))

    # ── STEP 8: Elite AI narrative ────────────────────────────────────────────
    t = time.perf_counter()
    context_parts = [f"Industry: {req.industry.value}", f"User question: {req.query}"]

    if df is not None:
        try:
            desc = analysis_service.describe(df)
            context_parts.append(
                f"Dataset: {desc['shape']['rows']} rows × {desc['shape']['columns']} columns")
            context_parts.append(f"Columns available: {list(desc['dtypes'].keys())}")
        except Exception:
            context_parts.append(f"Dataset: {len(df)} rows × {len(df.columns)} columns")
            context_parts.append(f"Columns available: {list(df.columns)}")

        # Give the model the ACTUAL data — otherwise it can only speak in generalities.
        try:
            context_parts.append("\n" + _data_facts(df, query=req.query))
        except Exception as e:
            logger.warning(f"Data facts failed: {e}")

    if rag_block:
        context_parts.append("\n" + rag_block)
        context_parts.append(
            "Compare the findings against the benchmarks above. "
            "Where a metric falls outside the acceptable band, say so explicitly and cite the benchmark."
        )
    if memory_block:
        context_parts.append("\n" + memory_block)

    context_parts.append(
        "\n════════ HOW TO ANSWER ════════\n"
        f"Answer THIS exact question: \"{req.query}\"\n"
        "RULES:\n"
        "1. ANSWER THE QUESTION ASKED — do not just describe or list the data. If the "
        "question asks to classify, group, rank, or compare, then DO that classification "
        "and present the result. A list of rows is NOT an answer unless a list was asked "
        "for. Work out the answer, then show the evidence.\n"
        "2. THE 'TOP ROWS' BLOCK IS AUTHORITATIVE. It is computed from the ENTIRE dataset, "
        "not a sample. If asked which is highest/top/best/leading, name the first row of "
        "that block outright — never say the answer 'cannot be determined' or hedge about "
        "not seeing the full data. You have the leaders.\n"
        "2. SPECIFIC, NOT GENERAL. Name the actual rows, categories, and values from the "
        "data above. 'Shooter dominates with 7.07M across 2 titles, led by Asteroids "
        "(4.31M)' — never 'some genres performed well'.\n"
        "3. LEAD WITH THE ANSWER. The first sentence must state the conclusion the "
        "question is asking for, with a concrete figure.\n"
        "4. SHOW THE RECORDS. Support the answer with a markdown table of the relevant "
        "rows — grouped or ranked the way the question implies, with real values.\n"
        "5. CITE REAL NUMBERS everywhere — totals, deltas, percentages, counts computed "
        "from the data above.\n"
        "6. NO FILLER. Delete any sentence that would be true of any dataset.\n"
        "Be precise and evidence-bound. This is an elite analyst's answer to a specific "
        "question — not a data description."
    )

    context_msg = "\n".join(context_parts)
    messages = list(req.conversation_history) + [{"role": "user", "content": context_msg}]

    try:
        narrative, tokens, provider_used = await llm_service.chat(
            messages=messages,
            industry=req.industry.value,
            provider=req.provider,
            model=req.model,
            max_tokens=3000,
            elite_context=elite_context,
        )
        steps.append(_step("AI report written", req.provider.value,
            ms=(time.perf_counter()-t)*1000,
            preview=f"{tokens} tokens · {provider_used}"))
    except Exception as e:
        logger.error(f"LLM failed: {e}", exc_info=True)
        narrative = f"AI analysis unavailable: {e}"
        tokens = 0
        provider_used = req.provider.value
        steps.append(_step("AI report written", provider_used, status="error",
            ms=(time.perf_counter()-t)*1000, preview=str(e)))

    response = AnalysisResponse(
        query=req.query,
        industry=req.industry.value,
        provider=provider_used,
        model=req.model or provider_used,
        narrative=narrative,
        metrics=metrics,
        insights=insights,
        charts=charts,
        pipeline_steps=steps,
        raw_data_preview=(
            json.loads(df.head(6).to_json(orient="records")) if df is not None else None
        ),
        execution_ms=round((time.perf_counter()-t0)*1000, 1),
        tokens_used=tokens,
    )

    # ── STEP 9: Persist to memory (best effort) ───────────────────────────────
    try:
        from app.services.memory_service import memory_service
        memory_service.save_analysis_memory(session_id, {
            "industry": req.industry.value,
            "query": req.query,
            "provider": provider_used,
            "narrative": narrative,
            "insights": [i.model_dump() for i in insights],
            "row_count": len(df) if df is not None else 0,
            "col_count": len(df.columns) if df is not None else 0,
        })
    except Exception as e:
        logger.info(f"Could not persist to memory: {e}")

    return response


# ── Diagnostics ───────────────────────────────────────────────────────────────

@router.post("/elite-debug")
async def elite_debug(data: list[dict], query: str = "test", industry: str = "general"):
    """
    Run elite_analyse in isolation and report exactly what it returns
    or exactly how it fails. Use this when the Elite analysis pipeline
    step reports an error.
    """
    if not data:
        raise HTTPException(400, "Provide data")
    df = pd.DataFrame(data)
    t = time.perf_counter()
    try:
        elite = analysis_service.elite_analyse(df, query, industry)
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc().splitlines()[-8:],
            "duration_ms": round((time.perf_counter()-t)*1000, 1),
        }

    keys = list(elite.keys()) if isinstance(elite, dict) else []
    return {
        "success": True,
        "duration_ms": round((time.perf_counter()-t)*1000, 1),
        "returned_type": type(elite).__name__,
        "top_level_keys": keys,
        "counts": {
            "findings": len(elite.get("findings") or []),
            "correlations": len(elite.get("correlations") or []),
            "segments": _count_segments(elite.get("segmentation")),
            "confounders": len(_g(elite, "causal_analysis", "potential_confounders", default=[])),
            "bias_flags": len(_g(elite, "bias_audit", "missingness_mechanism", default={})),
            "uncertainty": len(elite.get("uncertainty") or []),
        },
        "correlation_keys": (list(elite["correlations"][0].keys())
                             if elite.get("correlations") else []),
        "finding_keys": (list(elite["findings"][0].keys())
                         if elite.get("findings") else []),
        "segmentation_sample": (list(elite["segmentation"].items())[:1]
                                if elite.get("segmentation") else []),
    }


@router.post("/describe")
async def describe_data(data: list[dict]):
    df = pd.DataFrame(data)
    return {"statistics": analysis_service.describe(df), "quality": analysis_service.quality_report(df)}


@router.post("/anomaly")
async def detect_anomaly(column: str, method: str = "zscore", data: list[dict] = []):
    if not data:
        raise HTTPException(400, "Provide data")
    df = pd.DataFrame(data)
    if column not in df.columns:
        raise HTTPException(400, f"Column not found. Available: {df.columns.tolist()}")
    return analysis_service.detect_anomalies(df, column, method)


@router.post("/cluster")
async def cluster_data(req: ClusterRequest):
    if not req.data:
        raise HTTPException(400, "Provide data")
    df = pd.DataFrame(req.data)
    missing = [f for f in req.features if f not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")
    return analysis_service.cluster(df, req.features, req.n_clusters)
