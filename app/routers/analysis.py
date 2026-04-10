import time, logging, json
from pydantic import BaseModel as _BaseModel
from fastapi import APIRouter, HTTPException
import pandas as pd

from app.models.schemas import (
    AnalysisRequest, AnalysisResponse,
    Metric, Insight, ChartData, PipelineStep, LLMProvider
)
from app.services.llm_service import llm_service
from app.services.analysis_service import analysis_service
from app.services.viz_service import viz_service

router = APIRouter()
logger = logging.getLogger(__name__)


class ClusterRequest(_BaseModel):
    features: list[str]
    n_clusters: int = 4
    data: list[dict]


def _step(name, tool, status="done", ms=0.0, preview=None):
    return PipelineStep(name=name, tool=tool, status=status,
                        duration_ms=round(ms, 1), output_preview=preview)


@router.post("/analyse", response_model=AnalysisResponse)
async def analyse(req: AnalysisRequest):
    t0 = time.perf_counter()
    steps, metrics, insights, charts = [], [], [], []
    df = None
    elite_context = None

    # ── STEP 1: Ingest ────────────────────────────────────────────────────────
    t = time.perf_counter()
    if req.inline_data:
        raw_df = pd.DataFrame(req.inline_data)

        # ── STEP 2: Clean ─────────────────────────────────────────────────────
        tc = time.perf_counter()
        df, cleaning_report = analysis_service.clean_data(raw_df)
        imp = cleaning_report["steps"][3].get("columns_imputed", {})
        win = cleaning_report["steps"][4].get("columns_winsorised", {})
        dup = cleaning_report["steps"][2].get("rows_removed", 0)
        type_issues = len(cleaning_report.get("evidence", {}))
        steps.append(_step("Data received", "Pandas",
            ms=(time.perf_counter()-t)*1000,
            preview=f"{len(raw_df)} rows × {len(raw_df.columns)} columns ingested"))
        steps.append(_step("Data cleaning", "Pandas · NumPy · SciPy",
            ms=(time.perf_counter()-tc)*1000,
            preview=f"Duplicates removed: {dup} · Missing filled: {len(imp)} columns · "
                    f"Outliers capped: {len(win)} columns · Type issues flagged: {type_issues}"))

        # ── STEP 3: Elite deep analysis ───────────────────────────────────────
        te = time.perf_counter()
        try:
            elite_context = analysis_service.elite_analyse(df, req.query, req.industry.value)

            # Convert elite findings to Insight objects
            for f in elite_context["findings"][:6]:
                ev = f.get("evidence", {})
                confidence = f.get("confidence", 0.7)
                impact = f.get("impact_score", 0)

                if f["type"] == "anomaly":
                    body = (
                        f"Detected {ev.get('anomaly_count', '?')} anomalous records "
                        f"({ev.get('anomaly_pct', '?')}% of data). "
                        f"Normal range: [{ev.get('normal_range', ['?','?'])[0]}, {ev.get('normal_range', ['?','?'])[1]}]. "
                        f"Anomalous values found: {ev.get('anomaly_values', [])[:3]}. "
                        f"These values inflate the average by {ev.get('impact_on_mean_pct', '?')}%. "
                        f"Max Z-score: {ev.get('z_score_max', '?')}σ."
                    )
                elif f["type"] == "trend":
                    sig = "statistically significant" if ev.get("statistically_significant") else "not yet statistically significant"
                    body = (
                        f"Over {ev.get('period_count', '?')} periods, {f['column'].replace('_',' ')} changed "
                        f"{ev.get('total_change_pct', '?')}% (from {ev.get('first_value','?')} to {ev.get('last_value','?')}). "
                        f"R² = {ev.get('r_squared', '?')} — trend is {sig} "
                        f"(p = {ev.get('p_value', '?')})."
                    )
                else:
                    body = f["title"]

                insights.append(Insight(
                    title=f["title"],
                    body=body,
                    severity=f.get("severity", "info"),
                    source=f"{f.get('method', 'Statistical analysis')} · Confidence: {round(confidence*100)}%",
                    confidence=confidence,
                ))

            # Convert correlations to insights
            for c in elite_context.get("correlations", [])[:2]:
                if c["significant"]:
                    insights.append(Insight(
                        title=f"Strong relationship: {c['col1'].replace('_',' ')} ↔ {c['col2'].replace('_',' ')}",
                        body=(
                            f"Pearson r = {c['correlation']} ({c['strength']} {c['direction']} correlation), "
                            f"p = {c['p_value']} (statistically significant), "
                            f"based on {c['n_observations']} observations. "
                            f"{c['interpretation']}."
                        ),
                        severity="info",
                        source="Pearson correlation · scipy.stats",
                        confidence=0.90,
                    ))

            # Convert uncertainty flags to insights
            for u in elite_context.get("uncertainty", [])[:2]:
                insights.append(Insight(
                    title=f"Data limitation: {u['issue']}",
                    body=u["detail"],
                    severity="warning",
                    source="Self-audit",
                    confidence=1.0,
                ))

            # Build metrics from elite context
            dg = elite_context["data_grounding"]
            metrics.append(Metric(label="Rows analysed", value=str(dg["total_rows_analysed"])))
            metrics.append(Metric(label="Findings detected", value=str(len(elite_context["findings"]))))
            metrics.append(Metric(label="Segments checked", value=str(dg["segments_analysed"])))
            if dg["highest_confidence_finding"] > 0:
                metrics.append(Metric(
                    label="Top confidence",
                    value=f"{round(dg['highest_confidence_finding']*100)}%",
                    trend="up" if dg["highest_confidence_finding"] > 0.8 else "flat",
                ))

            steps.append(_step("Elite analysis", "NumPy · SciPy · Scikit-learn",
                ms=(time.perf_counter()-te)*1000,
                preview=(
                    f"{len(elite_context['findings'])} findings · "
                    f"{len(elite_context['correlations'])} correlations · "
                    f"{sum(len(v['metrics']) for v in elite_context['segmentation'].values())} segments"
                )))
        except Exception as e:
            logger.warning(f"Elite analysis failed: {e}")
            steps.append(_step("Elite analysis", "NumPy · SciPy", status="error", preview=str(e)))

        # ── STEP 4: Quality metrics ────────────────────────────────────────────
        t = time.perf_counter()
        quality = analysis_service.quality_report(df)
        metrics.insert(0, Metric(
            label="Data quality",
            value=f"{quality['overall_score']}%",
            trend="up" if quality["overall_score"] > 85 else "down",
        ))
        steps.append(_step("Quality scored", "Pandas",
            ms=(time.perf_counter()-t)*1000,
            preview=f"Score: {quality['overall_score']}% · Missing: {quality['missing_cells']} cells"))

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

    # ── STEP 7: Elite AI narrative ─────────────────────────────────────────────
    t = time.perf_counter()
    context_msg = f"Industry: {req.industry.value}\nUser question: {req.query}"
    if df is not None:
        desc = analysis_service.describe(df)
        context_msg += f"\nDataset: {desc['shape']['rows']} rows × {desc['shape']['columns']} columns"
        context_msg += f"\nColumns available: {list(desc['dtypes'].keys())}"

    messages = req.conversation_history + [{"role": "user", "content": context_msg}]

    try:
        narrative, tokens, provider_used = await llm_service.chat(
            messages=messages,
            industry=req.industry.value,
            provider=req.provider,
            model=req.model,
            max_tokens=2500,
            elite_context=elite_context,
        )
        steps.append(_step("AI report written", f"{req.provider.value}",
            ms=(time.perf_counter()-t)*1000,
            preview=f"{tokens} tokens · {provider_used}"))
    except Exception as e:
        narrative = f"AI analysis unavailable: {e}"
        tokens = 0
        provider_used = req.provider.value
        steps.append(_step("AI report", provider_used, status="error", preview=str(e)))

    return AnalysisResponse(
        query=req.query,
        industry=req.industry.value,
        provider=provider_used.split(" ")[0].lower() if " " in provider_used else req.provider.value,
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
