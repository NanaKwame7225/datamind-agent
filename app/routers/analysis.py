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
                        duration_ms=round(ms,1), output_preview=preview)

@router.post("/analyse", response_model=AnalysisResponse)
async def analyse(req: AnalysisRequest):
    t0 = time.perf_counter()
    steps, metrics, insights, charts = [], [], [], []
    df = None

    t = time.perf_counter()
    if req.inline_data:
        df = pd.DataFrame(req.inline_data)
        steps.append(_step("Data ingestion", "pandas.DataFrame",
                           ms=(time.perf_counter()-t)*1000,
                           preview=f"{len(df)} rows x {len(df.columns)} cols"))
    else:
        steps.append(_step("Data ingestion", "api", ms=1.0, preview="No dataset — LLM only"))

    if df is not None:
        t = time.perf_counter()
        quality = analysis_service.quality_report(df)
        steps.append(_step("Quality check", "pandas/numpy",
                           ms=(time.perf_counter()-t)*1000,
                           preview=f"Completeness: {quality['completeness_pct']}%"))
        metrics.append(Metric(label="Data Quality Score",
                              value=f"{quality['overall_score']}%",
                              trend="up" if quality["overall_score"] > 85 else "down"))
        if quality["missing_cells"] > 0:
            insights.append(Insight(title="Missing data detected",
                                    body=f"{quality['missing_cells']} missing values found.",
                                    severity="warning", source="Quality scanner"))

    if df is not None:
        t = time.perf_counter()
        desc = analysis_service.describe(df)
        numeric_cols = [c for c,t_ in desc["dtypes"].items() if "float" in t_ or "int" in t_]
        if numeric_cols:
            col = numeric_cols[0]
            metrics.append(Metric(label=f"{col} mean",
                                  value=round(float(df[col].dropna().mean()), 2)))
            metrics.append(Metric(label=f"{col} std dev",
                                  value=round(float(df[col].dropna().std()), 2)))
        steps.append(_step("Statistical analysis", "pandas+statsmodels",
                           ms=(time.perf_counter()-t)*1000))

    if df is not None and req.enable_anomaly_detection:
        t = time.perf_counter()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols[:3]:
            try:
                result = analysis_service.detect_anomalies(df, col, "zscore")
                if result["anomaly_count"] > 0:
                    insights.append(Insight(
                        title=f"Anomaly: {col}",
                        body=f"{result['anomaly_count']} anomalous values ({result['anomaly_pct']}%) detected.",
                        severity="critical" if result["anomaly_pct"] > 5 else "warning",
                        source="Z-score", confidence=0.92))
            except Exception as e:
                logger.warning(f"Anomaly scan failed for {col}: {e}")
        steps.append(_step("Anomaly detection", "sklearn.IsolationForest",
                           ms=(time.perf_counter()-t)*1000))

    if df is not None and req.enable_forecast:
        t = time.perf_counter()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols and len(df) >= 20:
            try:
                series = df[numeric_cols[0]].dropna()
                forecast_result = analysis_service.forecast_arima(series, 12)
                metrics.append(Metric(label="Next period forecast",
                                      value=round(forecast_result["forecast"][0], 2),
                                      trend="up" if forecast_result["forecast"][0] > series.iloc[-1] else "down"))
                fig = viz_service.plotly_forecast(
                    series.tolist()[-30:],
                    forecast_result["forecast"],
                    forecast_result["lower_bound"],
                    forecast_result["upper_bound"],
                    f"{numeric_cols[0]} Forecast")
                charts.append(ChartData(chart_type="forecast",
                                        title=f"{numeric_cols[0]} Forecast", data=fig))
                steps.append(_step("Forecast", "statsmodels.ARIMA",
                                   ms=(time.perf_counter()-t)*1000))
            except Exception as e:
                logger.warning(f"Forecast failed: {e}")

    if df is not None and req.enable_viz:
        t = time.perf_counter()
        try:
            auto = viz_service.auto_chart(df)
            for c in auto:
                charts.append(ChartData(chart_type=c["type"], title=c["title"], data=c["figure"]))
            steps.append(_step("Visualisation", "plotly",
                               ms=(time.perf_counter()-t)*1000,
                               preview=f"{len(auto)} charts"))
        except Exception as e:
            logger.warning(f"Viz failed: {e}")

    t = time.perf_counter()
    context = f"Industry: {req.industry.value}\nQuery: {req.query}"
    if df is not None:
        desc = analysis_service.describe(df)
        context += f"\nDataset: {desc['shape']['rows']} rows x {desc['shape']['columns']} cols"
        context += f"\nColumns: {list(desc['dtypes'].keys())}"
    if metrics:
        context += "\nMetrics: " + ", ".join([f"{m.label}={m.value}" for m in metrics])

    messages = req.conversation_history + [{"role": "user", "content": context}]
    try:
        narrative, tokens = await llm_service.chat(
            messages=messages, industry=req.industry.value,
            provider=req.provider, model=req.model, max_tokens=1500)
    except Exception as e:
        narrative = f"Analysis complete. LLM narrative unavailable: {e}"
        tokens = 0

    steps.append(_step("LLM synthesis", f"{req.provider.value}",
                       ms=(time.perf_counter()-t)*1000, preview=f"{tokens} tokens"))

    return AnalysisResponse(
        query=req.query, industry=req.industry.value,
        provider=req.provider.value, model=req.model or "default",
        narrative=narrative, metrics=metrics, insights=insights,
        charts=charts, pipeline_steps=steps,
        execution_ms=round((time.perf_counter()-t0)*1000, 1), tokens_used=tokens)

@router.post("/describe")
async def describe_data(data: list[dict]):
    df = pd.DataFrame(data)
    return {"statistics": analysis_service.describe(df),
            "quality": analysis_service.quality_report(df)}

@router.post("/anomaly")
async def detect_anomaly(column: str, method: str = "zscore", data: list[dict] = []):
    if not data:
        raise HTTPException(400, "Provide data")
    df = pd.DataFrame(data)
    if column not in df.columns:
        raise HTTPException(400, f"Column '{column}' not found")
    return analysis_service.detect_anomalies(df, column, method)

@router.post("/cluster")
async def cluster_data(req: ClusterRequest):
    if not req.data:
        raise HTTPException(400, "Provide data")
    df = pd.DataFrame(req.data)
    missing = [f for f in req.features if f not in df.columns]
    if missing:
        raise HTTPException(400, f"Features not found: {missing}")
    return analysis_service.cluster(df, req.features, req.n_clusters)
