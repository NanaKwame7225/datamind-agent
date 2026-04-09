import logging, pandas as pd
from app.services.analysis_service import analysis_service
from app.services.viz_service import viz_service

logger = logging.getLogger(__name__)

def run_steps(df, steps):
    results = {}
    for name, fn in steps:
        try:
            results[name] = fn(df)
            logger.info(f"  ✓ {name}")
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

def finance_pipeline(df):
    numeric = df.select_dtypes(include="number").columns.tolist()
    steps = [
        ("descriptive_stats", lambda d: analysis_service.describe(d)),
        ("quality_report",    lambda d: analysis_service.quality_report(d)),
    ]
    if numeric:
        steps.append(("anomaly", lambda d: analysis_service.detect_anomalies(d, numeric[0], "zscore")))
    if numeric and len(df) >= 20:
        steps.append(("forecast", lambda d: analysis_service.forecast_arima(d[numeric[0]].dropna(), 12)))
    steps.append(("charts", lambda d: viz_service.auto_chart(d)))
    return run_steps(df, steps)

def generic_pipeline(df):
    return run_steps(df, [
        ("descriptive_stats", lambda d: analysis_service.describe(d)),
        ("quality_report",    lambda d: analysis_service.quality_report(d)),
        ("charts",            lambda d: viz_service.auto_chart(d)),
    ])

INDUSTRY_PIPELINES = {
    "finance": finance_pipeline,
    "supply_chain": generic_pipeline,
    "education": generic_pipeline,
    "mining": generic_pipeline,
    "petroleum": generic_pipeline,
    "healthcare": generic_pipeline,
}

def get_pipeline(industry: str):
    return INDUSTRY_PIPELINES.get(industry, generic_pipeline)
