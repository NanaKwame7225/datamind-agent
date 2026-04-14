from fastapi import APIRouter
from core.advanced_audit import (
    benford_analysis,
    ratio_analysis,
    trend_analysis
)

router = APIRouter()

# =========================
# BENFORD'S LAW ENDPOINT
# =========================
@router.post("/benford")
def benford(data: dict):
    return benford_analysis(data["data"])

from core.ml_fraud_engine import fraud_detector

@router.post("/fraud/train")
def train(data: dict):
    return fraud_detector.train(data["transactions"])


@router.post("/fraud/predict")
def predict(data: dict):
    return fraud_detector.predict(data["transactions"])


# =========================
# RATIO ANALYSIS
# =========================
@router.post("/ratios")
def ratios(payload: dict):
    return ratio_analysis(payload["financials"])


# =========================
# TREND ANALYSIS
# =========================
@router.post("/trend")
def trend(payload: dict):
    return trend_analysis(payload["series"])
