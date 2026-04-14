from fastapi import APIRouter

from core.advanced_audit import (
    benford_analysis,
    ratio_analysis,
    trend_analysis
)

from core.ml_fraud_engine import fraud_detector
from core.audit_opinion_engine import generate_audit_opinion

router = APIRouter()

# FRAUD
@router.post("/fraud/train")
def train(data: dict):
    return fraud_detector.train(data["transactions"])


@router.post("/fraud/predict")
def predict(data: dict):
    return fraud_detector.predict(data["transactions"])


# BENFORD
@router.post("/benford")
def benford(data: dict):
    return benford_analysis(data["data"])


# RATIOS
@router.post("/ratios")
def ratios(data: dict):
    return ratio_analysis(data["financials"])


# TREND
@router.post("/trend")
def trend(data: dict):
    return trend_analysis(data["series"])


# AUDIT OPINION
@router.post("/opinion")
def opinion(data: dict):
    return generate_audit_opinion(data["findings"])
