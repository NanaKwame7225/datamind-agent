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
