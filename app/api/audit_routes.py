from fastapi import APIRouter

# =========================
# CORE AUDIT ENGINE IMPORTS
# =========================
from core.advanced_audit import (
    benford_analysis,
    ratio_analysis,
    trend_analysis
)

from core.ml_fraud_engine import fraud_detector
from core.audit_report_generator import generate_audit_report

# =========================
# ROUTER INITIALIZATION
# =========================
router = APIRouter()

# =========================================================
# 🧠 FRAUD DETECTION MODULE (ML-BASED)
# =========================================================

@router.post("/fraud/train")
def train_fraud_model(data: dict):
    """
    Train ML fraud detection model
    """
    return fraud_detector.train(data["transactions"])


@router.post("/fraud/predict")
def predict_fraud(data: dict):
    """
    Predict fraud on transaction dataset
    """
    return fraud_detector.predict(data["transactions"])


# =========================================================
# 📊 BENFORD'S LAW (FRAUD PATTERN DETECTION)
# =========================================================

@router.post("/benford")
def run_benford_analysis(data: dict):
    """
    Detect anomalies using Benford's Law
    """
    return benford_analysis(data["data"])


# =========================================================
# 📈 FINANCIAL RATIO ANALYSIS
# =========================================================

@router.post("/ratios")
def run_ratio_analysis(payload: dict):
    """
    Analyze financial health ratios
    """
    return ratio_analysis(payload["financials"])


# =========================================================
# 📉 TREND ANALYSIS (TIME SERIES AUDITING)
# =========================================================

@router.post("/trend")
def run_trend_analysis(payload: dict):
    """
    Detect financial performance trends
    """
    return trend_analysis(payload["series"])


# =========================================================
# 📄 AUDIT REPORT GENERATION (BIG 4 STYLE OUTPUT)
# =========================================================

@router.post("/report")
def generate_report(payload: dict):
    """
    Generate structured audit PDF report
    """
    return generate_audit_report(
        filename=payload.get("filename", "audit_report.pdf"),
        audit_data=payload["audit_data"]
    )
