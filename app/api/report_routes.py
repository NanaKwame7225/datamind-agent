from fastapi import APIRouter
from core.report_engine import generate_audit_report

router = APIRouter()

@router.post("/generate")
def generate(data: dict):
    return generate_audit_report(data)
