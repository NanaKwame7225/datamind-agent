from fastapi import APIRouter
from core.report_engine import generate_report

router = APIRouter()

@router.post("/generate")
def generate(data: dict):
    return generate_report(data, data.get("company_id","DEFAULT"))
