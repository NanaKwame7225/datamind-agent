from fastapi import APIRouter
from core.document_ai import (
    extract_text,
    validate_invoice,
    match_receipt_to_transaction
)

router = APIRouter()

# =========================
# OCR EXTRACTION
# =========================
@router.post("/ocr")
def ocr(payload: dict):
    return {"text": extract_text(payload["file"])}


# =========================
# INVOICE VALIDATION
# =========================
@router.post("/invoice")
def invoice(payload: dict):
    return validate_invoice(payload["invoice"])


# =========================
# RECEIPT MATCHING
# =========================
@router.post("/match")
def match(payload: dict):
    return match_receipt_to_transaction(
        payload["receipt"],
        payload["transactions"]
    )
