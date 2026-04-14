from fastapi import APIRouter
from core.document_ai import (
    extract_text,
    validate_invoice,
    match_receipt_to_transaction
)

router = APIRouter()

@router.post("/ocr")
def ocr(data: dict):
    return {"text": extract_text(data["file"])}


@router.post("/invoice")
def invoice(data: dict):
    return validate_invoice(data["invoice"])


@router.post("/match")
def match(data: dict):
    return match_receipt_to_transaction(
        data["receipt"],
        data["transactions"]
    )
