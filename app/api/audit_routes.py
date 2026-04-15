from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any
from core.database import audits_col
import datetime

router = APIRouter()

class AuditEntry(BaseModel):
    company_id: str = "DEFAULT"
    data: dict[str, Any] = {}
    notes: str = ""

@router.post("/submit")
def submit_audit(entry: AuditEntry):
    doc = {
        "company_id":   entry.company_id,
        "data":         entry.data,
        "notes":        entry.notes,
        "submitted_at": datetime.datetime.utcnow().isoformat(),
        "status":       "pending_review",
    }
    result = audits_col.insert_one(doc)
    return {"status": "submitted", "audit_id": str(result.inserted_id)}

@router.get("/list/{company_id}")
def list_audits(company_id: str):
    docs = list(audits_col.find({"company_id": company_id}, {"_id": 0}))
    return {"company_id": company_id, "audits": docs, "count": len(docs)}
