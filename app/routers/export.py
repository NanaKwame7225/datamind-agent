"""
DataMind Agent — Document Export Router
POST /api/v1/export/word  — download Word (.docx) report
POST /api/v1/export/pdf   — download PDF report
"""
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import io
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)


class ExportRequest(BaseModel):
    result: dict
    finance: Optional[dict] = None


@router.post("/word")
async def export_word(request: ExportRequest):
    """Generate and download a Word (.docx) report — Times New Roman, 12pt/14pt."""
    try:
        from app.services.document_service import document_service
        doc_bytes = document_service.build_word_report(request.result, request.finance)
        industry = request.result.get("industry", "report").replace(" ", "_")
        filename = f"DataMind_{industry}_{datetime.now().strftime('%Y%m%d')}.docx"
        return StreamingResponse(
            io.BytesIO(doc_bytes),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        logger.error(f"Word export error: {e}", exc_info=True)
        raise HTTPException(500, f"Word export failed: {str(e)}")


@router.post("/pdf")
async def export_pdf(request: ExportRequest):
    """Generate and download a PDF report — Times New Roman, 12pt/14pt."""
    try:
        from app.services.document_service import document_service
        pdf_bytes = document_service.build_pdf_report(request.result, request.finance)
        industry = request.result.get("industry", "report").replace(" ", "_")
        filename = f"DataMind_{industry}_{datetime.now().strftime('%Y%m%d')}.pdf"
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        logger.error(f"PDF export error: {e}", exc_info=True)
        raise HTTPException(500, f"PDF export failed: {str(e)}")
