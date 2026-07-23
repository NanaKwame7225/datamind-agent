"""
DataMind Agent — Document & Presentation Export Router
POST /api/v1/export/word   — Word (.docx) report, Times New Roman
POST /api/v1/export/pdf    — PDF report, Times New Roman
POST /api/v1/export/pptx   — PowerPoint (.pptx) slide deck
POST /api/v1/export/xlsx   — Excel (.xlsx) workbook (Summary/Metrics/Insights/Data/Finance/Charts)
All accept optional chart_images (base64 PNGs captured client-side).
"""
import logging, io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
router = APIRouter()
logger = logging.getLogger(__name__)
class ExportRequest(BaseModel):
    result: dict
    finance: Optional[dict] = None
    chart_images: Optional[list] = None   # [{title, subtitle, image(base64)}]
    title: Optional[str] = None           # custom document heading
    subtitle: Optional[str] = None        # custom document subheading
    line_spacing: Optional[float] = 1.5   # 1.5 or 2.0
@router.post("/word")
async def export_word(request: ExportRequest):
    try:
        from app.services.document_service import document_service
        doc_bytes = document_service.build_word_report(request.result, request.finance, request.chart_images,
            doc_title=request.title, doc_subtitle=request.subtitle, line_spacing=request.line_spacing)
        industry = request.result.get("industry", "report").replace(" ", "_")
        filename = f"DataMind_{industry}_{datetime.now().strftime('%Y%m%d')}.docx"
        return StreamingResponse(io.BytesIO(doc_bytes),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    except Exception as e:
        logger.error(f"Word export error: {e}", exc_info=True)
        raise HTTPException(500, f"Word export failed: {str(e)}")
@router.post("/pdf")
async def export_pdf(request: ExportRequest):
    try:
        from app.services.document_service import document_service
        pdf_bytes = document_service.build_pdf_report(request.result, request.finance, request.chart_images,
            doc_title=request.title, doc_subtitle=request.subtitle, line_spacing=request.line_spacing)
        industry = request.result.get("industry", "report").replace(" ", "_")
        filename = f"DataMind_{industry}_{datetime.now().strftime('%Y%m%d')}.pdf"
        return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    except Exception as e:
        logger.error(f"PDF export error: {e}", exc_info=True)
        raise HTTPException(500, f"PDF export failed: {str(e)}")
@router.post("/pptx")
async def export_pptx(request: ExportRequest):
    try:
        from app.services.pptx_service import pptx_service
        pptx_bytes = pptx_service.build_deck(request.result, request.finance, request.chart_images)
        industry = request.result.get("industry", "report").replace(" ", "_")
        filename = f"DataMind_{industry}_{datetime.now().strftime('%Y%m%d')}.pptx"
        return StreamingResponse(io.BytesIO(pptx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    except Exception as e:
        logger.error(f"PPTX export error: {e}", exc_info=True)
        raise HTTPException(500, f"PowerPoint export failed: {str(e)}")
@router.post("/xlsx")
async def export_xlsx(request: ExportRequest):
    try:
        from app.services.document_service import document_service
        xlsx_bytes = document_service.build_excel_report(request.result, request.finance, request.chart_images,
            doc_title=request.title, doc_subtitle=request.subtitle, line_spacing=request.line_spacing)
        industry = request.result.get("industry", "report").replace(" ", "_")
        filename = f"DataMind_{industry}_{datetime.now().strftime('%Y%m%d')}.xlsx"
        return StreamingResponse(io.BytesIO(xlsx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    except Exception as e:
        logger.error(f"Excel export error: {e}", exc_info=True)
        raise HTTPException(500, f"Excel export failed: {str(e)}")
