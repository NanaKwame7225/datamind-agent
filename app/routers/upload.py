"""
DataMind Agent — Upload & File Parsing Router
POST /api/v1/upload/parse    — parse any file into records (Excel, CSV, JSON, Parquet, PDF)
POST /api/v1/upload/inspect  — peek at a file: Excel sheets, PDF tables, preview rows
POST /api/v1/upload/sheet    — parse a specific Excel sheet
"""
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_MB = 100


@router.post("/inspect")
async def inspect_file(file: UploadFile = File(...)):
    """
    Peek at an uploaded file without committing to a full parse.
    Excel -> list of sheets with row counts and previews.
    PDF   -> number of tables detected per page.
    Other -> preview rows.
    """
    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_MB}MB limit")
    try:
        from app.services.file_parser_service import file_parser_service
        result = file_parser_service.inspect(content, file.filename)
        result["filename"] = file.filename
        result["size_bytes"] = len(content)
        return result
    except Exception as e:
        logger.error(f"Inspect failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@router.post("/parse")
async def parse_file(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None),
):
    """
    Parse any supported file into clean records ready for analysis.
    Optionally specify an Excel sheet_name.
    """
    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_MB}MB limit")
    try:
        from app.services.file_parser_service import file_parser_service
        result = file_parser_service.parse(content, file.filename, sheet_name)
        if not result.get("success"):
            # Return 200 with the error so the frontend can show the hint gracefully
            result["filename"] = file.filename
            return result
        result["filename"] = file.filename
        result["size_bytes"] = len(content)
        return result
    except Exception as e:
        logger.error(f"Parse failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@router.post("/sheet")
async def parse_sheet(
    file: UploadFile = File(...),
    sheet_name: str = Form(...),
):
    """Parse a specific sheet from an Excel workbook."""
    content = await file.read()
    try:
        from app.services.file_parser_service import file_parser_service
        result = file_parser_service.parse(content, file.filename, sheet_name)
        result["filename"] = file.filename
        return result
    except Exception as e:
        logger.error(f"Sheet parse failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@router.get("/formats")
async def supported_formats():
    """List supported upload formats."""
    return {
        "formats": [
            {"ext": ".csv",     "name": "CSV",        "notes": "Auto-detects delimiter"},
            {"ext": ".tsv",     "name": "TSV",        "notes": "Tab separated"},
            {"ext": ".xlsx",    "name": "Excel",      "notes": "Multi-sheet; auto-picks the largest sheet"},
            {"ext": ".xls",     "name": "Excel 97",   "notes": "Legacy format"},
            {"ext": ".xlsm",    "name": "Excel Macro","notes": "Macros are ignored"},
            {"ext": ".json",    "name": "JSON",       "notes": "Finds nested record arrays"},
            {"ext": ".parquet", "name": "Parquet",    "notes": "Columnar format"},
            {"ext": ".pdf",     "name": "PDF",        "notes": "Extracts tables; falls back to text"},
        ],
        "max_size_mb": MAX_MB,
    }
