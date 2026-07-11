"""
DataMind Agent — URL Ingestion Router

POST /api/v1/url/inspect  — peek at a remote file (Excel sheets, PDF tables, preview)
POST /api/v1/url/parse    — fetch and parse a remote file into records
GET  /api/v1/url/sources  — which link types are supported
"""
import logging, traceback
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


class URLRequest(BaseModel):
    url: str
    sheet_name: Optional[str] = None


def _svc():
    from app.services.url_ingest_service import url_ingest_service
    return url_ingest_service


@router.post("/inspect")
async def inspect_url(req: URLRequest):
    """Peek at a remote file without committing to a full parse."""
    try:
        return _svc().inspect(req.url)
    except Exception as e:
        logger.error(f"URL inspect failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": f"Could not inspect that link: {e}"}


@router.post("/parse")
async def parse_url(req: URLRequest):
    """Fetch a remote file and parse it into clean records."""
    try:
        return _svc().ingest(req.url, req.sheet_name)
    except Exception as e:
        logger.error(f"URL parse failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": f"Could not load that link: {e}"}


@router.get("/sources")
async def supported_sources():
    """Link types DataMind understands, and what each one needs."""
    return {
        "sources": [
            {"name": "Google Sheets", "example": "https://docs.google.com/spreadsheets/d/.../edit",
             "requires": "Share as 'Anyone with the link can view'",
             "note": "Reads the tab in the link (#gid=), or the first tab."},
            {"name": "Google Drive", "example": "https://drive.google.com/file/d/.../view",
             "requires": "Share as 'Anyone with the link can view'",
             "note": "Any supported file type."},
            {"name": "Dropbox", "example": "https://www.dropbox.com/s/.../data.csv?dl=0",
             "requires": "A share link", "note": "The ?dl=0 preview link works."},
            {"name": "OneDrive", "example": "https://1drv.ms/x/s!...",
             "requires": "Anonymous view access", "note": None},
            {"name": "GitHub", "example": "https://github.com/user/repo/blob/main/data.csv",
             "requires": "Public repository", "note": "The blob page link works."},
            {"name": "Direct link", "example": "https://bucket.s3.amazonaws.com/data.csv",
             "requires": "Publicly readable", "note": "S3, any static host, any raw file URL."},
        ],
        "formats": ["csv", "tsv", "xlsx", "xlsm", "xls", "json", "parquet", "pdf"],
        "max_size_mb": 100,
        "auth": "None. Files must be publicly readable via their link.",
    }
