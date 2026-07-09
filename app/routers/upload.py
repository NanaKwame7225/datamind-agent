"""
DataMind Agent — Upload & File Parsing Router
POST /api/v1/upload/inspect      — peek at one file (Excel sheets, PDF tables, preview)
POST /api/v1/upload/parse        — parse one file into records
POST /api/v1/upload/parse-multi  — parse MANY files at once, optionally merged
POST /api/v1/upload/sheet        — parse a specific Excel sheet
GET  /api/v1/upload/formats      — list supported formats
GET  /api/v1/upload/health       — check which parsing libraries are available
"""
import logging, traceback
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, List

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_MB = 100
MAX_FILES = 10


def _err(msg: str, exc: Exception = None, **extra):
    """Return a structured error at HTTP 200 so the frontend can show a real message."""
    if exc:
        logger.error(f"{msg}: {exc}\n{traceback.format_exc()}")
    return {"success": False, "error": msg, **extra}


def _get_parser():
    from app.services.file_parser_service import file_parser_service
    return file_parser_service


@router.get("/health")
async def parse_health():
    """Report which optional parsing libraries actually installed."""
    status = {}
    for lib, purpose in [
        ("pandas", "core"),
        ("openpyxl", "xlsx/xlsm"),
        ("xlrd", "legacy xls"),
        ("pdfplumber", "pdf tables"),
        ("pyarrow", "parquet"),
    ]:
        try:
            __import__(lib)
            status[lib] = {"installed": True, "purpose": purpose}
        except Exception as e:
            status[lib] = {"installed": False, "purpose": purpose, "error": str(e)}
    try:
        _get_parser()
        status["file_parser_service"] = {"installed": True, "purpose": "parser"}
    except Exception as e:
        status["file_parser_service"] = {"installed": False, "error": str(e)}
    all_ok = all(v.get("installed") for v in status.values())
    return {"success": True, "all_ready": all_ok, "libraries": status}


@router.post("/inspect")
async def inspect_file(file: UploadFile = File(...)):
    """Peek at an uploaded file without a full parse."""
    try:
        content = await file.read()
    except Exception as e:
        return _err("Could not read the uploaded file stream.", e)

    if len(content) > MAX_MB * 1024 * 1024:
        return _err(f"File exceeds the {MAX_MB}MB limit.", filename=file.filename)
    if not content:
        return _err("The uploaded file is empty.", filename=file.filename)

    try:
        parser = _get_parser()
    except Exception as e:
        return _err("File parser service is unavailable on the server.", e)

    try:
        result = parser.inspect(content, file.filename)
        result["filename"] = file.filename
        result["size_bytes"] = len(content)
        return result
    except ImportError as e:
        return _err(f"A required parsing library is missing on the server: {e}", e,
                    hint="Ensure pdfplumber, openpyxl and xlrd are in requirements.txt.")
    except Exception as e:
        return _err(f"Could not inspect this file: {e}", e, filename=file.filename)


@router.post("/parse")
async def parse_file(file: UploadFile = File(...), sheet_name: Optional[str] = Form(None)):
    """Parse a single file into clean records."""
    try:
        content = await file.read()
    except Exception as e:
        return _err("Could not read the uploaded file stream.", e)

    if len(content) > MAX_MB * 1024 * 1024:
        return _err(f"File exceeds the {MAX_MB}MB limit.", filename=file.filename)
    if not content:
        return _err("The uploaded file is empty.", filename=file.filename)

    try:
        parser = _get_parser()
    except Exception as e:
        return _err("File parser service is unavailable on the server.", e)

    try:
        result = parser.parse(content, file.filename, sheet_name)
        result["filename"] = file.filename
        result["size_bytes"] = len(content)
        return result
    except ImportError as e:
        return _err(f"A required parsing library is missing on the server: {e}", e,
                    hint="Ensure pdfplumber, openpyxl and xlrd are in requirements.txt.")
    except Exception as e:
        return _err(f"Could not parse this file: {e}", e, filename=file.filename)


@router.post("/parse-multi")
async def parse_multiple(
    files: List[UploadFile] = File(...),
    merge: bool = Form(False),
    merge_strategy: str = Form("stack"),   # "stack" (rows) or "join" (columns on shared key)
    join_key: Optional[str] = Form(None),
):
    """
    Parse several files at once.

    merge=False -> returns each dataset separately (user picks which to analyse)
    merge=True  -> combines them:
        stack: append rows (files must share most columns)
        join:  merge on a shared key column
    """
    if len(files) > MAX_FILES:
        return _err(f"Too many files. Maximum is {MAX_FILES} per upload.")

    try:
        parser = _get_parser()
    except Exception as e:
        return _err("File parser service is unavailable on the server.", e)

    datasets, failures = [], []

    for f in files:
        try:
            content = await f.read()
        except Exception as e:
            failures.append({"filename": f.filename, "error": "Could not read file stream"})
            continue

        if not content:
            failures.append({"filename": f.filename, "error": "File is empty"})
            continue
        if len(content) > MAX_MB * 1024 * 1024:
            failures.append({"filename": f.filename, "error": f"Exceeds {MAX_MB}MB limit"})
            continue

        try:
            res = parser.parse(content, f.filename)
            if res.get("success"):
                res["filename"] = f.filename
                res["size_bytes"] = len(content)
                datasets.append(res)
            else:
                failures.append({"filename": f.filename, "error": res.get("error", "Parse failed"),
                                 "hint": res.get("hint")})
        except Exception as e:
            logger.error(f"Multi-parse failed on {f.filename}: {e}\n{traceback.format_exc()}")
            failures.append({"filename": f.filename, "error": str(e)})

    if not datasets:
        return {"success": False, "error": "None of the uploaded files could be parsed.",
                "failures": failures, "datasets": []}

    response = {
        "success": True,
        "file_count": len(datasets),
        "failed_count": len(failures),
        "failures": failures,
        "datasets": [
            {k: v for k, v in d.items() if k != "records"} | {"preview": d["records"][:5]}
            for d in datasets
        ],
    }

    if not merge:
        # Return full records for each file so the user can pick
        response["records_by_file"] = {d["filename"]: d["records"] for d in datasets}
        return response

    # ── MERGE ──
    try:
        merged = parser.merge_datasets(datasets, strategy=merge_strategy, join_key=join_key)
        response["merged"] = merged
        return response
    except Exception as e:
        logger.error(f"Merge failed: {e}\n{traceback.format_exc()}")
        response["merge_error"] = str(e)
        response["records_by_file"] = {d["filename"]: d["records"] for d in datasets}
        return response


@router.post("/sheet")
async def parse_sheet(file: UploadFile = File(...), sheet_name: str = Form(...)):
    """Parse a specific sheet from an Excel workbook."""
    try:
        content = await file.read()
        parser = _get_parser()
        result = parser.parse(content, file.filename, sheet_name)
        result["filename"] = file.filename
        return result
    except Exception as e:
        return _err(f"Could not parse sheet '{sheet_name}': {e}", e)


@router.get("/formats")
async def supported_formats():
    return {
        "formats": [
            {"ext": ".csv",     "name": "CSV",         "notes": "Auto-detects delimiter"},
            {"ext": ".tsv",     "name": "TSV",         "notes": "Tab separated"},
            {"ext": ".xlsx",    "name": "Excel",       "notes": "Multi-sheet; auto-picks the largest sheet"},
            {"ext": ".xls",     "name": "Excel 97",    "notes": "Legacy format"},
            {"ext": ".xlsm",    "name": "Excel Macro", "notes": "Macros ignored"},
            {"ext": ".json",    "name": "JSON",        "notes": "Finds nested record arrays"},
            {"ext": ".parquet", "name": "Parquet",     "notes": "Columnar format"},
            {"ext": ".pdf",     "name": "PDF",         "notes": "Extracts tables; falls back to text"},
        ],
        "max_size_mb": MAX_MB,
        "max_files_per_upload": MAX_FILES,
        "merge_strategies": [
            {"key": "stack", "name": "Stack rows",  "notes": "Append files with the same columns"},
            {"key": "join",  "name": "Join columns","notes": "Merge on a shared key column"},
        ],
    }
