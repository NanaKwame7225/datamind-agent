import uuid, logging, io
from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from app.services.analysis_service import analysis_service
from app.models.schemas import FileUploadResponse

router = APIRouter()
logger = logging.getLogger(__name__)
SUPPORTED = {"csv":"csv","json":"json","parquet":"parquet","xlsx":"excel","xls":"excel"}
_store: dict = {}

@router.post("/file", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    fmt = SUPPORTED.get(ext)
    if not fmt:
        raise HTTPException(400, f"Unsupported: .{ext}")
    content = await file.read()
    try:
        if fmt == "csv":       df = pd.read_csv(io.BytesIO(content))
        elif fmt == "json":    df = pd.read_json(io.BytesIO(content))
        elif fmt == "parquet": df = pd.read_parquet(io.BytesIO(content))
        else:                  df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(422, str(e))
    fid = str(uuid.uuid4())[:12]
    _store[fid] = {"df": df, "fmt": fmt, "name": file.filename}
    return FileUploadResponse(
        file_id=fid, filename=file.filename, format=fmt,
        rows=len(df), columns=len(df.columns), size_bytes=len(content),
        file_schema=df.dtypes.astype(str).to_dict(),
        sample=analysis_service.df_to_records(df, 5),
        quality_report=analysis_service.quality_report(df))

@router.get("/file/{file_id}/preview")
async def preview(file_id: str, n: int = 20):
    if file_id not in _store:
        raise HTTPException(404, "File not found")
    df = _store[file_id]["df"]
    return {"rows": len(df), "columns": df.columns.tolist(),
            "sample": analysis_service.df_to_records(df, n)}
