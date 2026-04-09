import io, logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
from app.services.analysis_service import analysis_service
from app.routers.upload import _store as file_store

router = APIRouter()

@router.get("/download/{file_id}")
async def export(file_id: str, fmt: str = "csv"):
    if file_id not in file_store:
        raise HTTPException(404, "File not found")
    df = file_store[file_id]["df"]
    if fmt == "csv":
        return StreamingResponse(io.StringIO(df.to_csv(index=False)),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=export_{file_id}.csv"})
    elif fmt == "excel":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="Data")
        buf.seek(0)
        return StreamingResponse(buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=export_{file_id}.xlsx"})
    else:
        raise HTTPException(400, f"Unsupported: {fmt}")
