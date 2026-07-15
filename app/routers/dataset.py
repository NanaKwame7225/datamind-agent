"""
DataMind Agent — Datasets Router

Server-side dataset storage so large data never round-trips through the browser.

POST   /api/v1/datasets/upload            store a parsed file → {dataset_id, preview}
GET    /api/v1/datasets                   list my datasets
GET    /api/v1/datasets/{did}/preview     first N rows (default 50)
GET    /api/v1/datasets/{did}/stats       column stats computed server-side
GET    /api/v1/datasets/{did}/sample      evenly-spread sample (for AI/charts)
POST   /api/v1/datasets/{did}/aggregate   group_by + metric → tiny chart series
DELETE /api/v1/datasets/{did}             delete
"""
import logging
from fastapi import APIRouter, Header, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


def _svc():
    from app.services.dataset_service import dataset_service
    return dataset_service

async def _user(authorization):
    from app.routers.auth import current_user
    return await current_user(authorization)

async def _access(user_id, workspace_id, need_write=False):
    from app.routers.auth import _ws_access
    return await _ws_access(user_id, workspace_id, need_write)


class StoreReq(BaseModel):
    records: list
    filename: Optional[str] = None
    workspace_id: Optional[str] = None

class AggReq(BaseModel):
    group_by: str
    metric: Optional[str] = None
    agg: Optional[str] = "sum"
    limit: Optional[int] = 20
    workspace_id: Optional[str] = None


@router.post("/store")
async def store(req: StoreReq, authorization: Optional[str] = Header(None)):
    """Store already-parsed records (the frontend parses, then hands off)."""
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to store datasets.")
    ok, err = await _access(user["_id"], req.workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().store(user["_id"], req.records, req.filename,
                              workspace_id=req.workspace_id)


@router.post("/upload")
async def upload(file: UploadFile = File(...),
                 workspace_id: Optional[str] = None,
                 authorization: Optional[str] = Header(None)):
    """Parse a file on the server and store it — the browser never holds the rows."""
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to upload datasets.")
    ok, err = await _access(user["_id"], workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    try:
        from app.services.file_parser_service import file_parser_service
        content = await file.read()
        parsed = file_parser_service.parse(content, file.filename)
        if not parsed.get("success"):
            return {"success": False, "error": parsed.get("error", "Could not parse file.")}
        records = parsed.get("records") or parsed.get("data") or []
        return await _svc().store(user["_id"], records, file.filename,
                                  workspace_id=workspace_id)
    except Exception as e:
        logger.error(f"Dataset upload failed: {e}", exc_info=True)
        return {"success": False, "error": f"Upload failed: {e}"}


@router.get("")
@router.get("/")
async def list_datasets(workspace_id: Optional[str] = None,
                        authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        return {"success": True, "items": []}
    ok, err = await _access(user["_id"], workspace_id)
    if not ok:
        return {"success": False, "error": err, "items": []}
    return await _svc().list(user["_id"], workspace_id=workspace_id)


@router.get("/{did}/preview")
async def preview(did: str, limit: int = 50, workspace_id: Optional[str] = None,
                  authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], workspace_id)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().preview(user["_id"], did, limit, workspace_id=workspace_id)


@router.get("/{did}/stats")
async def stats(did: str, workspace_id: Optional[str] = None,
                authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], workspace_id)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().stats(user["_id"], did, workspace_id=workspace_id)


@router.get("/{did}/sample")
async def sample(did: str, n: int = 4000, workspace_id: Optional[str] = None,
                 authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], workspace_id)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().sample(user["_id"], did, n, workspace_id=workspace_id)


@router.post("/{did}/aggregate")
async def aggregate(did: str, req: AggReq, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], req.workspace_id)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().aggregate(user["_id"], did, req.group_by, req.metric,
                                  req.agg, req.limit, workspace_id=req.workspace_id)


@router.delete("/{did}")
async def delete(did: str, workspace_id: Optional[str] = None,
                 authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().delete(user["_id"], did, workspace_id=workspace_id)
