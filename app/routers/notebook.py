"""
DataMind Agent — Notebooks Router

GET    /api/v1/notebooks                          list my notebooks
POST   /api/v1/notebooks                          create {title, data, columns, industry, workspace_id}
GET    /api/v1/notebooks/{nid}                    get one (with cells)
PATCH  /api/v1/notebooks/{nid}                    rename {title}
DELETE /api/v1/notebooks/{nid}                    delete
POST   /api/v1/notebooks/{nid}/cells             add a cell {question}  → runs multi-agent
POST   /api/v1/notebooks/{nid}/cells/{cid}/rerun re-run a cell {question?}
DELETE /api/v1/notebooks/{nid}/cells/{cid}       delete a cell
POST   /api/v1/notebooks/{nid}/reorder           reorder {order:[cell_id,...]}

Workspace access is enforced the same way as history: personal = own only,
shared = must be a member, writes need owner/editor.
"""
import logging
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


def _svc():
    from app.services.notebook_service import notebook_service
    return notebook_service

async def _user(authorization):
    from app.routers.auth import current_user
    return await current_user(authorization)

async def _access(user_id, workspace_id, need_write=False):
    from app.routers.auth import _ws_access
    return await _ws_access(user_id, workspace_id, need_write)


class CreateReq(BaseModel):
    title: Optional[str] = "Untitled notebook"
    data: list
    columns: Optional[list] = None
    industry: Optional[str] = "general"
    workspace_id: Optional[str] = None

class RenameReq(BaseModel):
    title: str
    workspace_id: Optional[str] = None

class CellReq(BaseModel):
    question: str
    workspace_id: Optional[str] = None
    deep: Optional[bool] = False

class RerunReq(BaseModel):
    question: Optional[str] = None
    workspace_id: Optional[str] = None
    deep: Optional[bool] = False

class ReorderReq(BaseModel):
    order: list
    workspace_id: Optional[str] = None


@router.get("")
@router.get("/")
async def list_notebooks(workspace_id: Optional[str] = None, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        return {"success": True, "items": []}
    ok, err = await _access(user["_id"], workspace_id)
    if not ok:
        return {"success": False, "error": err, "items": []}
    return await _svc().list(user["_id"], workspace_id=workspace_id)

@router.post("")
@router.post("/")
async def create_notebook(req: CreateReq, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to create a notebook.")
    ok, err = await _access(user["_id"], req.workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().create(user["_id"], req.title, req.data, req.columns,
                               req.industry, workspace_id=req.workspace_id)

@router.get("/{nid}")
async def get_notebook(nid: str, workspace_id: Optional[str] = None, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], workspace_id)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().get(user["_id"], nid, workspace_id=workspace_id)

@router.patch("/{nid}")
async def rename_notebook(nid: str, req: RenameReq, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], req.workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().rename(user["_id"], nid, req.title, workspace_id=req.workspace_id)

@router.delete("/{nid}")
async def delete_notebook(nid: str, workspace_id: Optional[str] = None, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().delete(user["_id"], nid, workspace_id=workspace_id)

@router.post("/{nid}/cells")
async def add_cell(nid: str, req: CellReq, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], req.workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().add_cell(user["_id"], nid, req.question, workspace_id=req.workspace_id, deep=req.deep)

@router.post("/{nid}/cells/{cid}/rerun")
async def rerun_cell(nid: str, cid: str, req: RerunReq, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], req.workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().rerun_cell(user["_id"], nid, cid, req.question, workspace_id=req.workspace_id, deep=req.deep)

@router.delete("/{nid}/cells/{cid}")
async def delete_cell(nid: str, cid: str, workspace_id: Optional[str] = None, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().delete_cell(user["_id"], nid, cid, workspace_id=workspace_id)

@router.post("/{nid}/reorder")
async def reorder(nid: str, req: ReorderReq, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    ok, err = await _access(user["_id"], req.workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _svc().reorder_cells(user["_id"], nid, req.order, workspace_id=req.workspace_id)
