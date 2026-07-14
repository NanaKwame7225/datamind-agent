"""
DataMind Agent — Auth & History Router

Auth:
  POST /api/v1/auth/register   {email, password, name?, guest_id?}
  POST /api/v1/auth/login      {email, password}
  POST /api/v1/auth/guest      -> throwaway session
  GET  /api/v1/auth/me         (Authorization: Bearer <token>)
  GET  /api/v1/auth/status     -> is auth/DB configured?

History (all require a token):
  POST   /api/v1/auth/history          save an analysis
  GET    /api/v1/auth/history          list my analyses
  GET    /api/v1/auth/history/{id}     get one
  PATCH  /api/v1/auth/history/{id}     rename
  DELETE /api/v1/auth/history/{id}     delete
"""
import logging
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


def _auth():
    from app.services.auth_service import auth_service
    return auth_service

def _history():
    from app.services.history_service import history_service
    return history_service

async def _ws_access(user_id: str, workspace_id: str, need_write: bool = False):
    """
    Returns (ok, error). Personal or no workspace → always ok (scoped to user).
    Shared workspace → must be a member; write ops need owner/editor.
    """
    if not workspace_id:
        return True, None
    from app.services.workspace_service import workspace_service, is_personal
    if is_personal(workspace_id):
        # A personal id must be the caller's own
        return (workspace_id == f"personal:{user_id}"), "Not your workspace."
    role = await workspace_service.role_of(user_id, workspace_id)
    if role is None:
        return False, "You don't have access to that workspace."
    if need_write and role not in ("owner", "editor"):
        return False, "You have view-only access to this workspace."
    return True, None


async def current_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Resolve the bearer token to a user, or None if absent/invalid."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    return await _auth().user_from_token(token)


async def require_user(authorization: Optional[str] = Header(None)) -> dict:
    user = await current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return user


# ── Schemas ───────────────────────────────────────────────────────────────────
class RegisterReq(BaseModel):
    email: str
    password: str
    name: Optional[str] = ""
    guest_id: Optional[str] = None

class LoginReq(BaseModel):
    email: str
    password: str

class SaveReq(BaseModel):
    query: Optional[str] = ""
    industry: Optional[str] = "general"
    result: Optional[dict] = None
    row_count: Optional[int] = None
    col_count: Optional[int] = None
    columns: Optional[list] = None
    data_preview: Optional[list] = None
    source: Optional[str] = None
    save_as: Optional[str] = "auto"          # auto | new | version
    target_id: Optional[str] = None
    workspace_id: Optional[str] = None

class RenameReq(BaseModel):
    title: str


# ── Auth endpoints ────────────────────────────────────────────────────────────
@router.post("/register")
async def register(req: RegisterReq):
    return await _auth().register(req.email, req.password, req.name or "", req.guest_id)

@router.post("/login")
async def login(req: LoginReq):
    return await _auth().login(req.email, req.password)

@router.post("/guest")
async def guest():
    return await _auth().guest()

@router.get("/me")
async def me(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        return {"success": False, "error": "No token provided."}
    return await _auth().me(authorization.split(" ", 1)[1].strip())

@router.get("/status")
async def status():
    from app.database import status as db_status
    ds = db_status()
    return {"auth_enabled": ds.get("configured", False),
            "database": ds, "guest_mode": True}


# ── History endpoints ─────────────────────────────────────────────────────────
@router.post("/history")
async def save_history(req: SaveReq, authorization: Optional[str] = Header(None)):
    user = await current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to save analyses.")
    payload = req.dict()
    save_as = payload.pop("save_as", "auto") or "auto"
    target_id = payload.pop("target_id", None)
    workspace_id = payload.pop("workspace_id", None)
    ok, err = await _ws_access(user["_id"], workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _history().save(user["_id"], payload, save_as=save_as, target_id=target_id, workspace_id=workspace_id)

@router.get("/history")
async def list_history(limit: int = 50, skip: int = 0, workspace_id: Optional[str] = None,
                       authorization: Optional[str] = Header(None)):
    user = await current_user(authorization)
    if not user:
        return {"success": True, "items": [], "total": 0}
    ok, err = await _ws_access(user["_id"], workspace_id)
    if not ok:
        return {"success": False, "error": err, "items": []}
    return await _history().list(user["_id"], limit, skip, workspace_id=workspace_id)

@router.get("/history/{analysis_id}")
async def get_history(analysis_id: str, version_id: Optional[str] = None,
                      workspace_id: Optional[str] = None,
                      authorization: Optional[str] = Header(None)):
    user = await current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to view saved analyses.")
    ok, err = await _ws_access(user["_id"], workspace_id)
    if not ok:
        return {"success": False, "error": err}
    return await _history().get(user["_id"], analysis_id, version_id, workspace_id=workspace_id)

@router.get("/history/{analysis_id}/versions")
async def list_versions(analysis_id: str, workspace_id: Optional[str] = None,
                        authorization: Optional[str] = Header(None)):
    user = await current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to view version history.")
    ok, err = await _ws_access(user["_id"], workspace_id)
    if not ok:
        return {"success": False, "error": err}
    return await _history().versions(user["_id"], analysis_id, workspace_id=workspace_id)

@router.post("/history/{analysis_id}/restore/{version_id}")
async def restore_version(analysis_id: str, version_id: str, workspace_id: Optional[str] = None,
                          authorization: Optional[str] = Header(None)):
    user = await current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to restore versions.")
    ok, err = await _ws_access(user["_id"], workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _history().restore_version(user["_id"], analysis_id, version_id, workspace_id=workspace_id)

@router.delete("/history/{analysis_id}/version/{version_id}")
async def delete_version(analysis_id: str, version_id: str, workspace_id: Optional[str] = None,
                         authorization: Optional[str] = Header(None)):
    user = await current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to edit versions.")
    ok, err = await _ws_access(user["_id"], workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _history().delete_version(user["_id"], analysis_id, version_id, workspace_id=workspace_id)

@router.patch("/history/{analysis_id}")
async def rename_history(analysis_id: str, req: RenameReq,
                         workspace_id: Optional[str] = None,
                         authorization: Optional[str] = Header(None)):
    user = await current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to edit saved analyses.")
    ok, err = await _ws_access(user["_id"], workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _history().rename(user["_id"], analysis_id, req.title, workspace_id=workspace_id)

@router.delete("/history/{analysis_id}")
async def delete_history(analysis_id: str, workspace_id: Optional[str] = None,
                         authorization: Optional[str] = Header(None)):
    user = await current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to delete saved analyses.")
    ok, err = await _ws_access(user["_id"], workspace_id, need_write=True)
    if not ok:
        return {"success": False, "error": err}
    return await _history().delete(user["_id"], analysis_id, workspace_id=workspace_id)
