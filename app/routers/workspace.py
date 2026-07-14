"""
DataMind Agent — Workspaces Router

GET    /api/v1/workspaces                     list my workspaces (personal + shared)
POST   /api/v1/workspaces                     create a shared workspace {name}
GET    /api/v1/workspaces/{wid}/members       list members
POST   /api/v1/workspaces/{wid}/invite        invite {email, role}
PATCH  /api/v1/workspaces/{wid}/member/{uid}  change role {role}
DELETE /api/v1/workspaces/{wid}/member/{uid}  remove a member
PATCH  /api/v1/workspaces/{wid}               rename {name}
DELETE /api/v1/workspaces/{wid}               delete the workspace
"""
import logging
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


def _svc():
    from app.services.workspace_service import workspace_service
    return workspace_service

async def _user(authorization):
    from app.routers.auth import current_user
    return await current_user(authorization)


class CreateReq(BaseModel):
    name: str

class InviteReq(BaseModel):
    email: str
    role: str = "viewer"

class RoleReq(BaseModel):
    role: str

class RenameReq(BaseModel):
    name: str


@router.get("")
@router.get("/")
async def list_workspaces(authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        return {"success": True, "workspaces": []}
    return await _svc().list_for_user(user["_id"], user.get("email"))

@router.post("")
@router.post("/")
async def create_workspace(req: CreateReq, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to create a workspace.")
    return await _svc().create(user["_id"], req.name)

@router.get("/{wid}/members")
async def members(wid: str, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    return await _svc().members(user["_id"], wid)

@router.post("/{wid}/invite")
async def invite(wid: str, req: InviteReq, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    return await _svc().invite(user["_id"], wid, req.email, req.role)

@router.patch("/{wid}/member/{uid}")
async def set_role(wid: str, uid: str, req: RoleReq, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    return await _svc().set_role(user["_id"], wid, uid, req.role)

@router.delete("/{wid}/member/{uid}")
async def remove_member(wid: str, uid: str, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    return await _svc().remove_member(user["_id"], wid, uid)

@router.patch("/{wid}")
async def rename(wid: str, req: RenameReq, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    return await _svc().rename(user["_id"], wid, req.name)

@router.delete("/{wid}")
async def delete(wid: str, authorization: Optional[str] = Header(None)):
    user = await _user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in.")
    return await _svc().delete(user["_id"], wid)
