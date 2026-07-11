"""
DataMind Agent — Scheduled Reports Router

POST   /api/v1/schedules            create a schedule
GET    /api/v1/schedules            list mine
POST   /api/v1/schedules/{id}/run   run now
PATCH  /api/v1/schedules/{id}       toggle active {active: bool}
DELETE /api/v1/schedules/{id}       delete
GET    /api/v1/schedules/channels   which delivery channels are configured
"""
import logging
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


def _svc():
    from app.services.schedule_service import schedule_service
    return schedule_service

async def _uid(authorization):
    from app.routers.auth import current_user
    user = await current_user(authorization)
    return user["_id"] if user else None


class CreateReq(BaseModel):
    name: Optional[str] = ""
    query: str
    industry: Optional[str] = "general"
    data: list
    columns: Optional[list] = None
    frequency: Optional[str] = "weekly"       # daily | weekly | monthly | hourly
    hour: Optional[int] = 8
    minute: Optional[int] = 0
    weekday: Optional[str] = "monday"
    day: Optional[int] = 1
    channels: Optional[dict] = None           # {email, email_to, sms, sms_to}

class ToggleReq(BaseModel):
    active: bool


@router.post("")
@router.post("/")
async def create_schedule(req: CreateReq, authorization: Optional[str] = Header(None)):
    uid = await _uid(authorization)
    if not uid:
        raise HTTPException(status_code=401, detail="Sign in to schedule reports.")
    return await _svc().create(uid, req.dict())

@router.get("")
@router.get("/")
async def list_schedules(authorization: Optional[str] = Header(None)):
    uid = await _uid(authorization)
    if not uid:
        return {"success": True, "items": []}
    return await _svc().list(uid)

@router.post("/{sid}/run")
async def run_now(sid: str, authorization: Optional[str] = Header(None)):
    uid = await _uid(authorization)
    if not uid:
        raise HTTPException(status_code=401, detail="Sign in.")
    return await _svc().run_now(uid, sid)

@router.patch("/{sid}")
async def toggle_schedule(sid: str, req: ToggleReq, authorization: Optional[str] = Header(None)):
    uid = await _uid(authorization)
    if not uid:
        raise HTTPException(status_code=401, detail="Sign in.")
    return await _svc().toggle(uid, sid, req.active)

@router.delete("/{sid}")
async def delete_schedule(sid: str, authorization: Optional[str] = Header(None)):
    uid = await _uid(authorization)
    if not uid:
        raise HTTPException(status_code=401, detail="Sign in.")
    return await _svc().delete(uid, sid)

@router.get("/channels")
async def channels():
    from app.services.notify_service import notify_service
    return notify_service.status()
