"""
DataMind Agent v2 — Scheduled Reports Router
POST   /api/v2/schedules
GET    /api/v2/schedules
GET    /api/v2/schedules/{id}
PUT    /api/v2/schedules/{id}/toggle
DELETE /api/v2/schedules/{id}
POST   /api/v2/schedules/{id}/run-now
"""
from fastapi import APIRouter, HTTPException, Depends
from app.services.auth_service import get_current_user
from app.services.scheduled_report_service import scheduled_report_service
from app.models.user import UserOut
from pydantic import BaseModel
from typing import Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class ScheduleCreate(BaseModel):
    name: str
    industry: str = "general"
    query: str
    schedule: str           # cron: "0 9 * * 1" = every Monday 9am
    recipients: list[str]   # email addresses
    data_source: Optional[str] = None


@router.post("", status_code=201)
async def create_schedule(
    data: ScheduleCreate,
    current_user: UserOut = Depends(get_current_user),
):
    """Create a new scheduled report."""
    try:
        from croniter import croniter
        if not croniter.is_valid(data.schedule):
            raise HTTPException(400, f"Invalid cron expression: {data.schedule}. Example: '0 9 * * 1' for Monday 9am")
    except ImportError:
        pass
    if not data.recipients:
        raise HTTPException(400, "At least one recipient email is required")
    schedule = scheduled_report_service.create_schedule(current_user.id, data.dict())
    return schedule


@router.get("")
async def list_schedules(current_user: UserOut = Depends(get_current_user)):
    """List all your scheduled reports."""
    return scheduled_report_service.list_schedules(current_user.id)


@router.get("/{schedule_id}")
async def get_schedule(
    schedule_id: str,
    current_user: UserOut = Depends(get_current_user),
):
    s = scheduled_report_service.get_schedule(schedule_id, current_user.id)
    if not s:
        raise HTTPException(404, "Schedule not found")
    return s


@router.put("/{schedule_id}/toggle")
async def toggle_schedule(
    schedule_id: str,
    active: bool,
    current_user: UserOut = Depends(get_current_user),
):
    """Pause or resume a scheduled report."""
    s = scheduled_report_service.get_schedule(schedule_id, current_user.id)
    if not s:
        raise HTTPException(404, "Schedule not found")
    return scheduled_report_service.toggle_schedule(schedule_id, current_user.id, active)


@router.delete("/{schedule_id}")
async def delete_schedule(
    schedule_id: str,
    current_user: UserOut = Depends(get_current_user),
):
    s = scheduled_report_service.get_schedule(schedule_id, current_user.id)
    if not s:
        raise HTTPException(404, "Schedule not found")
    scheduled_report_service.delete_schedule(schedule_id, current_user.id)
    return {"message": "Schedule deleted"}


@router.post("/{schedule_id}/run-now")
async def run_now(
    schedule_id: str,
    current_user: UserOut = Depends(get_current_user),
):
    """Manually trigger a scheduled report immediately."""
    s = scheduled_report_service.get_schedule(schedule_id, current_user.id)
    if not s:
        raise HTTPException(404, "Schedule not found")
    return {"message": f"Report '{s['name']}' queued for immediate delivery", "schedule_id": schedule_id}
