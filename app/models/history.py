"""
DataMind Agent v2 — Analysis History & Report Models
"""
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


class ReportStatus(str, Enum):
    pending   = "pending"
    running   = "running"
    completed = "completed"
    failed    = "failed"


class AnalysisHistoryItem(BaseModel):
    id: str
    user_id: str
    title: str
    query: str
    industry: str
    provider: str
    model: str
    row_count: int
    col_count: int
    execution_ms: float
    tokens_used: int
    status: ReportStatus
    created_at: datetime
    has_finance: bool = False
    fraud_risk_score: Optional[float] = None
    health_score: Optional[float] = None


class AnalysisDetail(AnalysisHistoryItem):
    narrative: Optional[str]
    metrics: list[dict]
    insights: list[dict]
    charts: list[dict]
    pipeline_steps: list[dict]
    raw_data_preview: Optional[list[dict]]
    finance_results: Optional[dict] = None


class ScheduledReport(BaseModel):
    id: str
    user_id: str
    name: str
    industry: str
    query: str
    schedule: str           # cron expression
    recipients: list[str]   # email addresses
    is_active: bool
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    created_at: datetime


class ScheduledReportCreate(BaseModel):
    name: str
    industry: str
    query: str
    schedule: str           # e.g. "0 9 * * 1" = every Monday 9am
    recipients: list[str]
    data_source: Optional[str] = None   # connection_id or None for sample


class UsageStats(BaseModel):
    user_id: str
    period: str             # "2024-01" etc
    analyses_run: int
    tokens_used: int
    api_calls: int
    plan: str
    limit: int
    pct_used: float
