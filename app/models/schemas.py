"""
DataMind Agent — Pydantic Schemas
All request/response models and enums for the API.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class LLMProvider(str, Enum):
    groq      = "groq"
    anthropic = "anthropic"
    openai    = "openai"
    gemini    = "gemini"
    cohere    = "cohere"
    mistral   = "mistral"
    auto      = "auto"


class Industry(str, Enum):
    finance       = "finance"
    education     = "education"
    supply_chain  = "supply_chain"
    procurement   = "procurement"
    healthcare    = "healthcare"
    mining        = "mining"
    petroleum     = "petroleum"
    retail        = "retail"
    agriculture   = "agriculture"
    manufacturing = "manufacturing"
    ngo           = "ngo"
    general       = "general"


# ── Sub-models ─────────────────────────────────────────────────────────────────

class Metric(BaseModel):
    label:      str
    value:      Any
    change_pct: Optional[float] = None
    trend:      Optional[str]   = None   # "up" | "down" | "flat"
    benchmark:  Optional[str]   = None


class Insight(BaseModel):
    title:      str
    body:       str
    severity:   str            = "info"  # "info" | "warning" | "critical" | "success"
    source:     Optional[str]  = None
    confidence: Optional[float] = None


class ChartData(BaseModel):
    chart_type: str
    title:      str
    data:       Any
    description: Optional[str] = None


class PipelineStep(BaseModel):
    name:           str
    tool:           str
    status:         str   = "done"   # "done" | "error" | "running"
    duration_ms:    float = 0.0
    output_preview: Optional[str] = None


# ── Request models ─────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    query:               str
    industry:            Industry            = Industry.general
    provider:            LLMProvider         = LLMProvider.groq
    model:               Optional[str]       = None
    inline_data:         Optional[list[dict]] = None
    enable_viz:          bool                = True
    enable_forecast:     bool                = False
    enable_anomaly_detection: bool           = True
    conversation_history: list[dict]         = Field(default_factory=list)


class ChatMessage(BaseModel):
    role:    str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system:   str = ""


# ── Response models ────────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    query:            str
    industry:         str
    provider:         str
    model:            Optional[str]         = None
    narrative:        Optional[str]         = None
    metrics:          list[Metric]          = Field(default_factory=list)
    insights:         list[Insight]         = Field(default_factory=list)
    charts:           list[ChartData]       = Field(default_factory=list)
    pipeline_steps:   list[PipelineStep]    = Field(default_factory=list)
    raw_data_preview: Optional[list[dict]]  = None
    execution_ms:     float                 = 0.0
    tokens_used:      int                   = 0


class ChatResponse(BaseModel):
    reply:    str
    provider: str = "unknown"


# ── Pipeline models ────────────────────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    steps: list[dict] = Field(default_factory=list)
    industry: Industry = Industry.general
    provider: LLMProvider = LLMProvider.groq
    inline_data: Optional[list[dict]] = None


class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    steps_completed: int = 0
    steps_total: int = 0
    duration_ms: float = 0.0
    outputs: dict = Field(default_factory=dict)


# ── Upload models ──────────────────────────────────────────────────────────────

class FileUploadResponse(BaseModel):
    file_id:      str
    filename:     str
    format:       str
    rows:         int
    columns:      int
    size_bytes:   int
    file_schema:  dict                    = Field(default_factory=dict)
    sample:       list[dict]              = Field(default_factory=list)
    quality_report: Optional[dict]        = None
