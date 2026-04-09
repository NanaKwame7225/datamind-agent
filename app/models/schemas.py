from pydantic import BaseModel, Field
from typing import Any, Optional, Literal
from enum import Enum

class LLMProvider(str, Enum):
    anthropic = "anthropic"
    openai = "openai"
    gemini = "gemini"
    cohere = "cohere"
    mistral = "mistral"

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
    telecom       = "telecom"
    manufacturing = "manufacturing"
    ngo           = "ngo"
    general       = "general"

class AnalysisRequest(BaseModel):
    query: str
    industry: Industry                       = Industry.general
    provider: LLMProvider                    = LLMProvider.anthropic
    model: Optional[str]                     = None
    inline_data: Optional[list[dict]]        = None
    file_id: Optional[str]                   = None
    db_connection_id: Optional[str]          = None
    enable_viz: bool                         = True
    enable_forecast: bool                    = False
    enable_anomaly_detection: bool           = False
    conversation_history: list[dict]         = Field(default_factory=list)

class Metric(BaseModel):
    label: str
    value: Any
    change: Optional[float]                  = None
    change_pct: Optional[float]              = None
    trend: Optional[Literal["up","down","flat"]] = None

class Insight(BaseModel):
    title: str
    body: str
    severity: Literal["info","warning","critical","success"] = "info"
    source: Optional[str]                    = None
    confidence: Optional[float]              = None

class ChartData(BaseModel):
    chart_type: str
    title: str
    data: dict
    description: Optional[str]              = None

class PipelineStep(BaseModel):
    name: str
    tool: str
    status: Literal["done","running","pending","error"]
    duration_ms: Optional[float]            = None
    output_preview: Optional[str]           = None

class AnalysisResponse(BaseModel):
    query: str
    industry: str
    provider: str
    model: str
    narrative: str
    metrics: list[Metric]                   = Field(default_factory=list)
    insights: list[Insight]                 = Field(default_factory=list)
    charts: list[ChartData]                 = Field(default_factory=list)
    pipeline_steps: list[PipelineStep]      = Field(default_factory=list)
    raw_data_preview: Optional[list[dict]]  = None
    execution_ms: Optional[float]           = None
    tokens_used: Optional[int]              = None

class PipelineRunRequest(BaseModel):
    name: str
    steps: list[dict]
    industry: Industry                      = Industry.general
    orchestrator: Literal["local","celery","ray"] = "local"

class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    steps_completed: int
    steps_total: int
    duration_ms: float
    outputs: dict

class DBConnectRequest(BaseModel):
    db_type: str
    connection_string: Optional[str]        = None
    database: Optional[str]                 = None
    options: dict                           = Field(default_factory=dict)

class DBConnectResponse(BaseModel):
    connection_id: str
    db_type: str
    status: str
    tables: list[str]

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    format: str
    rows: int
    columns: int
    size_bytes: int
    file_schema: dict
    sample: list[dict]
    quality_report: dict
