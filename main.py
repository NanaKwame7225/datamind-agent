"""
DataMind Agent — Main Application v2
Full stack: Analysis + Finance + SQL + Memory + RAG
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DataMind Agent API",
    description="""
## Universal Elite AI Data Analysis Platform

Full stack with:
- **Analysis** — Elite statistical analysis across 12 industries
- **Finance** — Tax, accounting, fraud detection
- **SQL** — Live SQL execution against your data (DuckDB)
- **Memory** — Cross-query context and conversation history
- **RAG** — Industry benchmarks and domain knowledge retrieval
- **Visualisation** — Plotly charts auto-generated from data

Built by NkaySolutions · Accra, Ghana
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialise memory DB ──────────────────────────────────────────────────────
try:
    from app.services.memory_service import init_memory_db, memory_service
    init_memory_db()
    logger.info("Memory layer initialised")
except Exception as e:
    logger.warning(f"Memory layer init failed: {e}")

# ── Core routers ──────────────────────────────────────────────────────────────
from app.routers import analysis, pipeline, connectors, upload, export

app.include_router(analysis.router,   prefix="/api/v1/analysis",   tags=["Analysis"])
app.include_router(pipeline.router,   prefix="/api/v1/pipeline",   tags=["Pipeline"])
app.include_router(connectors.router, prefix="/api/v1/connectors", tags=["Connectors"])
app.include_router(upload.router,     prefix="/api/v1/upload",     tags=["Upload"])
app.include_router(export.router,     prefix="/api/v1/export",     tags=["Export"])

# ── Finance router ────────────────────────────────────────────────────────────
try:
    from app.routers import finance
    app.include_router(finance.router, prefix="/api/v1/finance", tags=["Finance"])
    logger.info("Finance router loaded")
except Exception as e:
    logger.error(f"Finance router failed: {e}", exc_info=True)

# ── SQL router ────────────────────────────────────────────────────────────────
try:
    from app.routers import sql as sql_router
    app.include_router(sql_router.router, prefix="/api/v1/sql", tags=["SQL Engine"])
    logger.info("SQL router loaded")
except Exception as e:
    logger.error(f"SQL router failed: {e}", exc_info=True)

# ── Memory router ─────────────────────────────────────────────────────────────
try:
    from app.routers import memory as memory_router
    app.include_router(memory_router.router, prefix="/api/v1/memory", tags=["Memory"])
    logger.info("Memory router loaded")
except Exception as e:
    logger.error(f"Memory router failed: {e}", exc_info=True)

# ── Forecast / Predictive Analytics router ────────────────────────────────────
try:
    from app.routers import forecast as forecast_router
    app.include_router(forecast_router.router, prefix="/api/v1/forecast", tags=["Predictive Analytics"])
    logger.info("Forecast router loaded")
except Exception as e:
    logger.error(f"Forecast router failed: {e}", exc_info=True)

# ── URL ingestion router (Google Sheets, Dropbox, S3, GitHub…) ────────────────
try:
    from app.routers import url as url_router
    app.include_router(url_router.router, prefix="/api/v1/url", tags=["URL Ingestion"])
    logger.info("URL ingestion router loaded")
except Exception as e:
    logger.error(f"URL router failed: {e}", exc_info=True)

# ── Voice transcription router (all-device voice input) ───────────────────────
try:
    from app.routers import transcribe as transcribe_router
    app.include_router(transcribe_router.router, prefix="/api/v1/transcribe", tags=["Voice"])
    logger.info("Transcription router loaded")
except Exception as e:
    logger.error(f"Transcription router failed: {e}", exc_info=True)

# ── Auth + saved history router (accounts, login, guest, history) ─────────────
try:
    from app.routers import auth as auth_router
    app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["Auth"])
    logger.info("Auth router loaded")
except Exception as e:
    logger.error(f"Auth router failed: {e}", exc_info=True)

# ── Scheduled reports router ──────────────────────────────────────────────────
try:
    from app.routers import schedule as schedule_router
    app.include_router(schedule_router.router, prefix="/api/v1/schedules", tags=["Scheduled Reports"])
    logger.info("Schedules router loaded")
except Exception as e:
    logger.error(f"Schedules router failed: {e}", exc_info=True)

# ── Workspaces router (shared spaces, roles, invites) ─────────────────────────
try:
    from app.routers import workspace as workspace_router
    app.include_router(workspace_router.router, prefix="/api/v1/workspaces", tags=["Workspaces"])
    logger.info("Workspaces router loaded")
except Exception as e:
    logger.error(f"Workspaces router failed: {e}", exc_info=True)

# ── Datasets router (server-side storage for large data) ──────────────────────
try:
    from app.routers import dataset as dataset_router
    app.include_router(dataset_router.router, prefix="/api/v1/datasets", tags=["Datasets"])
    logger.info("Datasets router loaded")
except Exception as e:
    logger.error(f"Datasets router failed: {e}", exc_info=True)

# ── Notebooks router (multi-agent analysis notebooks) ─────────────────────────
try:
    from app.routers import notebook as notebook_router
    app.include_router(notebook_router.router, prefix="/api/v1/notebooks", tags=["Notebooks"])
    logger.info("Notebooks router loaded")
except Exception as e:
    logger.error(f"Notebooks router failed: {e}", exc_info=True)

# ── Database warm-up (auth + history) ─────────────────────────────────────────
@app.on_event("startup")
async def _startup_db():
    try:
        from app.database import connect, status as db_status
        await connect()
        logger.info(f"Database status: {db_status()}")
    except Exception as e:
        logger.error(f"Database warm-up failed: {e}", exc_info=True)
    # Start the background scheduler for reports
    try:
        from app.scheduler_loop import start as start_scheduler
        start_scheduler()
    except Exception as e:
        logger.error(f"Scheduler start failed: {e}", exc_info=True)



@app.get("/")
async def root():
    return {
        "service": "DataMind Agent",
        "version": "2.0.0",
        "status": "online",
        "stack": {
            "llm": "Claude Sonnet 4 → GPT-4o → Gemini 2.0 Flash → Groq → Command R+",
            "analytics": "Pandas + NumPy + SciPy + Scikit-learn",
            "forecasting": "Holt-Winters + OLS with 95% prediction intervals",
            "sql": "DuckDB in-process SQL engine",
            "memory": "SQLite persistent conversation memory",
            "rag": "Industry benchmarks + domain knowledge retrieval",
            "parsing": "Excel, PDF, CSV, JSON, Parquet — files, multi-file, or a link",
            "ingestion": "Google Sheets, Drive, Dropbox, OneDrive, GitHub, S3 — no OAuth",
            "viz": "Plotly interactive charts",
            "export": "Word, PDF, PowerPoint",
        },
        "endpoints": {
            "analysis":   "/api/v1/analysis/analyse",
            "finance":    "/api/v1/finance/full",
            "sql":        "/api/v1/sql/query",
            "sql_nl":     "/api/v1/sql/natural",
            "memory":     "/api/v1/memory/context",
            "forecast":   "/api/v1/forecast/predict",
            "scenario":   "/api/v1/forecast/scenario",
            "upload":     "/api/v1/upload/parse",
            "upload_multi": "/api/v1/upload/parse-multi",
            "url_parse":  "/api/v1/url/parse",
            "url_sources":"/api/v1/url/sources",
            "export_pptx":  "/api/v1/export/pptx",
        },
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"},
    )
