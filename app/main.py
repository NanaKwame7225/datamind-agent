from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
import uuid

from app.routers import analysis, pipeline, connectors, upload, export
from config.settings import settings

# =========================
# LOGGING CONFIG
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(request_id)s] %(message)s"
)
logger = logging.getLogger("DataMind")


# =========================
# APP INIT
# =========================
app = FastAPI(
    title="DataMind Agent API",
    description="Enterprise AI Data Intelligence Platform",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# =========================
# CORS (PRODUCTION SAFE)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, "ALLOWED_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# REQUEST TIMING + TRACE ID MIDDLEWARE
# =========================
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # attach request id
    request.state.request_id = request_id

    response = await call_next(request)

    process_time = time.time() - start_time

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)

    logger.info(
        "Request completed",
        extra={"request_id": request_id}
    )

    return response


# =========================
# ROUTERS
# =========================
app.include_router(analysis.router,   prefix="/api/v1/analysis",   tags=["Analysis"])
app.include_router(pipeline.router,   prefix="/api/v1/pipeline",   tags=["Pipeline"])
app.include_router(connectors.router, prefix="/api/v1/connectors", tags=["Connectors"])
app.include_router(upload.router,     prefix="/api/v1/upload",     tags=["Upload"])
app.include_router(export.router,     prefix="/api/v1/export",     tags=["Export"])


# =========================
# LIFECYCLE EVENTS
# =========================
@app.on_event("startup")
async def startup_event():
    logger.info("DataMind Agent starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("DataMind Agent shutting down...")


# =========================
# HEALTH CHECKS
# =========================
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "DataMind Agent",
        "version": app.version,
        "status": "online"
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


@app.get("/ready", tags=["Health"])
async def readiness():
    return {
        "status": "ready",
        "services": {
            "api": "up"
        }
    }


# =========================
# GLOBAL EXCEPTION HANDLER (SAFE FOR PRODUCTION)
# =========================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):

    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"Unhandled error: {str(exc)}",
        extra={"request_id": request_id},
        exc_info=True
    )

    # IMPORTANT: avoid leaking internal errors in production
    if getattr(settings, "ENV", "dev") == "prod":
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id
            }
        )

    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "request_id": request_id
        }
    )
