from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging

from app.routers import analysis, pipeline, connectors, upload, export
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DataMind Agent API",
    description="Universal AI Data Analysis Platform",
    version="2.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router,   prefix="/api/v1/analysis",   tags=["Analysis"])
app.include_router(pipeline.router,   prefix="/api/v1/pipeline",   tags=["Pipeline"])
app.include_router(connectors.router, prefix="/api/v1/connectors", tags=["Connectors"])
app.include_router(upload.router,     prefix="/api/v1/upload",     tags=["Upload"])
app.include_router(export.router,     prefix="/api/v1/export",     tags=["Export"])

@app.get("/", tags=["Health"])
async def root():
    return {"service": "DataMind Agent", "version": "2.4.0", "status": "online"}

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})
