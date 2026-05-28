"""
DataMind Agent — Universal AI Data Analysis Backend
FastAPI application with full integration stack
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging

from app.routers import analysis, pipeline, connectors, upload, export, finance
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DataMind Agent API",
    description="Universal AI Data Analysis Platform — Finance, Education, Supply Chain, Mining, Petroleum & more",
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
app.include_router(finance.router,    prefix="/api/v1/finance",    tags=["Finance"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "DataMind Agent",
        "version": "2.4.0",
        "status": "online",
        "modules": {
            "analysis":   "/api/v1/analysis",
            "finance":    "/api/v1/finance",
            "pipeline":   "/api/v1/pipeline",
            "connectors": "/api/v1/connectors",
            "upload":     "/api/v1/upload",
            "export":     "/api/v1/export",
        },
        "finance_endpoints": {
            "tax":        "/api/v1/finance/tax",
            "accounting": "/api/v1/finance/accounting",
            "fraud":      "/api/v1/finance/fraud",
            "full":       "/api/v1/finance/full",
        },
        "integrations": {
            "llm":    ["anthropic", "openai", "gemini", "cohere", "mistral"],
            "data":   ["pandas", "polars", "numpy", "dask"],
            "ml":     ["scikit-learn", "xgboost", "lightgbm", "statsmodels"],
            "db":     ["postgresql", "mysql", "sqlite", "mongodb", "bigquery", "snowflake"],
            "viz":    ["plotly", "matplotlib", "seaborn", "bokeh"],
            "mlops":  ["mlflow", "wandb", "dvc"],
            "vector": ["pinecone", "weaviate", "chroma", "faiss"],
        },
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
