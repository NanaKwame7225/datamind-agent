"""
DataMind Agent v2 — Main Application
Full SaaS platform with auth, history, rate limiting, payments, scheduled reports.
"""
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn, logging

from app.routers import analysis, pipeline, connectors, upload, export, finance
from app.routers import auth, history, schedules, payments
from app.services.auth_service import init_db, get_current_user_optional
from app.middleware.rate_limit import RateLimitMiddleware
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialise database on startup
init_db()

app = FastAPI(
    title="DataMind Agent API",
    description="""
## Universal AI Data Analysis Platform

Full SaaS platform with:
- **Authentication** — JWT login, registration, refresh tokens
- **Analysis** — AI-powered data analysis across 12 industries
- **Finance** — Tax, accounting, fraud detection modules
- **History** — Save and retrieve past analyses
- **Scheduled Reports** — Automated analysis delivery by email
- **Payments** — Stripe subscription management

Built by NkaySolutions · Accra, Ghana
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — restricted to your frontend ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiting ──────────────────────────────────────────────────────────────
app.add_middleware(RateLimitMiddleware)


# ── Inject user context into request state ────────────────────────────────────
@app.middleware("http")
async def inject_user_context(request: Request, call_next):
    try:
        from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
        from app.services.auth_service import verify_access_token, get_user_by_id
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = verify_access_token(token)
            user = get_user_by_id(payload["sub"])
            if user:
                request.state.user_id = user.id
                request.state.plan    = user.plan.value
    except Exception:
        pass
    return await call_next(request)


# ── v1 API Routers (existing — backward compatible) ────────────────────────────
app.include_router(analysis.router,   prefix="/api/v1/analysis",   tags=["Analysis"])
app.include_router(pipeline.router,   prefix="/api/v1/pipeline",   tags=["Pipeline"])
app.include_router(connectors.router, prefix="/api/v1/connectors", tags=["Connectors"])
app.include_router(upload.router,     prefix="/api/v1/upload",     tags=["Upload"])
app.include_router(export.router,     prefix="/api/v1/export",     tags=["Export"])
app.include_router(finance.router,    prefix="/api/v1/finance",    tags=["Finance"])

# ── v2 API Routers (new SaaS features) ────────────────────────────────────────
app.include_router(auth.router,      prefix="/api/v2/auth",      tags=["Auth"])
app.include_router(history.router,   prefix="/api/v2/history",   tags=["History"])
app.include_router(schedules.router, prefix="/api/v2/schedules", tags=["Scheduled Reports"])
app.include_router(payments.router,  prefix="/api/v2/payments",  tags=["Payments"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "DataMind Agent",
        "version": "2.0.0",
        "status": "online",
        "v1_endpoints": {
            "analysis":   "/api/v1/analysis",
            "finance":    "/api/v1/finance",
            "pipeline":   "/api/v1/pipeline",
            "upload":     "/api/v1/upload",
            "export":     "/api/v1/export",
        },
        "v2_endpoints": {
            "auth":       "/api/v2/auth",
            "history":    "/api/v2/history",
            "schedules":  "/api/v2/schedules",
            "payments":   "/api/v2/payments",
        },
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "version": "2.0.0"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
