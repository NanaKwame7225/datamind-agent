"""
DataMind Audit AI Enterprise 3.0
Single-service deployment — FastAPI serves both the frontend and all API routes.

File layout in your repo:
  index.html          ← the frontend (served at / and /app)
  main.py             ← this file
  Dockerfile
  railway.toml
  requirements.txt
  api/
    auth_routes.py
    audit_routes.py
    report_routes.py
    analyse_routes.py
  core/
    database.py
    report_engine.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

from api.auth_routes    import router as auth_router
from api.audit_routes   import router as audit_router
from api.report_routes  import router as report_router
from api.analyse_routes import router as analyse_router
from core.database      import client as mongo_client

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DataMind Audit AI Enterprise 3.0",
    description="AI-powered audit, fraud detection, and financial analysis platform.",
    version="3.0.0",
    docs_url="/api/docs",       # Swagger UI at /api/docs
    redoc_url="/api/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("✅ DataMind 3.0 started")
    print("   MongoDB client:", mongo_client)
    print("   ANTHROPIC_API_KEY set:", bool(os.environ.get("ANTHROPIC_API_KEY")))
    print("   GEMINI_API_KEY   set:", bool(os.environ.get("GEMINI_API_KEY")))

# ── API Routes ────────────────────────────────────────────────────────────────
app.include_router(auth_router,    prefix="/api/auth",           tags=["Auth"])
app.include_router(audit_router,   prefix="/api/audit",          tags=["Audit"])
app.include_router(report_router,  prefix="/api/report",         tags=["Report"])
app.include_router(analyse_router, prefix="/api/v1/analysis",    tags=["Analysis"])

# ── Health (Railway checks this) ──────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "healthy", "service": "DataMind 3.0"}

# ── System info ───────────────────────────────────────────────────────────────
@app.get("/api/status", tags=["System"])
def status():
    return {
        "system":  "DataMind Enterprise 3.0",
        "status":  "LIVE",
        "ai": {
            "primary":   "Claude (Anthropic)",
            "secondary": "Gemini (Google)",
            "fallback":  "Local statistical engine",
        },
        "features": [
            "Multi-Tenant SaaS",
            "JWT Authentication",
            "Fraud AI Engine",
            "Audit Intelligence",
            "Report Generation",
            "Claude + Gemini Dual AI",
        ],
    }

# ── Frontend — serve index.html ───────────────────────────────────────────────
# The HTML file sits in the root of your repo alongside main.py
FRONTEND = os.path.join(os.path.dirname(__file__), "index.html")

@app.get("/", include_in_schema=False)
@app.get("/app", include_in_schema=False)
def serve_frontend():
    """Serve the DataMind frontend application."""
    if os.path.exists(FRONTEND):
        return FileResponse(FRONTEND, media_type="text/html")
    return {
        "error": "index.html not found in root directory.",
        "hint":  "Make sure index.html is in the same folder as main.py",
    }

# ── Catch-all: send unknown paths back to the frontend ────────────────────────
@app.get("/{full_path:path}", include_in_schema=False)
def catch_all(full_path: str):
    """Any unknown path that doesn't start with /api → return the frontend."""
    if full_path.startswith("api/"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="API route not found")
    if os.path.exists(FRONTEND):
        return FileResponse(FRONTEND, media_type="text/html")
    return {"error": "Frontend not found"}
