from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# =========================
# ROUTES
# =========================
from api.audit_routes import router as audit_router
from api.document_routes import router as document_router
from api.integration_routes import router as integration_router

# =========================
# DATABASE
# =========================
from core.database import client

# =========================
# MIDDLEWARE
# =========================
from core.tenant_middleware import TenantMiddleware

# =========================
# APP INIT
# =========================
app = FastAPI(
    title="DataMind Audit AI Enterprise",
    version="3.0.0",
    description="Multi-Tenant AI Audit, Finance & Tax Intelligence Platform"
)

# =========================
# STARTUP EVENT (MongoDB)
# =========================
@app.on_event("startup")
async def startup_db():
    print("🚀 MongoDB Connected:", client)

# =========================
# MIDDLEWARE
# =========================
app.add_middleware(TenantMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ROUTES
# =========================
app.include_router(audit_router, prefix="/api/audit", tags=["Audit"])
app.include_router(document_router, prefix="/api/document", tags=["Document AI"])
app.include_router(integration_router, prefix="/api/integrations", tags=["Integrations"])

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
async def health():
    return {
        "status": "running",
        "version": "3.0.0",
        "system": "DataMind Audit AI Enterprise"
    }

# =========================
# ROOT
# =========================
@app.get("/")
async def home():
    return {
        "message": "DataMind Audit AI Enterprise Running 🚀",
        "modules": [
            "Audit Engine",
            "Fraud ML Engine",
            "Document AI",
            "Tax Intelligence",
            "Enterprise Integrations",
            "Multi-Tenant SaaS"
        ]
    }
