from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ROUTES
from api.audit_routes import router as audit_router
from api.document_routes import router as document_router
from api.integration_routes import router as integration_router

# MIDDLEWARE
from core.tenant_middleware import TenantMiddleware

app = FastAPI(
    title="DataMind Audit AI Enterprise",
    version="3.0.0",
    description="Multi-Tenant AI Audit, Finance & Tax Intelligence Platform"
)

# =========================
# MIDDLEWARE
# =========================
app.add_middleware(TenantMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ROUTES
# =========================
app.include_router(audit_router, prefix="/api/audit")
app.include_router(document_router, prefix="/api/document")
app.include_router(integration_router, prefix="/api/integrations")

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {
        "status": "running",
        "version": "3.0.0",
        "system": "DataMind Audit AI Enterprise"
    }

# =========================
# ROOT
# =========================
@app.get("/")
def home():
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
