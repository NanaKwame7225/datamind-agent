from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# =========================
# ROUTES IMPORTS
# =========================
from api.audit_routes import router as audit_router
from api.document_routes import router as document_router
from api.integration_routes import router as integration_router

# =========================
# APP INITIALIZATION
# =========================
app = FastAPI(
    title="DataMind Audit AI",
    description="AI-Powered Audit, Finance, and Tax Intelligence Platform",
    version="2.0.0"
)

# =========================
# CORS CONFIGURATION
# (ALLOW FRONTEND CONNECTION)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# REGISTER ROUTES (CORE SYSTEM)
# =========================

# 🔴 AUDIT ENGINE (Fraud, Benford, Ratios, Trends, Reports)
app.include_router(audit_router, prefix="/api/audit", tags=["Audit Engine"])

# 🧾 DOCUMENT AI (OCR, Invoice validation, Receipt matching)
app.include_router(document_router, prefix="/api/document", tags=["Document AI"])

# 🔌 ENTERPRISE INTEGRATIONS (SAP, QuickBooks)
app.include_router(integration_router, prefix="/api/integrations", tags=["Enterprise Integrations"])

# =========================
# HEALTH CHECK ENDPOINT
# =========================
@app.get("/health")
def health_check():
    return {
        "status": "active",
        "system": "DataMind Audit AI",
        "version": "2.0.0"
    }

# =========================
# ROOT ENDPOINT
# =========================
@app.get("/")
def home():
    return {
        "message": "DataMind Audit AI Running 🚀",
        "modules": [
            "Audit Intelligence",
            "Document AI",
            "Fraud Detection ML",
            "Enterprise Integrations"
        ]
    }
