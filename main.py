from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.audit_routes import router as audit_router
from api.document_routes import router as document_router
from api.auth_routes import router as auth_router
from api.report_routes import router as report_router

from core.database import client

app = FastAPI(
    title="DataMind Audit AI Enterprise",
    version="2.0.0"
)

# =========================
# DATABASE CONNECTION
# =========================
@app.on_event("startup")
async def startup_db():
    print("MongoDB Connected:", client)

# =========================
# MIDDLEWARE
# =========================
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
app.include_router(auth_router, prefix="/api/auth")
app.include_router(audit_router, prefix="/api/audit")
app.include_router(document_router, prefix="/api/document")
app.include_router(report_router, prefix="/api/report")

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {
        "system": "DataMind Audit AI Enterprise",
        "status": "running",
        "features": [
            "Multi-Tenant Audit Engine",
            "Fraud Detection AI",
            "Tax Intelligence",
            "Accounting Automation",
            "Report Generation",
            "PDF + Excel Export"
        ]
    }
