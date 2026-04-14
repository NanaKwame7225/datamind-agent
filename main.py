from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth_routes import router as auth_router
from api.audit_routes import router as audit_router
from api.report_routes import router as report_router

from core.database import client

app = FastAPI(title="DataMind Audit AI Enterprise 3.0")

@app.on_event("startup")
async def startup():
    print("MongoDB Connected:", client)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth")
app.include_router(audit_router, prefix="/api/audit")
app.include_router(report_router, prefix="/api/report")

@app.get("/")
def home():
    return {
        "system": "DataMind Enterprise 3.0",
        "status": "LIVE",
        "features": [
            "Multi-Tenant SaaS",
            "JWT Authentication",
            "Fraud AI Engine",
            "Audit Intelligence",
            "Report Generation",
            "Enterprise Ready"
        ]
    }
