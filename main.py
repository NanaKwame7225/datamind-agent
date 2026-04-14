from fastapi import FastAPI

from api.audit_routes import router as audit_router
from api.document_routes import router as document_router

app = FastAPI(title="DataMind Audit AI")

# =========================
# REGISTER ROUTES
# =========================

app.include_router(audit_router, prefix="/api/audit")
app.include_router(document_router, prefix="/api/document")


@app.get("/")
def home():
    return {"message": "DataMind Audit AI Running"}
