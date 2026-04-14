from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

client = AsyncIOMotorClient(MONGO_URI)

db = client.datamind_audit_ai


# =========================
# COLLECTIONS
# =========================
tenants = db.tenants
users = db.users
transactions = db.transactions
findings = db.audit_findings
reports = db.audit_reports
logs = db.audit_logs
documents = db.documents
