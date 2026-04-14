from core.database import reports
from datetime import datetime

async def save_audit_report(tenant_id, report_data):
    record = {
        "tenant_id": tenant_id,
        "report": report_data,
        "created_at": datetime.utcnow()
    }

    await reports.insert_one(record)
    return record
