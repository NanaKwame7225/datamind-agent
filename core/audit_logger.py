from core.database import logs
from datetime import datetime

async def log_event(tenant_id, action, status="SUCCESS"):
    log = {
        "tenant_id": tenant_id,
        "action": action,
        "status": status,
        "timestamp": datetime.utcnow()
    }

    await logs.insert_one(log)
    return log
