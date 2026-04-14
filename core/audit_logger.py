import datetime

def log_event(tenant_id, action, status):
    return {
        "tenant_id": tenant_id,
        "action": action,
        "status": status,
        "timestamp": str(datetime.datetime.utcnow())
    }
