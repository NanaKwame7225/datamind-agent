import datetime

def log_event(tenant, action):
    return {
        "tenant": tenant,
        "action": action,
        "timestamp": str(datetime.datetime.utcnow())
    }
