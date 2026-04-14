from core.database import findings

async def save_fraud_results(tenant_id, results):
    record = {
        "tenant_id": tenant_id,
        "type": "FRAUD_DETECTION",
        "results": results
    }

    await findings.insert_one(record)
    return record
