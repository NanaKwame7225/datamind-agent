from fastapi import Request
from core.ml_fraud_engine import fraud_detector, save_fraud_results

@router.post("/fraud/predict")
async def predict(data: dict, request: Request):

    tenant_id = request.state.tenant_id

    results = fraud_detector.predict(data["transactions"])

    await save_fraud_results(tenant_id, results)

    return {
        "tenant": tenant_id,
        "results": results
    }
