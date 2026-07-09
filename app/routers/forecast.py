"""
DataMind Agent — Predictive Analytics Router
POST /api/v1/forecast/predict   — forecast future periods with confidence intervals
POST /api/v1/forecast/scenario  — what-if: change a driver, see the projected effect
POST /api/v1/forecast/drivers   — which variables most move with the target
"""
import logging, traceback
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()
logger = logging.getLogger(__name__)


class ForecastRequest(BaseModel):
    data: List[dict]
    periods: int = 3
    target_columns: Optional[List[str]] = None
    time_column: Optional[str] = None


class ScenarioRequest(BaseModel):
    data: List[dict]
    target: str
    driver: str
    change_pct: float


def _svc():
    from app.services.forecast_service import forecast_service
    return forecast_service


@router.post("/predict")
async def predict(req: ForecastRequest):
    """Forecast future periods for the numeric columns in the dataset."""
    if not req.data:
        raise HTTPException(400, "Provide data as a list of records")
    if req.periods < 1 or req.periods > 24:
        raise HTTPException(400, "periods must be between 1 and 24")
    try:
        df = pd.DataFrame(req.data)
        return _svc().forecast(df, req.periods, req.target_columns, req.time_column)
    except Exception as e:
        logger.error(f"Forecast failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


@router.post("/scenario")
async def scenario(req: ScenarioRequest):
    """Estimate the effect on a target of changing a driver by some percentage."""
    if not req.data:
        raise HTTPException(400, "Provide data")
    try:
        df = pd.DataFrame(req.data)
        if req.target not in df.columns:
            return {"success": False, "error": f"Target column '{req.target}' not found in the data."}
        if req.driver not in df.columns:
            return {"success": False, "error": f"Driver column '{req.driver}' not found in the data."}
        return _svc().scenario(df, req.target, req.driver, req.change_pct)
    except Exception as e:
        logger.error(f"Scenario failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


@router.post("/drivers")
async def drivers(req: ForecastRequest):
    """Return the variables most strongly associated with each target column."""
    if not req.data:
        raise HTTPException(400, "Provide data")
    try:
        df = pd.DataFrame(req.data)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        targets = [c for c in (req.target_columns or num_cols) if c in num_cols][:3]
        result = _svc()._driver_analysis(df, targets, num_cols)
        return {"success": True, "drivers": result}
    except Exception as e:
        logger.error(f"Drivers failed: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}
