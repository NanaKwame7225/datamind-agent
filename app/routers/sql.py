"""
DataMind Agent — SQL Execution Router
POST /api/v1/sql/query      — run SQL against uploaded data
POST /api/v1/sql/natural    — natural language to SQL
POST /api/v1/sql/profile    — full data profile via SQL
GET  /api/v1/sql/samples    — sample query suggestions
"""
import logging
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


class SQLRequest(BaseModel):
    data: list[dict]
    query: Optional[str] = None
    table_name: str = "data"


class NLSQLRequest(BaseModel):
    data: list[dict]
    question: str
    table_name: str = "data"
    execute: bool = True


@router.post("/query")
async def run_sql(request: SQLRequest):
    """Execute a SQL query against provided data."""
    if not request.data:
        raise HTTPException(400, "Provide data as a list of records")
    if not request.query:
        raise HTTPException(400, "Provide a SQL query")
    try:
        from app.services.sql_service import sql_service
        df = pd.DataFrame(request.data)
        result = sql_service.execute_query(df, request.query, request.table_name)
        return result
    except Exception as e:
        logger.error(f"SQL router error: {e}")
        raise HTTPException(500, str(e))


@router.post("/natural")
async def natural_to_sql(request: NLSQLRequest):
    """Convert a natural language question to SQL and optionally execute it."""
    if not request.data:
        raise HTTPException(400, "Provide data")
    if not request.question:
        raise HTTPException(400, "Provide a question")
    try:
        from app.services.sql_service import sql_service
        df = pd.DataFrame(request.data)
        sql_result = sql_service.natural_language_to_sql(request.question, df, request.table_name)
        if request.execute:
            exec_result = sql_service.execute_query(df, sql_result["sql"], request.table_name)
            return {**sql_result, "result": exec_result}
        return sql_result
    except Exception as e:
        logger.error(f"NL-SQL error: {e}")
        raise HTTPException(500, str(e))


@router.post("/profile")
async def profile_data(request: SQLRequest):
    """Generate a comprehensive SQL-powered data profile."""
    if not request.data:
        raise HTTPException(400, "Provide data")
    try:
        from app.services.sql_service import sql_service
        df = pd.DataFrame(request.data)
        return sql_service.profile_table(df, request.table_name)
    except Exception as e:
        logger.error(f"Profile error: {e}")
        raise HTTPException(500, str(e))


@router.post("/load")
async def load_schema(request: SQLRequest):
    """Load data and return schema + sample queries."""
    if not request.data:
        raise HTTPException(400, "Provide data")
    try:
        from app.services.sql_service import sql_service
        df = pd.DataFrame(request.data)
        return sql_service.load_dataframe(df, request.table_name)
    except Exception as e:
        logger.error(f"Load error: {e}")
        raise HTTPException(500, str(e))
