from fastapi import APIRouter

router = APIRouter()

@router.post("/db/connect")
async def connect_database(db_type: str, connection_string: str = ""):
    return {"connection_id": "demo-" + db_type, "db_type": db_type,
            "status": "connected", "tables": []}

@router.get("/available")
async def list_connectors():
    return {
        "llm": ["anthropic","openai","gemini","cohere","mistral"],
        "data": ["pandas","polars","numpy","dask"],
        "ml": ["scikit-learn","xgboost","lightgbm","statsmodels"],
        "databases": ["postgresql","mysql","sqlite","mongodb"],
        "warehouses": ["bigquery","snowflake"],
        "vector_stores": ["pinecone","weaviate","chroma","faiss"],
        "visualisation": ["plotly","matplotlib","seaborn","bokeh"],
        "pipeline": ["airflow","prefect","dagster","celery","ray"],
    }
