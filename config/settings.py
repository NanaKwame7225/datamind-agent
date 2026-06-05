"""
DataMind Agent v2 — Settings
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str = "DataMind Agent"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production-32-chars-min"
    ALLOWED_ORIGINS: list[str] = ["https://nanakwame7225.github.io", "http://localhost:3000", "http://localhost:8080"]

    # LLM
    GROQ_API_KEY: Optional[str] = None
    GROK_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None

    # Databases
    POSTGRES_URL: Optional[str] = None
    MONGODB_URI: Optional[str] = None
    SQLITE_PATH: str = "./datamind.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"

    # Cloud
    BIGQUERY_PROJECT: Optional[str] = None
    SNOWFLAKE_ACCOUNT: Optional[str] = None
    SNOWFLAKE_USER: Optional[str] = None
    SNOWFLAKE_PASSWORD: Optional[str] = None
    SNOWFLAKE_DATABASE: Optional[str] = None
    SNOWFLAKE_WAREHOUSE: Optional[str] = None

    # Vector
    PINECONE_API_KEY: Optional[str] = None
    WEAVIATE_URL: Optional[str] = None
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001

    # MLOps
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    WANDB_API_KEY: Optional[str] = None

    # Email (for scheduled reports)
    SMTP_HOST: Optional[str] = None        # e.g. smtp.gmail.com
    SMTP_PORT: int = 465
    SMTP_USER: Optional[str] = None        # your email
    SMTP_PASSWORD: Optional[str] = None    # app password
    SMTP_FROM: Optional[str] = None        # display name

    # Stripe (for payments)
    STRIPE_SECRET_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None
    STRIPE_STARTER_PRICE_ID: Optional[str] = None
    STRIPE_PRO_PRICE_ID: Optional[str] = None
    STRIPE_ENTERPRISE_PRICE_ID: Optional[str] = None

    # File storage
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_MB: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
