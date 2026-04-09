from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "DataMind Agent"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    POSTGRES_URL: Optional[str] = None
    MONGODB_URI: Optional[str] = None
    SQLITE_PATH: str = "./datamind.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_MB: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = Settings()
