from celery import Celery
from config.settings import settings

celery_app = Celery(
    "datamind",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.REDIS_URL,
)
celery_app.conf.update(task_serializer="json", result_serializer="json", accept_content=["json"])
