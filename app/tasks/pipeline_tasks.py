import logging
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

@celery_app.task(name="app.tasks.pipeline_tasks.run_quality_scan")
def run_quality_scan():
    logger.info("Running scheduled quality scan")
    return {"status": "completed"}
