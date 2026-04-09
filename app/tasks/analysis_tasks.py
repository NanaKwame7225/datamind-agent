import logging, pandas as pd
from app.tasks.celery_app import celery_app
from app.services.analysis_service import analysis_service

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="app.tasks.analysis_tasks.run_full_analysis")
def run_full_analysis(self, file_path, industry, query):
    self.update_state(state="PROGRESS", meta={"step": "loading", "pct": 10})
    df = pd.read_csv(file_path)
    quality = analysis_service.quality_report(df)
    self.update_state(state="PROGRESS", meta={"step": "complete", "pct": 100})
    return {"industry": industry, "query": query, "quality": quality}
