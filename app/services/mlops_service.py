import logging
from config.settings import settings
logger = logging.getLogger(__name__)

class MLOpsService:
    def mlflow_start_run(self, experiment_name: str, run_name: str) -> str:
        import mlflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        return run.info.run_id

    def mlflow_log(self, run_id, params, metrics):
        import mlflow
        with mlflow.start_run(run_id=run_id):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

mlops_service = MLOpsService()
