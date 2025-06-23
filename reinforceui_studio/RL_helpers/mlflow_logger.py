import mlflow
import os
from typing import Optional, Dict, Any


class MLflowLogger:
    def __init__(self, experiment_name: str = "ReinforceUI", run_name: Optional[str] = None, use_mlflow: bool = True, tags: Optional[Dict[str, Any]] = None):
        self.use_mlflow = use_mlflow
        self.run = None
        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)
            self.run = mlflow.start_run(run_name=run_name)
            if tags:
                mlflow.set_tags(tags)

    def log_param(self, key: str, value: Any):
        if self.use_mlflow:
            mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]):
        if self.use_mlflow:
            mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if self.use_mlflow:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, filepath: str):
        if self.use_mlflow and os.path.exists(filepath):
            mlflow.log_artifact(filepath)

    def log_artifacts(self, dirpath: str):
        if self.use_mlflow and os.path.isdir(dirpath):
            mlflow.log_artifacts(dirpath)

    def set_tag(self, key: str, value: Any):
        if self.use_mlflow:
            mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, Any]):
        if self.use_mlflow:
            mlflow.set_tags(tags)

    def end_run(self):
        if self.use_mlflow and self.run is not None:
            mlflow.end_run()
            self.run = None
