import os
import mlflow
from functools import wraps
from typing import Optional, Dict, Any, Callable
from mlflow.models.signature import infer_signature


def check_enabled(func: Callable) -> Callable:
    """Decorator to gracefully skip logging if MLflow is disabled."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.use_mlflow:
            return  # Skip execution if MLflow is not active
        return func(self, *args, **kwargs)
    return wrapper


class MLflowLogger:
    def __init__(
            self,
            experiment_name: str = "ReinforceUI",
            run_name: Optional[str] = None,
            use_mlflow: bool = True,
            tags: Optional[Dict[str, Any]] = None,
            tracking_uri: Optional[str] = None
    ) -> None:
        """        Initialize the MLflow logger.
        Args:
            experiment_name: Name of the MLflow experiment.
            run_name: Name of the MLflow run. If None, a random name will be generated.
            use_mlflow: Whether to use MLflow for logging. If False, no logging will occur.
            tags: Optional dictionary of tags to set for the run.
            tracking_uri: Optional URI for the MLflow tracking server. If None, defaults to a local directory.
        """
        self.use_mlflow = use_mlflow
        self.run = None
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        if not self.use_mlflow:
            print("[MLflowLogger] MLflow logging is disabled.")
            return

        if tracking_uri is None:
            tracking_uri = os.path.join(os.path.expanduser("~"), "mlflow_tracking")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.run = mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)

    # def start_run(self, run_name: Optional[str] = None) -> None:
    #     """Start a new MLflow run."""
    #     if self.run is not None:
    #         print("Warning: A run is already active. Ending the previous run.")
    #         mlflow.end_run()
    #     self.run = mlflow.start_run(run_name="run1")
    #     print(f"MLflow run '{run_name}' started successfully.")

    @check_enabled
    def log_param(self, key: str, value: Any) -> None:
        mlflow.log_param(key, value)

    @check_enabled
    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    @check_enabled
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        mlflow.log_metric(key, value, step=step)

    @check_enabled
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    @check_enabled
    def log_artifact(self, filepath: str) -> None:
        if os.path.exists(filepath):
            mlflow.log_artifact(filepath)
        else:
            print(f"Warning: File not found: {filepath}")

    @check_enabled
    def log_artifacts(self, dirpath: str) -> None:
        if os.path.isdir(dirpath):
            mlflow.log_artifacts(dirpath)
        else:
            print(f"Warning: Directory not found: {dirpath}")

    @check_enabled
    def set_tag(self, key: str, value: Any) -> None:
        mlflow.set_tag(key, value)

    @check_enabled
    def set_tags(self, tags: Dict[str, Any]) -> None:
        mlflow.set_tags(tags)

    @check_enabled
    def end_run(self) -> None:
        if self.run is not None:
            mlflow.end_run()
            self.run = None
            print("MLflow run ended successfully.")

    def log_model(self, model, model_type: str = "pytorch", model_name: str = "model",
              registered_model_name: Optional[str] = None,
              input_example=None):
        """
        Log a machine learning model with optional registration.
        """
        if model_type == "pytorch":
            if input_example is not None:
                signature = infer_signature(input_example, model(input_example))
            else:
                signature = None
                print("Warning: Logging PyTorch model without signature. Inference may fail later.")

            mlflow.pytorch.log_model(
                pytorch_model=model,
                name=model_name,
                registered_model_name=registered_model_name,
                signature=signature
            )

        else:
            print(f"Error: Unsupported model type '{model_type}'. Only 'pytorch' supported here.")

