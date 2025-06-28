import os
import mlflow
from functools import wraps
from typing import Optional, Dict, Any, Callable

import torch
from mlflow.models.signature import infer_signature
from mlflow.transformers import generate_signature_output


def check_enabled(func: Callable) -> Callable:
    """Decorator to gracefully skip logging if MLflow is disabled."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.use_mlflow:
            return  None
        return func(self, *args, **kwargs)
    return wrapper


class MLflowLogger:
    def __init__(
            self,
            experiment_name: str = "ReinforceUI Experiment",
            run_name: str = "Run 1",
            use_mlflow: bool = True,
            tags: Optional[Dict[str, Any]] = None,
            tracking_uri: Optional[str] = None
    ) -> None:
        """Initialize the MLflow logger.
        Args:
            experiment_name: Name of the MLflow experiment.
            run_name: Name of the MLflow run. If None, a random name will be generated.
            use_mlflow: Whether to use MLflow for logging. If False, no logging will occur.
            tags: Optional dictionary of tags to set for the run.
            tracking_uri: Optional URI for the MLflow tracking server. If None, defaults to a local directory.

        Returns:
            None
        """
        self.use_mlflow = use_mlflow
        self.run = None
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.tags = tags

        if not self.use_mlflow:
            print("[MLflowLogger] MLflow logging is disabled.")
            return

        if tracking_uri is None:
            tracking_uri = os.path.join(os.path.expanduser("~"), "mlflow_tracking")
            # tracking_uri = "file:///home/david/mlflow_tracking"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @check_enabled
    def start_run(self) -> None:
        self.run = mlflow.start_run(run_name=self.run_name)
        if self.tags:
            mlflow.set_tags(self.tags)

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
        mlflow.end_run(status="FINISHED")

    @check_enabled
    def log_model(self, model, model_type: str = "pytorch", model_name: str = "model",
              registered_model_name: Optional[str] = None,
              input_example=None, model_input=None):
        """
        Log a machine learning model with optional registration.
        input_example: numpy array or DataFrame for MLflow
        model_input: actual input to call the model for signature inference (can be tensor, tuple, etc.)
        """
        if model_type == "pytorch":
            signature = None
            if input_example is not None:
                # Use model_input if provided, else infer from input_example
                if model_input is not None:
                    if isinstance(model_input, (tuple, list)):
                        model_output = model(*model_input)
                    else:
                        model_output = model(model_input)
                else:
                    # Accept both ndarray and dict for input_example
                    if isinstance(input_example, dict):
                        model_output = model(**{k: torch.from_numpy(v) for k, v in input_example.items()})
                    else:
                        model_output = model(torch.from_numpy(input_example))

                if isinstance(model_output, torch.Tensor):
                    model_output = model_output.detach().cpu().numpy()
                signature = infer_signature(model_input=input_example, model_output=model_output)
            else:
                print("Warning: Logging PyTorch model without signature. Inference may fail later.")

            mlflow.pytorch.log_model(
                pytorch_model=model,
                name=model_name,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )
        else:
            print(f"Error: Unsupported model type '{model_type}'. Only 'pytorch' supported here.")
