import os
import mlflow
from functools import wraps
from typing import Optional, Dict, Any, Callable

import torch
from mlflow.models.signature import infer_signature


def check_enabled(func: Callable) -> Callable:
    """Decorator to gracefully skip logging if MLflow is disabled."""

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Optional[Any]:  # noqa: ANN001, ANN002, ANN003
        if not self.use_mlflow:
            return None
        return func(self, *args, **kwargs)
    return wrapper


class MLflowLogger:
    def __init__(
        self,
        experiment_name: str = "ReinforceUI Experiment",
        run_name: str = "Run 1",
        use_mlflow: bool = True,
        tags: Optional[Dict[str, Any]] = None,
        tracking_uri: Optional[str] = None,
    ) -> None:
        """Initialize the MLflowLogger.

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
            tracking_uri = os.path.join(
                os.path.expanduser("~"),
                "reinforceui_studio_logs/mlflow_tracking",
            )

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def set_experiment(self, experiment_name: str) -> None:
        """Set the MLflow experiment.

        Args:
            experiment_name (str): Name of the experiment to set.
        """
        self.experiment_name = experiment_name
        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)

    @check_enabled
    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start an MLflow run.

        Args:
            run_name (Optional[str]): Name of the run to start.
        """
        if run_name is not None:
            self.run_name = run_name
        self.run = mlflow.start_run(run_name=self.run_name)
        if self.tags:
            mlflow.set_tags(self.tags)

    @check_enabled
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run(status="FINISHED")

    @check_enabled
    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter to MLflow.

        Args:
            key (str): Parameter name.
            value (Any): Parameter value.
        """
        mlflow.log_param(key, value)

    @check_enabled
    def log_params(self, params: dict) -> None:
        """Log multiple parameters to MLflow.

        Args:
            params (dict): Dictionary of parameters to log.
        """
        mlflow.log_params(params)

    @check_enabled
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric to MLflow.

        Args:
            key (str): Metric name.
            value (float): Metric value.
            step (Optional[int]): Step at which the metric was logged.
        """
        mlflow.log_metric(key, value, step=step)

    @check_enabled
    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log multiple metrics to MLflow.

        Args:
            metrics (dict): Dictionary of metrics to log.
            step (Optional[int]): Step at which the metrics were logged.
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    @check_enabled
    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """Log an artifact to MLflow.

        Args:
            local_path (str): Path to the local file.
            artifact_path (Optional[str]): Path in the artifact store.
        """
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)
        else:
            print(f"Warning: File not found: {local_path}")

    @check_enabled
    def log_artifacts(
        self, local_dir: str, artifact_path: Optional[str] = None
    ) -> None:
        """Log multiple artifacts to MLflow.

        Args:
            local_dir (str): Directory containing artifacts.
            artifact_path (Optional[str]): Path in the artifact store.
        """
        if os.path.isdir(local_dir):
            mlflow.log_artifacts(local_dir, artifact_path)
        else:
            print(f"Warning: Directory not found: {local_dir}")

    @check_enabled
    def set_tags(self, tags: dict) -> None:
        """Set tags for the MLflow run.

        Args:
            tags (dict): Dictionary of tags to set.
        """
        mlflow.set_tags(tags)

    @check_enabled
    def log_model(
        self,
        model: Any,
        model_type: str = "pytorch",
        model_name: str = "model",
        registered_model_name: Optional[str] = None,
        input_example: Optional[Any] = None,
        model_input: Optional[Any] = None,
        device: str = "cpu",
    ) -> None:
        """Log a machine learning model with optional registration.

        Args:
            model (Any): The model to log.
            model_type (str): Type of the model (e.g., 'pytorch').
            model_name (str): Name to log the model under.
            registered_model_name (Optional[str]): Name to register the model as.
            input_example (Optional[Any]): Example input for MLflow signature.
            model_input (Optional[Any]): Actual input for signature inference.
            device (str): Device string (e.g., 'cpu' or 'cuda:0').
        """
        if model_type == "pytorch":
            model.to(device)
            signature = None

            if input_example is not None:
                # Use model_input if provided, else infer from input_example
                if model_input is not None:
                    if isinstance(model_input, torch.Tensor):
                        model_input = model_input.to(device)
                    else:
                        model_input = torch.from_numpy(model_input).to(device)
                    model_output = model(model_input)
                else:
                    # Accept both ndarray and dict for input_example
                    if isinstance(input_example, dict):
                        model_input = {
                            k: torch.from_numpy(v).to(device)
                            for k, v in input_example.items()
                        }
                        model_output = model(**model_input)
                    else:
                        model_input = torch.from_numpy(input_example).to(device)
                        model_output = model(model_input)

                if isinstance(model_output, torch.Tensor):
                    model_output = model_output.detach().cpu().numpy()
                signature = infer_signature(
                    model_input=input_example, model_output=model_output
                )
            else:
                print(
                    "Warning: Logging PyTorch model without signature. Inference may fail later."
                )

            mlflow.pytorch.log_model(
                pytorch_model=model,
                name=model_name,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
            )
        else:
            print(
                f"Error: Unsupported model type '{model_type}'. Only 'pytorch' supported here."
            )
