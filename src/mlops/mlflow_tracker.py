"""MLflow experiment tracking and model registry."""

from pathlib import Path
from typing import Any

import joblib
import mlflow


class MLflowTracker:
    """Track experiments and version registered models with MLflow."""

    def __init__(self, tracking_uri: str, experiment_name: str, artifact_location: str | None = None):
        mlflow.set_tracking_uri(tracking_uri)
        if artifact_location:
            mlflow.set_experiment(experiment_name)
        else:
            mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def start_run(self, run_name: str | None = None) -> mlflow.ActiveRun:
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        safe_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(safe_params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_model_artifact(self, model: Any, name: str) -> str:
        model_path = Path("models") / f"{name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="models")
        return str(model_path)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    def register_model(self, model: Any, name: str, metrics: dict[str, float], params: dict[str, Any]) -> str:
        with self.start_run(run_name=f"register_{name}"):
            self.log_params(params)
            self.log_metrics(metrics)
            self.log_model_artifact(model, name)
            return mlflow.active_run().info.run_id

    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        local_path = Path("models") / f"{model_name}.pkl"
        if local_path.exists():
            return joblib.load(local_path)
        raise FileNotFoundError(f"Model not found: {model_name}")
