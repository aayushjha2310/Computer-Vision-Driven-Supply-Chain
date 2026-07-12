"""Configuration and path utilities."""

import os
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | None = None) -> dict[str, Any]:
    root = get_project_root()
    path = Path(config_path) if config_path else root / "config" / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key in ("data_raw", "data_processed", "data_synthetic", "models", "feature_store"):
        if key in config.get("paths", {}):
            config["paths"][key] = str(root / config["paths"][key])
    mlflow_path = root / config["paths"].get("mlflow_uri", "mlruns")
    os.makedirs(mlflow_path, exist_ok=True)
    db_path = (root / "mlflow.db").resolve().as_posix()
    config["paths"]["mlflow_uri"] = f"sqlite:///{db_path}"
    config["paths"]["mlflow_artifacts"] = str(mlflow_path)
    return config


def ensure_dirs(config: dict[str, Any]) -> None:
    root = get_project_root()
    paths = config.get("paths", {})
    for key in ("data_raw", "data_processed", "data_synthetic", "models", "feature_store"):
        if key in paths:
            os.makedirs(paths[key], exist_ok=True)
    os.makedirs(root / "data" / "checkpoints", exist_ok=True)
    os.makedirs(root / "mlruns", exist_ok=True)
