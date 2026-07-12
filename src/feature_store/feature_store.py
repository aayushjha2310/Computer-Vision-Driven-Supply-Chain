"""Unified feature store for storing and serving reusable predictive components."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class FeatureStore:
    """Store and serve unified, reusable predictive feature components."""

    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.store_path / "registry.json"
        self._registry: dict[str, Any] = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        if self.registry_path.exists():
            with open(self.registry_path, encoding="utf-8") as f:
                return json.load(f)
        return {"features": {}, "feature_groups": {}, "metadata": {}}

    def _save_registry(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def register_feature_group(
        self,
        name: str,
        df: pd.DataFrame,
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        timestamp = datetime.utcnow().isoformat()
        file_path = self.store_path / f"{name}.parquet"
        df.to_parquet(file_path, index=False)
        entry = {
            "name": name,
            "path": str(file_path),
            "columns": list(df.columns),
            "n_rows": len(df),
            "created_at": timestamp,
            "description": description,
            "tags": tags or [],
        }
        self._registry["feature_groups"][name] = entry
        self._save_registry()
        return str(file_path)

    def get_feature_group(self, name: str) -> pd.DataFrame:
        if name not in self._registry["feature_groups"]:
            raise KeyError(f"Feature group '{name}' not found")
        path = self._registry["feature_groups"][name]["path"]
        return pd.read_parquet(path)

    def register_transformer(self, name: str, transformer: Any) -> None:
        path = self.store_path / f"{name}_transformer.pkl"
        joblib.dump(transformer, path)
        self._registry["features"][name] = {
            "type": "transformer",
            "path": str(path),
            "registered_at": datetime.utcnow().isoformat(),
        }
        self._save_registry()

    def get_transformer(self, name: str) -> Any:
        if name not in self._registry["features"]:
            raise KeyError(f"Transformer '{name}' not found")
        return joblib.load(self._registry["features"][name]["path"])

    def compile_unified_features(self, groups: list[str]) -> pd.DataFrame:
        frames = []
        for group in groups:
            df = self.get_feature_group(group)
            df = df.add_prefix(f"{group}__")
            frames.append(df)
        unified = pd.concat(frames, axis=1)
        meta = {
            "compiled_at": datetime.utcnow().isoformat(),
            "source_groups": groups,
            "n_features": unified.shape[1],
            "n_rows": unified.shape[0],
        }
        self._registry["metadata"]["last_compilation"] = meta
        self._save_registry()
        return unified

    def list_feature_groups(self) -> list[str]:
        return list(self._registry["feature_groups"].keys())

    def get_metadata(self) -> dict[str, Any]:
        return self._registry
