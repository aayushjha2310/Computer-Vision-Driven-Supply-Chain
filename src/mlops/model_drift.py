"""Continuous model drift evaluation monitors."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


class ModelDriftMonitor:
    """Detect data/prediction shifts post-deployment."""

    def __init__(self, drift_threshold: float = 0.15):
        self.drift_threshold = drift_threshold
        self.reference_stats: dict[str, Any] = {}

    def set_reference(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        for col in feature_cols:
            if col in df.columns:
                self.reference_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "distribution": df[col].values,
                }

    def detect_feature_drift(self, current_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
        drift_report = {}
        for col in feature_cols:
            if col not in self.reference_stats or col not in current_df.columns:
                continue
            ref = self.reference_stats[col]
            current = current_df[col].values
            ks_stat, ks_pvalue = stats.ks_2samp(ref["distribution"], current)
            current_mean = float(current.mean())
            mean_shift = abs(current_mean - ref["mean"]) / (ref["std"] + 1e-8)
            drifted = ks_stat > self.drift_threshold or mean_shift > self.drift_threshold
            drift_report[col] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "mean_shift": float(mean_shift),
                "drifted": drifted,
            }
        n_drifted = sum(1 for v in drift_report.values() if v["drifted"])
        return {
            "features": drift_report,
            "n_drifted": n_drifted,
            "drift_detected": n_drifted > 0,
            "drift_ratio": n_drifted / max(len(drift_report), 1),
        }

    def detect_prediction_drift(
        self,
        reference_preds: np.ndarray,
        current_preds: np.ndarray,
    ) -> dict[str, Any]:
        ks_stat, ks_pvalue = stats.ks_2samp(reference_preds, current_preds)
        mean_shift = abs(current_preds.mean() - reference_preds.mean()) / (reference_preds.std() + 1e-8)
        return {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "mean_shift": float(mean_shift),
            "drift_detected": ks_stat > self.drift_threshold,
            "reference_mean": float(reference_preds.mean()),
            "current_mean": float(current_preds.mean()),
        }

    def generate_drift_report(self, feature_drift: dict, prediction_drift: dict | None = None) -> str:
        lines = ["=== Model Drift Report ==="]
        lines.append(f"Features drifted: {feature_drift['n_drifted']}/{len(feature_drift['features'])}")
        for col, info in feature_drift["features"].items():
            status = "DRIFT" if info["drifted"] else "OK"
            lines.append(f"  [{status}] {col}: KS={info['ks_statistic']:.4f}, shift={info['mean_shift']:.4f}")
        if prediction_drift:
            status = "DRIFT" if prediction_drift["drift_detected"] else "OK"
            lines.append(f"Predictions [{status}]: KS={prediction_drift['ks_statistic']:.4f}")
        return "\n".join(lines)
