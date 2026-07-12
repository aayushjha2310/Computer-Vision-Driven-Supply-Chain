"""Classical machine learning pipeline with preprocessing and metrics."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler


class ClassicalMLPipeline:
    """Scikit-learn core suite for regression, preprocessing, and metrics."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.models: dict[str, Any] = {}

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        available = [c for c in feature_cols if c in df.columns]
        X = df[available].fillna(0).values
        y = df[target_col].values
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y, available

    def temporal_kfold_cv(self, model: Any, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> dict[str, float]:
        kfold = KFold(n_splits=n_folds, shuffle=False)
        scores = cross_val_score(model, X, y, cv=kfold, scoring="neg_mean_squared_error")
        return {
            "cv_rmse_mean": float(np.sqrt(-scores.mean())),
            "cv_rmse_std": float(np.sqrt(scores.std())),
        }

    def unsupervised_damage_clustering(self, X: np.ndarray, n_clusters: int = 4) -> dict[str, Any]:
        pca = PCA(n_components=min(5, X.shape[1]), random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        return {
            "labels": labels,
            "pca_components": pca.components_,
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "cluster_centers": kmeans.cluster_centers_,
        }

    def market_clustering(self, df: pd.DataFrame, feature_cols: list[str], n_clusters: int = 4) -> pd.DataFrame:
        available = [c for c in feature_cols if c in df.columns]
        X = StandardScaler().fit_transform(df[available].fillna(0))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        result = df.copy()
        result["market_segment"] = kmeans.fit_predict(X)
        return result

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        return metrics
