"""SMOTE and class weight handling for imbalanced fraud/defect data."""

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


class ImbalancedDataHandler:
    """Manage highly sparse fraud/defect occurrences with SMOTE and class weights."""

    def __init__(self, smote_ratio: float = 0.5, random_state: int = 42):
        self.smote_ratio = smote_ratio
        self.random_state = random_state

    def compute_class_weights(self, y: np.ndarray) -> dict[int, float]:
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        return dict(zip(classes.astype(int), weights))

    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        minority_count = int(np.sum(y == 1))
        majority_count = int(np.sum(y == 0))
        if minority_count < 6:
            return X, y
        target = min(majority_count, int(minority_count / self.smote_ratio))
        k_neighbors = min(5, minority_count - 1)
        smote = SMOTE(
            sampling_strategy={1: target},
            random_state=self.random_state,
            k_neighbors=k_neighbors,
        )
        return smote.fit_resample(X, y)

    def prepare_fraud_dataset(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "is_fraud",
        use_smote: bool = True,
    ) -> dict[str, Any]:
        available = [c for c in feature_cols if c in df.columns]
        X = df[available].fillna(0).values
        y = df[target_col].values.astype(int)
        class_weights = self.compute_class_weights(y)
        original_distribution = {
            "class_0": int(np.sum(y == 0)),
            "class_1": int(np.sum(y == 1)),
        }
        if use_smote and np.sum(y == 1) >= 6:
            X, y = self.apply_smote(X, y)
        return {
            "X": X,
            "y": y,
            "class_weights": class_weights,
            "original_distribution": original_distribution,
            "resampled_distribution": {
                "class_0": int(np.sum(y == 0)),
                "class_1": int(np.sum(y == 1)),
            },
            "feature_cols": available,
        }

    def prepare_defect_dataset(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "is_defect",
    ) -> dict[str, Any]:
        return self.prepare_fraud_dataset(df, feature_cols, target_col, use_smote=True)
