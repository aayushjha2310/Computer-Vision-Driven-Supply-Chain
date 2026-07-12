"""Competing gradient-boosted models for price optimization and anomaly detection."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from src.ml_models.classical_ml import ClassicalMLPipeline


def _try_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        return LGBMRegressor, LGBMClassifier
    except (ImportError, OSError, FileNotFoundError):
        return None, None


def _try_import_catboost():
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        return CatBoostRegressor, CatBoostClassifier
    except (ImportError, OSError, FileNotFoundError):
        return None, None


class GradientBoostingEnsemble:
    """XGBoost, LightGBM, and CatBoost competing models for forecasting."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.classical = ClassicalMLPipeline(random_state)
        self.trained_models: dict[str, Any] = {}
        self.best_model_name: str | None = None
        self.best_score: float = float("inf")
        self.feature_cols: list[str] = []
        self._build_model_factories()

    def _build_model_factories(self) -> None:
        LGBMRegressor, LGBMClassifier = _try_import_lightgbm()
        CatBoostRegressor, CatBoostClassifier = _try_import_catboost()
        rs = self.random_state

        self.REGRESSORS = {
            "xgboost": lambda: XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05, random_state=rs, verbosity=0
            ),
        }
        self.CLASSIFIERS = {
            "xgboost": lambda: XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05, random_state=rs, verbosity=0
            ),
        }

        if LGBMRegressor is not None:
            self.REGRESSORS["lightgbm"] = lambda: LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05, random_state=rs, verbose=-1
            )
            self.CLASSIFIERS["lightgbm"] = lambda: LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05, random_state=rs, verbose=-1
            )
        else:
            self.REGRESSORS["sklearn_gbm"] = lambda: GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05, random_state=rs
            )

        if CatBoostRegressor is not None:
            self.REGRESSORS["catboost"] = lambda: CatBoostRegressor(
                iterations=200, depth=6, learning_rate=0.05, random_state=rs, verbose=0
            )
            self.CLASSIFIERS["catboost"] = lambda: CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.05, random_state=rs, verbose=0
            )
        else:
            self.CLASSIFIERS["sklearn_rf"] = lambda: RandomForestClassifier(
                n_estimators=200, max_depth=6, random_state=rs
            )

    def train_regressors(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        X, y, used_features = self.classical.prepare_features(df, feature_cols, target_col)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        results = {}
        for name, factory in self.REGRESSORS.items():
            model = factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = self.classical.evaluate_regression(y_test, y_pred)
            cv_metrics = self.classical.temporal_kfold_cv(model, X, y, n_folds=5)
            results[name] = {**metrics, **cv_metrics}
            self.trained_models[name] = model
            if metrics["rmse"] < self.best_score:
                self.best_score = metrics["rmse"]
                self.best_model_name = name
        self.feature_cols = used_features
        return {
            "results": results,
            "best_model": self.best_model_name,
            "feature_cols": used_features,
            "X_test": X_test,
            "y_test": y_test,
        }

    def train_classifiers(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        class_weights: dict | None = None,
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        X, y, used_features = self.classical.prepare_features(df, feature_cols, target_col)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        results = {}
        for name, factory in self.CLASSIFIERS.items():
            model = factory()
            if class_weights and name == "xgboost":
                model.set_params(scale_pos_weight=class_weights.get(1, 1.0))
            if class_weights and name == "sklearn_rf":
                model.set_params(class_weight="balanced")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            metrics = self.classical.evaluate_classification(y_test, y_pred, y_prob)
            results[name] = metrics
            self.trained_models[f"fraud_{name}"] = model
        return {"results": results, "feature_cols": used_features}

    def predict_price(self, X: np.ndarray, model_name: str | None = None) -> np.ndarray:
        name = model_name or self.best_model_name
        if name not in self.trained_models:
            raise ValueError(f"Model '{name}' not trained")
        return self.trained_models[name].predict(X)

    def get_best_model(self) -> Any:
        if not self.best_model_name:
            raise ValueError("No model trained yet")
        return self.trained_models[self.best_model_name]

    def forecast_regional_prices(self, df: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
        if not self.best_model_name:
            raise ValueError("Train models first")
        model = self.trained_models[self.best_model_name]
        available = [c for c in self.feature_cols if c in df.columns]
        forecasts = []
        for region in df["region"].unique() if "region" in df.columns else ["ALL"]:
            subset = df[df["region"] == region] if "region" in df.columns else df
            X = subset[available].fillna(0).values
            X_scaled = self.classical.scaler.transform(X)
            preds = model.predict(X_scaled)
            forecasts.append(
                pd.DataFrame(
                    {
                        "region": region,
                        "forecast_price": preds,
                        "horizon_days": horizon_days,
                    }
                )
            )
        return pd.concat(forecasts, ignore_index=True)
