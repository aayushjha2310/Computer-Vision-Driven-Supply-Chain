"""Optuna hyperparameter tuning for automated parameter search sweeps."""

from typing import Any, Callable

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


class HyperparameterTuner:
    """Automated hyperparameter search using Optuna."""

    def __init__(self, n_trials: int = 20, random_state: int = 42):
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params: dict[str, Any] = {}
        self.study: optuna.Study | None = None

    def tune_xgboost_regressor(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
    ) -> dict[str, Any]:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": self.random_state,
                "verbosity": 0,
            }
            model = XGBRegressor(**params)
            scores = cross_val_score(model, X, y, cv=n_folds, scoring="neg_mean_squared_error")
            return float(np.sqrt(-scores.mean()))

        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.best_params = self.study.best_params
        return {
            "best_params": self.best_params,
            "best_rmse": self.study.best_value,
            "n_trials": len(self.study.trials),
        }

    def tune_generic(
        self,
        model_factory: Callable[[optuna.Trial], Any],
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = "neg_mean_squared_error",
        n_folds: int = 5,
    ) -> dict[str, Any]:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            model = model_factory(trial)
            scores = cross_val_score(model, X, y, cv=n_folds, scoring=scoring)
            if "neg" in scoring:
                return float(-scores.mean())
            return float(scores.mean())

        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.best_params = self.study.best_params
        return {"best_params": self.best_params, "best_value": self.study.best_value}

    def get_tuned_model(self, model_class: type = XGBRegressor) -> Any:
        if not self.best_params:
            raise ValueError("Run tuning first")
        return model_class(**self.best_params, random_state=self.random_state, verbosity=0)
