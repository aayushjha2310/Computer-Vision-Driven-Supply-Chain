"""Weights & Biases training and validation tracking."""

import os
from typing import Any


class WandBTracker:
    """Track loss trajectories and visual validation outputs during training."""

    def __init__(self, project: str, entity: str | None = None):
        self.project = project
        self.entity = entity
        self.run = None
        self.enabled = bool(os.getenv("WANDB_API_KEY"))

    def init(self, run_name: str, config: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        try:
            import wandb
            os.environ.setdefault("WANDB_MODE", "offline")
            self.run = wandb.init(project=self.project, entity=self.entity, name=run_name, config=config)
        except Exception:
            self.enabled = False

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled or self.run is None:
            return
        import wandb
        wandb.log(metrics, step=step)

    def log_image(self, key: str, image_path: str, caption: str = "") -> None:
        if not self.enabled or self.run is None:
            return
        try:
            import wandb
            wandb.log({key: wandb.Image(image_path, caption=caption)})
        except Exception:
            pass

    def log_model_comparison(self, results: dict[str, dict[str, float]]) -> None:
        if not self.enabled or self.run is None:
            return
        import wandb
        table = wandb.Table(columns=["model", "metric", "value"])
        for model_name, metrics in results.items():
            for metric_name, value in metrics.items():
                table.add_data(model_name, metric_name, value)
        wandb.log({"model_comparison": table})

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            import wandb
            wandb.finish()
