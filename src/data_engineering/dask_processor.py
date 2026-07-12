"""Dask parallel array processing for sprawling feature matrices."""

from typing import Any

import dask.array as da
import numpy as np
import pandas as pd


class DaskFeatureProcessor:
    """Parallel Python processing of large numerical arrays with Dask."""

    def __init__(self, n_workers: int = 4, threads_per_worker: int = 2):
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker

    def create_feature_matrix(self, n_rows: int, n_cols: int, chunks: tuple = (1000, 50)) -> da.Array:
        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_rows, n_cols))
        return da.from_array(data, chunks=chunks)

    def compute_pca_components(self, matrix: da.Array, n_components: int = 10) -> np.ndarray:
        computed = matrix.compute()
        centered = computed - computed.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        return vh[:n_components].T

    def parallel_feature_engineering(self, df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
        arrays = {col: da.from_array(df[col].values, chunks=1000) for col in numeric_cols}
        result = df.copy()
        for col in numeric_cols:
            arr = arrays[col]
            result[f"{col}_zscore"] = ((arr - arr.mean()) / (arr.std() + 1e-8)).compute()
            result[f"{col}_log"] = da.log1p(da.abs(arr)).compute() * np.sign(df[col].values)
        return result

    def kmeans_clustering_dask(self, matrix: da.Array, n_clusters: int = 5, max_iter: int = 50) -> np.ndarray:
        n_samples = matrix.shape[0]
        rng = np.random.default_rng(42)
        indices = rng.choice(n_samples, size=min(n_clusters, n_samples), replace=False)
        centroids = matrix[indices].compute()

        labels = np.zeros(n_samples, dtype=int)
        for _ in range(max_iter):
            computed_matrix = matrix.compute()
            distances = np.linalg.norm(computed_matrix[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    centroids[k] = computed_matrix[mask].mean(axis=0)
        return labels

    def process_dataframe_features(self, df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        numeric_cols = ["quantity", "unit_cost", "price_usd", "shipping_days", "market_index"]
        available = [c for c in numeric_cols if c in df.columns]
        enriched = self.parallel_feature_engineering(df, available)
        matrix = da.from_array(enriched[available].values, chunks=(1000, len(available)))
        labels = self.kmeans_clustering_dask(matrix, n_clusters=4)
        enriched["market_cluster"] = labels
        pca_components = self.compute_pca_components(matrix, n_components=min(3, len(available)))
        for i in range(pca_components.shape[1]):
            enriched[f"pca_{i}"] = enriched[available].values @ pca_components[:, i]
        return {"dataframe": enriched, "pca_components": pca_components, "n_clusters": 4}
