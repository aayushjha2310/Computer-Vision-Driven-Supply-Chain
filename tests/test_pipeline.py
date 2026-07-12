"""Unit tests for the Supply Chain Intelligence Platform."""

import numpy as np
import pandas as pd
import pytest

from src.data.sample_data import generate_market_features, generate_transaction_logs
from src.data_engineering.parquet_handler import ParquetArchive
from src.feature_store.feature_store import FeatureStore
from src.ml_models.classical_ml import ClassicalMLPipeline
from src.ml_models.imbalanced_handler import ImbalancedDataHandler
from src.mlops.model_drift import ModelDriftMonitor
from src.utils.config import load_config, ensure_dirs


@pytest.fixture
def sample_df():
    df = generate_transaction_logs(200)
    return generate_market_features(df)


@pytest.fixture
def config():
    cfg = load_config()
    ensure_dirs(cfg)
    return cfg


class TestConfig:
    def test_load_config(self, config):
        assert "project" in config
        assert "spark" in config
        assert "ml" in config

    def test_paths_resolved(self, config):
        assert config["paths"]["data_raw"].endswith("data\\raw") or config["paths"]["data_raw"].endswith("data/raw")


class TestSampleData:
    def test_transaction_logs(self):
        df = generate_transaction_logs(100)
        assert len(df) == 100
        assert "price_usd" in df.columns
        assert "is_fraud" in df.columns

    def test_market_features(self, sample_df):
        assert "region_encoded" in sample_df.columns
        assert "log_quantity" in sample_df.columns


class TestParquet:
    def test_write_read(self, sample_df, config, tmp_path):
        archive = ParquetArchive(str(tmp_path))
        archive.write(sample_df, "test")
        loaded = archive.read("test")
        assert len(loaded) == len(sample_df)


class TestFeatureStore:
    def test_register_and_get(self, sample_df, config, tmp_path):
        store = FeatureStore(str(tmp_path))
        store.register_feature_group("test_group", sample_df)
        loaded = store.get_feature_group("test_group")
        assert len(loaded) == len(sample_df)
        assert "test_group" in store.list_feature_groups()


class TestClassicalML:
    def test_regression_metrics(self):
        ml = ClassicalMLPipeline()
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        metrics = ml.evaluate_regression(y_true, y_pred)
        assert "rmse" in metrics
        assert metrics["r2"] > 0.9

    def test_clustering(self, sample_df):
        ml = ClassicalMLPipeline()
        cols = ["quantity", "unit_cost", "price_usd"]
        result = ml.market_clustering(sample_df, cols, n_clusters=3)
        assert "market_segment" in result.columns


class TestImbalancedHandler:
    def test_class_weights(self, sample_df):
        handler = ImbalancedDataHandler()
        data = handler.prepare_fraud_dataset(
            sample_df, ["quantity", "unit_cost", "shipping_days"], use_smote=False
        )
        assert "class_weights" in data
        assert data["X"].shape[0] == len(sample_df)


class TestDriftMonitor:
    def test_feature_drift(self, sample_df):
        monitor = ModelDriftMonitor(drift_threshold=0.15)
        cols = ["quantity", "price_usd", "market_index"]
        monitor.set_reference(sample_df.iloc[:100], cols)
        drift = monitor.detect_feature_drift(sample_df.iloc[100:], cols)
        assert "drift_detected" in drift
        assert "features" in drift
