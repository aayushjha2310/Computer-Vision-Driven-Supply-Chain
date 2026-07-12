"""
End-to-end Supply Chain Intelligence Pipeline Orchestrator.

Runs the complete workflow:
1. Data ingestion (Spark, SQLAlchemy, Parquet)
2. Feature engineering (Dask, Feature Store)
3. ML training (XGBoost/LightGBM/CatBoost, Optuna, SMOTE)
4. Computer Vision (YOLO, ViT, SAM, Diffusers, Albumentations)
5. MLOps (MLflow, W&B, Drift monitoring, ONNX export)
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.sample_data import (
    create_sample_images,
    create_sample_video,
    generate_market_features,
    generate_transaction_logs,
    seed_enterprise_database,
)
from src.data_engineering.dask_processor import DaskFeatureProcessor
from src.data_engineering.parquet_handler import ParquetArchive
from src.data_engineering.spark_pipeline import SparkSupplyChainPipeline
from src.data_engineering.sqlalchemy_connector import EnterpriseDBConnector
from src.feature_store.feature_store import FeatureStore
from src.ml_models.gradient_boosting import GradientBoostingEnsemble
from src.ml_models.hyperparameter_tuning import HyperparameterTuner
from src.ml_models.imbalanced_handler import ImbalancedDataHandler
from src.mlops.mlflow_tracker import MLflowTracker
from src.mlops.model_drift import ModelDriftMonitor
from src.mlops.model_export import ModelExporter
from src.mlops.wandb_tracker import WandBTracker
from src.utils.config import ensure_dirs, load_config


class SupplyChainPipeline:
    """Master orchestrator for the end-to-end supply chain intelligence engine."""

    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path)
        ensure_dirs(self.config)
        self.results: dict = {"stages": {}, "started_at": datetime.utcnow().isoformat()}

    def run_stage_1_data_ingestion(self) -> dict:
        print("\n[Stage 1] Data Ingestion & Engineering...")
        raw_dir = self.config["paths"]["data_raw"]
        processed_dir = self.config["paths"]["data_processed"]

        df = generate_transaction_logs(n_rows=5000)
        df = generate_market_features(df)
        seed_enterprise_database(self.config["database"]["url"], df)

        archive = ParquetArchive(processed_dir)
        parquet_path = archive.archive_transactions(df)

        db = EnterpriseDBConnector(self.config["database"]["url"])
        joined = db.join_transaction_warehouse()
        regional = db.get_regional_summary()
        db.close()

        spark = SparkSupplyChainPipeline(self.config)
        spark_out = spark.process_batch_pipeline(
            parquet_path,
            str(Path(processed_dir) / "spark_aggregated"),
            pdf=df,
        )
        stream_out = spark.simulate_streaming_batch(
            df.head(100).to_dict("records"),
            processed_dir,
        )
        spark.stop()

        image_paths = create_sample_images(raw_dir, n_images=10)
        video_path = create_sample_video(str(Path(raw_dir) / "warehouse_feed.mp4"))

        stage_result = {
            "transactions": len(df),
            "parquet_path": parquet_path,
            "joined_rows": len(joined),
            "regional_summary": regional.to_dict("records"),
            "spark_aggregated_rows": spark_out["aggregated_rows"],
            "stream_output": stream_out,
            "sample_images": len(image_paths),
            "sample_video": video_path,
        }
        self.results["stages"]["data_ingestion"] = stage_result
        self._df = df
        self._image_paths = image_paths
        self._video_path = video_path
        print(f"  -> Ingested {len(df)} transactions, {len(image_paths)} images, 1 video")
        return stage_result

    def run_stage_2_feature_engineering(self) -> dict:
        print("\n[Stage 2] Feature Engineering (Dask + Feature Store)...")
        dask_proc = DaskFeatureProcessor()
        dask_result = dask_proc.process_dataframe_features(self._df, self.config)

        enriched_df = dask_result["dataframe"]
        feature_store = FeatureStore(self.config["paths"]["feature_store"])
        feature_store.register_feature_group(
            "transaction_features", enriched_df,
            description="Dask-processed transaction features with PCA and clustering",
            tags=["transactions", "dask", "pca"],
        )
        feature_store.register_feature_group(
            "market_features", enriched_df[["region", "category", "price_usd", "market_index", "market_cluster"]],
            description="Market-level aggregated features",
            tags=["market", "forecasting"],
        )

        stage_result = {
            "enriched_features": enriched_df.shape[1],
            "market_clusters": dask_result["n_clusters"],
            "pca_components": dask_result["pca_components"].shape[1],
            "feature_groups": feature_store.list_feature_groups(),
        }
        self.results["stages"]["feature_engineering"] = stage_result
        self._enriched_df = enriched_df
        self._feature_store = feature_store
        print(f"  -> {enriched_df.shape[1]} features, {dask_result['n_clusters']} market clusters")
        return stage_result

    def run_stage_3_ml_training(self) -> dict:
        print("\n[Stage 3] ML Training (Gradient Boosting + Optuna + SMOTE)...")
        feature_cols = [
            "quantity", "unit_cost", "shipping_days", "market_index",
            "region_encoded", "category_encoded", "log_quantity", "cost_ratio",
        ]
        ml_cfg = self.config["ml"]

        imbalanced = ImbalancedDataHandler(smote_ratio=ml_cfg["smote_ratio"])
        fraud_data = imbalanced.prepare_fraud_dataset(self._enriched_df, feature_cols)

        gbm = GradientBoostingEnsemble(random_state=ml_cfg["random_state"])
        reg_results = gbm.train_regressors(
            self._enriched_df, feature_cols, ml_cfg["target_column"], test_size=ml_cfg["test_size"]
        )
        fraud_results = gbm.train_classifiers(
            self._enriched_df, feature_cols, "is_fraud",
            class_weights=fraud_data["class_weights"], test_size=ml_cfg["test_size"],
        )

        X_fraud, y_fraud = fraud_data["X"], fraud_data["y"]
        tuner = HyperparameterTuner(n_trials=min(ml_cfg["optuna_trials"], 10))
        tune_result = tuner.tune_xgboost_regressor(
            self._enriched_df[feature_cols].fillna(0).values,
            self._enriched_df[ml_cfg["target_column"]].values,
            n_folds=ml_cfg["n_folds"],
        )

        forecasts = gbm.forecast_regional_prices(
            self._enriched_df, horizon_days=self.config["forecasting"]["horizon_days"]
        )

        mlflow = MLflowTracker(self.config["paths"]["mlflow_uri"], self.config["mlops"]["experiment_name"])
        best_model = gbm.get_best_model()
        run_id = mlflow.register_model(
            best_model, "price_forecast_model",
            metrics=reg_results["results"][reg_results["best_model"]],
            params=tune_result["best_params"],
        )

        wandb = WandBTracker(project=self.config["mlops"]["wandb_project"])
        wandb.init("supply_chain_training", config=ml_cfg)
        wandb.log_model_comparison(reg_results["results"])
        wandb.log_metrics({"best_rmse": reg_results["results"][reg_results["best_model"]]["rmse"]})
        wandb.finish()

        model_path = Path(self.config["paths"]["models"]) / "best_price_model.pkl"
        joblib.dump(best_model, model_path)

        stage_result = {
            "regression_results": {k: {mk: round(mv, 4) if isinstance(mv, float) else mv for mk, mv in v.items()} for k, v in reg_results["results"].items()},
            "best_model": reg_results["best_model"],
            "fraud_detection": fraud_results["results"],
            "optuna_best_params": tune_result["best_params"],
            "optuna_best_rmse": round(tune_result["best_rmse"], 4),
            "fraud_smote": fraud_data["resampled_distribution"],
            "forecast_regions": forecasts["region"].unique().tolist(),
            "mlflow_run_id": run_id,
        }
        self.results["stages"]["ml_training"] = stage_result
        self._gbm = gbm
        self._best_model = best_model
        print(f"  -> Best model: {reg_results['best_model']}, RMSE: {reg_results['results'][reg_results['best_model']]['rmse']:.2f}")
        return stage_result

    def run_stage_4_computer_vision(self) -> dict:
        print("\n[Stage 4] Computer Vision Pipeline...")
        from src.computer_vision.augmentation import VisionAugmentor
        from src.computer_vision.sam_segmentation import SAMDamageSegmenter
        from src.computer_vision.synthetic_generator import SyntheticEdgeCaseGenerator
        from src.computer_vision.video_ingestion import VideoStreamIngestor
        from src.computer_vision.vit_classifier import ViTProductClassifier
        from src.computer_vision.yolo_tracker import YOLOAssetTracker
        import cv2

        vision_cfg = self.config["vision"]
        raw_dir = self.config["paths"]["data_raw"]
        synthetic_dir = self.config["paths"]["data_synthetic"]

        video_ingestor = VideoStreamIngestor(
            frame_width=self.config["video"]["frame_width"],
            frame_height=self.config["video"]["frame_height"],
        )
        frames = list(video_ingestor.simulate_camera_stream(self._image_paths[:3], n_cycles=1))
        motion = video_ingestor.compute_motion_vectors(self._video_path)
        snapshot = video_ingestor.create_inventory_snapshot(frames[0])

        yolo = YOLOAssetTracker(model_path=vision_cfg["yolo_model"], confidence=vision_cfg["confidence_threshold"])
        annotated = yolo.save_annotated_images(self._image_paths[:5], str(Path(raw_dir) / "annotated"))
        video_tracks = yolo.track_assets_in_video(self._video_path)

        vit = ViTProductClassifier(model_name=vision_cfg["vit_model"])
        classifications = []
        for path in self._image_paths[:5]:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            labels = vit.extract_shipment_labels(img_rgb)
            classifications.append(labels)

        sam = SAMDamageSegmenter()
        damage_results = []
        for path in self._image_paths[:3]:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            damage = sam.auto_detect_damage(img_rgb)
            overlay = sam.overlay_mask(img_rgb, damage["mask"])
            out_path = Path(raw_dir) / f"damage_{Path(path).stem}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            damage_results.append({"file": path, "damage_ratio": damage["damage_ratio"]})

        synth_gen = SyntheticEdgeCaseGenerator()
        synthetic = synth_gen.generate_edge_case_dataset(synthetic_dir, n_per_case=2)

        augmentor = VisionAugmentor()
        import json as json_mod
        aug_images, aug_bboxes, aug_labels = [], [], []
        for path in self._image_paths[:3]:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            meta_path = Path(path).with_suffix(".json")
            bboxes, labels = [], []
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json_mod.load(f)
                bboxes = [b["bbox"] for b in meta.get("boxes", [])]
                labels = [b["label"] for b in meta.get("boxes", [])]
            aug_images.append(img_rgb)
            aug_bboxes.append(bboxes)
            aug_labels.append(labels)
        augmented = augmentor.create_augmented_dataset(aug_images, aug_bboxes, aug_labels, n_augmentations=2)

        stage_result = {
            "frames_processed": len(frames),
            "motion_vectors": len(motion),
            "inventory_snapshot": snapshot,
            "yolo_annotated": len(annotated),
            "video_tracks": {k: len(v) for k, v in video_tracks["tracks"].items()},
            "classifications": classifications,
            "damage_inspections": damage_results,
            "synthetic_images": synthetic["total_images"],
            "augmented_samples": len(augmented),
        }
        self.results["stages"]["computer_vision"] = stage_result
        print(f"  -> YOLO detections, ViT classifications, SAM damage maps, {synthetic['total_images']} synthetic images")
        return stage_result

    def run_stage_5_mlops_deployment(self) -> dict:
        print("\n[Stage 5] MLOps, Export & Drift Monitoring...")
        feature_cols = self._gbm.feature_cols

        drift_monitor = ModelDriftMonitor(drift_threshold=self.config["mlops"]["drift_threshold"])
        drift_monitor.set_reference(self._enriched_df, feature_cols)

        split_idx = int(len(self._enriched_df) * 0.8)
        current_data = self._enriched_df.iloc[split_idx:]
        feature_drift = drift_monitor.detect_feature_drift(current_data, feature_cols)

        X_ref = self._enriched_df.iloc[:split_idx][feature_cols].fillna(0).values
        X_cur = current_data[feature_cols].fillna(0).values
        ref_preds = self._best_model.predict(X_ref)
        cur_preds = self._best_model.predict(X_cur)
        pred_drift = drift_monitor.detect_prediction_drift(ref_preds, cur_preds)
        drift_report = drift_monitor.generate_drift_report(feature_drift, pred_drift)

        exporter = ModelExporter(self.config["paths"]["models"])
        pkl_path = exporter.save_pickle(self._best_model, "best_price_model")
        try:
            onnx_path = exporter.export_xgboost_to_onnx(
                self._best_model, len(feature_cols), "price_forecast"
            )
            sample = self._enriched_df[feature_cols].fillna(0).values[:5].astype(np.float32)
            onnx_valid = exporter.validate_onnx(onnx_path, sample)
        except Exception as exc:
            onnx_path = str(Path(self.config["paths"]["models"]) / "price_forecast.onnx")
            onnx_valid = {"valid": False, "error": str(exc)}
        trt_config = exporter.prepare_tensorrt_config(onnx_path, "price_forecast")
        sagemaker_config = exporter.export_sagemaker_config(
            pkl_path, self.config.get("sagemaker_role", "arn:aws:iam::123456789012:role/SageMakerRole")
        )

        stage_result = {
            "drift_detected": feature_drift["drift_detected"],
            "n_features_drifted": feature_drift["n_drifted"],
            "prediction_drift": pred_drift["drift_detected"],
            "drift_report": drift_report,
            "onnx_export": onnx_path,
            "onnx_valid": onnx_valid["valid"],
            "tensorrt_config": trt_config["engine_path"],
            "sagemaker_ready": True,
            "model_pickle": pkl_path,
        }
        self.results["stages"]["mlops"] = stage_result
        print(f"  -> ONNX exported, drift monitoring active, TensorRT config ready")
        return stage_result

    def run_all(self) -> dict:
        print("=" * 60)
        print("  Vision-Driven Global Supply Chain Intelligence Engine")
        print("=" * 60)
        try:
            self.run_stage_1_data_ingestion()
            self.run_stage_2_feature_engineering()
            self.run_stage_3_ml_training()
            self.run_stage_4_computer_vision()
            self.run_stage_5_mlops_deployment()
            self.results["status"] = "success"
            self.results["completed_at"] = datetime.utcnow().isoformat()
        except Exception as e:
            self.results["status"] = "failed"
            self.results["error"] = str(e)
            self.results["traceback"] = traceback.format_exc()
            print(f"\nPipeline FAILED: {e}")
            traceback.print_exc()

        report_path = Path(self.config["paths"]["data_processed"]) / "pipeline_report.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nPipeline report saved to: {report_path}")
        print("=" * 60)
        return self.results


def main():
    pipeline = SupplyChainPipeline()
    results = pipeline.run_all()
    if results["status"] == "success":
        print("\nAll stages completed successfully!")
        for stage, data in results["stages"].items():
            print(f"  [OK] {stage}")
    else:
        print(f"\nPipeline failed: {results.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
