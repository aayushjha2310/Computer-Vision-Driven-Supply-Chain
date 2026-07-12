"""ONNX model serialization and TensorRT export utilities."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np


class ModelExporter:
    """Universal model serialization to ONNX and optimization for edge deployment."""

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def export_sklearn_to_onnx(self, model: Any, feature_count: int, name: str) -> str:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, feature_count]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        output_path = self.models_dir / f"{name}.onnx"
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        return str(output_path)

    def export_xgboost_to_onnx(self, model: Any, feature_count: int, name: str) -> str:
        try:
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType

            initial_type = [("float_input", FloatTensorType([None, feature_count]))]
            onnx_model = convert_xgboost(model, initial_types=initial_type)
            output_path = self.models_dir / f"{name}.onnx"
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            return str(output_path)
        except Exception:
            return self.export_sklearn_to_onnx(model, feature_count, name)

    def validate_onnx(self, onnx_path: str, sample_input: np.ndarray) -> dict[str, Any]:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: sample_input.astype(np.float32)})
        return {
            "valid": True,
            "input_shape": list(sample_input.shape),
            "output_shape": [list(o.shape) for o in output],
        }

    def save_pickle(self, model: Any, name: str) -> str:
        path = self.models_dir / f"{name}.pkl"
        joblib.dump(model, path)
        return str(path)

    def load_pickle(self, name: str) -> Any:
        return joblib.load(self.models_dir / f"{name}.pkl")

    def prepare_tensorrt_config(self, onnx_path: str, name: str) -> dict[str, Any]:
        return {
            "onnx_path": onnx_path,
            "engine_path": str(self.models_dir / f"{name}.trt"),
            "precision": "FP16",
            "max_batch_size": 32,
            "workspace_size_mb": 1024,
            "instructions": [
                f"trtexec --onnx={onnx_path} --saveEngine={self.models_dir / f'{name}.trt'} --fp16",
            ],
        }

    def export_sagemaker_config(self, model_path: str, role: str, instance_type: str = "ml.m5.xlarge") -> dict[str, Any]:
        return {
            "ModelName": "supply-chain-price-forecast",
            "PrimaryContainer": {
                "Image": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
                "ModelDataUrl": model_path,
            },
            "ExecutionRoleArn": role,
            "TrainingJob": {
                "InstanceType": instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            },
        }
