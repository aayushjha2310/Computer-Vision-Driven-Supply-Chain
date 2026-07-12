"""FastAPI inference microservice for supply chain predictions."""

from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from src.computer_vision.yolo_tracker import YOLOAssetTracker
from src.computer_vision.vit_classifier import ViTProductClassifier
from src.utils.config import load_config

app = FastAPI(title="Supply Chain Intelligence API", version="1.0.0")
config = load_config()
yolo_tracker = YOLOAssetTracker()
vit_classifier = ViTProductClassifier()
price_model = None


class PredictionRequest(BaseModel):
    quantity: float
    unit_cost: float
    shipping_days: float
    market_index: float
    region_encoded: float = 0
    category_encoded: float = 0
    log_quantity: float = 0


class PredictionResponse(BaseModel):
    predicted_price: float
    model_used: str


@app.on_event("startup")
async def load_models():
    global price_model
    model_path = Path(config["paths"]["models"]) / "best_price_model.pkl"
    if model_path.exists():
        price_model = joblib.load(model_path)


@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": price_model is not None}


@app.post("/predict/price", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    if price_model is None:
        base = request.quantity * request.unit_cost * request.market_index
        return PredictionResponse(predicted_price=round(base, 2), model_used="heuristic")
    features = np.array([[
        request.quantity, request.unit_cost, request.shipping_days,
        request.market_index, request.region_encoded, request.category_encoded, request.log_quantity,
    ]])
    pred = float(price_model.predict(features)[0])
    return PredictionResponse(predicted_price=round(pred, 2), model_used="xgboost")


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = yolo_tracker.detect(image_rgb)
    labels = vit_classifier.extract_shipment_labels(image_rgb)
    return {
        "detections": detections,
        "shipment_labels": labels,
        "object_count": len(detections),
    }


@app.get("/metrics/summary")
async def metrics_summary():
    return {
        "service": "supply-chain-intelligence",
        "endpoints": ["/health", "/predict/price", "/analyze/image", "/metrics/summary"],
        "vision_models": ["yolo", "vit"],
    }
