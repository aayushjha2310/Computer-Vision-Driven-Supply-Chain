# Vision-Driven Global Supply Chain Intelligent Logistics

Enterprise intelligence engine for global manufacturing/e-commerce conglomerates. Processes streaming telemetry, warehouse video feeds, and market pricing to prevent fraud, manage inventory via machine vision, and forecast multi-million dollar pricing strategies.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SUPPLY CHAIN INTELLIGENCE ENGINE                  │
├──────────────┬──────────────┬──────────────┬───────────────────────┤
│ Data Eng.    │ Feature Store│ ML Models    │ Computer Vision       │
│ Spark/Dask   │ Unified Repo │ XGB/LGBM/CB  │ YOLO/ViT/SAM/Diffusers│
│ SQL/Parquet  │              │ Optuna/SMOTE │ Albumentations        │
├──────────────┴──────────────┴──────────────┴───────────────────────┤
│ MLOps: MLflow | W&B | Drift Monitor | ONNX | TensorRT | SageMaker  │
├─────────────────────────────────────────────────────────────────────┤
│ Serving: Gradio UI | FastAPI | Docker | Kubernetes | GitHub Actions│
└─────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Domain | Technologies |
|--------|-------------|
| Data Engineering | Apache Spark, Dask, NumPy, SQLAlchemy, Parquet, Feature Store |
| Classical ML | Scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SMOTE, K-Fold CV |
| Computer Vision | OpenCV, YOLO, Hugging Face ViT, SAM, Diffusers, Albumentations |
| MLOps | Docker, Kubernetes, MLflow, Weights & Biases, GitHub Actions |
| Deployment | ONNX, TensorRT, AWS SageMaker, Gradio, FastAPI, Model Drift |

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline

```bash
python run_pipeline.py
```

This executes all 5 stages:
- **Stage 1**: Data ingestion (Spark, SQLAlchemy, Parquet, sample video/images)
- **Stage 2**: Feature engineering (Dask parallel processing, Feature Store)
- **Stage 3**: ML training (XGBoost/LightGBM/CatBoost, Optuna tuning, SMOTE)
- **Stage 4**: Computer vision (YOLO tracking, ViT classification, SAM damage inspection, synthetic data)
- **Stage 5**: MLOps (MLflow tracking, drift monitoring, ONNX export)

### 3. Launch Services

```bash
# Gradio QA UI (port 7860)
python src/serving/gradio_app.py

# FastAPI inference (port 8000)
uvicorn src.serving.inference_api:app --host 0.0.0.0 --port 8000
```

### 4. Docker Deployment

```bash
docker-compose up --build
```

Services: pipeline, API (8000), Gradio (7860), MLflow (5000)

### 5. Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
```

## Project Structure

```
Project-4/
├── config/settings.yaml          # Central configuration
├── src/
│   ├── data_engineering/         # Spark, Dask, SQLAlchemy, Parquet
│   ├── feature_store/            # Unified feature repository
│   ├── ml_models/                # GBM ensemble, Optuna, SMOTE
│   ├── computer_vision/          # YOLO, ViT, SAM, Diffusers
│   ├── mlops/                    # MLflow, W&B, drift, ONNX
│   └── serving/                  # Gradio UI, FastAPI
├── run_pipeline.py               # End-to-end orchestrator
├── docker-compose.yml
├── k8s/deployment.yaml
├── .github/workflows/ci-cd.yml
└── tests/
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/predict/price` | Price forecasting |
| POST | `/analyze/image` | Warehouse image analysis |
| GET | `/metrics/summary` | Service metrics |

## Configuration

Edit `config/settings.yaml` for Spark, ML, vision model, and serving parameters. Copy `.env.example` to `.env` for W&B and AWS credentials.

## Testing

```bash
python -m pytest tests/ -v
```
