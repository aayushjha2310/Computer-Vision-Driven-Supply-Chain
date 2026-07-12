"""Sample data generation for end-to-end pipeline testing."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def generate_transaction_logs(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = ["NA", "EU", "APAC", "LATAM"]
    categories = ["electronics", "apparel", "food", "industrial", "pharma"]
    start = datetime(2024, 1, 1)

    records = []
    for i in range(n_rows):
        ts = start + timedelta(hours=int(rng.integers(0, 8760)))
        region = rng.choice(regions)
        category = rng.choice(categories)
        base_price = {"electronics": 450, "apparel": 80, "food": 25, "industrial": 1200, "pharma": 350}[
            category
        ]
        price = base_price * rng.uniform(0.7, 1.4) * (1 + 0.1 * regions.index(region))
        qty = int(rng.integers(1, 500))
        is_fraud = int(rng.random() < 0.02)
        is_defect = int(rng.random() < 0.03)
        records.append(
            {
                "transaction_id": f"TXN-{i:06d}",
                "timestamp": ts.isoformat(),
                "region": region,
                "category": category,
                "quantity": qty,
                "unit_cost": round(float(rng.uniform(5, 200)), 2),
                "price_usd": round(price * qty, 2),
                "warehouse_id": f"WH-{rng.integers(1, 12)}",
                "supplier_id": f"SUP-{rng.integers(100, 999)}",
                "is_fraud": is_fraud,
                "is_defect": is_defect,
                "shipping_days": int(rng.integers(1, 21)),
                "market_index": round(float(rng.uniform(0.8, 1.2)), 4),
            }
        )
    return pd.DataFrame(records)


def generate_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["region_encoded"] = df["region"].astype("category").cat.codes
    df["category_encoded"] = df["category"].astype("category").cat.codes
    df["price_per_unit"] = df["price_usd"] / df["quantity"].clip(lower=1)
    df["cost_ratio"] = df["unit_cost"] / df["price_per_unit"].clip(lower=0.01)
    df["log_quantity"] = np.log1p(df["quantity"])
    return df


def create_sample_images(output_dir: str, n_images: int = 20, seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = [(200, 80, 60), (60, 120, 200), (80, 180, 80), (180, 140, 60), (140, 60, 180)]
    labels = ["pallet", "box", "crate", "drum", "container"]

    for i in range(n_images):
        img = Image.new("RGB", (640, 480), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        n_objects = int(rng.integers(1, 5))
        boxes = []
        for _ in range(n_objects):
            x1, y1 = int(rng.integers(50, 400)), int(rng.integers(50, 300))
            w, h = int(rng.integers(60, 180)), int(rng.integers(60, 150))
            color = colors[int(rng.integers(0, len(colors)))]
            draw.rectangle([x1, y1, x1 + w, y1 + h], fill=color, outline=(0, 0, 0), width=2)
            label_idx = int(rng.integers(0, len(labels)))
            boxes.append({"label": labels[label_idx], "bbox": [x1, y1, x1 + w, y1 + h]})
        path = out / f"warehouse_frame_{i:04d}.jpg"
        img.save(path, quality=90)
        meta_path = out / f"warehouse_frame_{i:04d}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"boxes": boxes, "camera": f"bay_{i % 3 + 1}"}, f)
        paths.append(str(path))
    return paths


def create_sample_video(output_path: str, n_frames: int = 30, seed: int = 42) -> str:
    import cv2

    rng = np.random.default_rng(seed)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 10.0, (640, 480))

    x, y = 100, 200
    for frame_idx in range(n_frames):
        frame = np.full((480, 640, 3), 230, dtype=np.uint8)
        x = min(500, x + int(rng.integers(5, 15)))
        cv2.rectangle(frame, (x, y), (x + 80, y + 60), (60, 120, 200), -1)
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        writer.write(frame)
    writer.release()
    return str(out)


def seed_enterprise_database(db_url: str, df: pd.DataFrame) -> None:
    db_path = db_url.replace("sqlite:///", "")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS warehouses (
            warehouse_id TEXT PRIMARY KEY,
            region TEXT,
            capacity INTEGER,
            lat REAL,
            lon REAL
        )
        """
    )
    warehouses = pd.DataFrame(
        {
            "warehouse_id": [f"WH-{i}" for i in range(1, 12)],
            "region": np.random.choice(["NA", "EU", "APAC", "LATAM"], 11),
            "capacity": np.random.randint(5000, 50000, 11),
            "lat": np.random.uniform(-60, 60, 11),
            "lon": np.random.uniform(-150, 150, 11),
        }
    )
    warehouses.to_sql("warehouses", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
