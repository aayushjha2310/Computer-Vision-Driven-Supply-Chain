"""Apache Spark streaming and batch pipeline for transactional logs."""

import json
import os
import platform
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


class SparkSupplyChainPipeline:
    """Distributed real-time transactional log processing with Apache Spark."""

    SCHEMA = StructType(
        [
            StructField("transaction_id", StringType(), False),
            StructField("timestamp", StringType(), False),
            StructField("region", StringType(), False),
            StructField("category", StringType(), False),
            StructField("quantity", IntegerType(), False),
            StructField("unit_cost", DoubleType(), False),
            StructField("price_usd", DoubleType(), False),
            StructField("warehouse_id", StringType(), False),
            StructField("supplier_id", StringType(), False),
            StructField("is_fraud", IntegerType(), False),
            StructField("is_defect", IntegerType(), False),
            StructField("shipping_days", IntegerType(), False),
            StructField("market_index", DoubleType(), False),
        ]
    )

    def __init__(self, config: dict[str, Any]):
        import sys
        spark_cfg = config["spark"]
        python_path = sys.executable
        os.environ["PYSPARK_PYTHON"] = python_path
        os.environ["PYSPARK_DRIVER_PYTHON"] = python_path
        self.checkpoint_dir = str(Path(config["paths"]["data_processed"]) / "spark_checkpoints")
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        builder = (
            SparkSession.builder.appName(spark_cfg["app_name"])
            .master(spark_cfg["master"])
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.python.profile", "false")
            .config("spark.executorEnv.PYSPARK_PYTHON", python_path)
            .config("spark.pyspark.python", python_path)
            .config("spark.pyspark.driver.python", python_path)
        )
        if platform.system() == "Windows":
            builder = builder.config("spark.hadoop.io.native.lib.available", "false")
        self.spark = builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        self.use_pandas_fallback = platform.system() == "Windows"

    def ingest_json_stream(self, input_dir: str) -> Any:
        return (
            self.spark.readStream.schema(self.SCHEMA)
            .option("maxFilesPerTrigger", 1)
            .json(input_dir)
        )

    def ingest_parquet_batch(self, input_path: str) -> Any:
        try:
            return self.spark.read.parquet(input_path)
        except Exception:
            return self.ingest_from_pandas_parquet(input_path)

    def ingest_from_pandas(self, df: pd.DataFrame, columns: list[str] | None = None) -> Any:
        pdf = df.copy()
        if columns:
            available = [c for c in columns if c in pdf.columns]
            pdf = pdf[available]
        for col in pdf.columns:
            if pd.api.types.is_datetime64_any_dtype(pdf[col]):
                pdf[col] = pdf[col].astype(str)
            elif pd.api.types.is_object_dtype(pdf[col]) or pd.api.types.is_string_dtype(pdf[col]):
                pdf[col] = pdf[col].map(lambda x: str(x) if x is not None else None)
            elif pd.api.types.is_integer_dtype(pdf[col]):
                pdf[col] = pdf[col].astype(int)
            elif pd.api.types.is_float_dtype(pdf[col]):
                pdf[col] = pdf[col].astype(float)
        records = pdf.to_dict(orient="records")
        return self.spark.createDataFrame(records)

    def ingest_from_pandas_parquet(self, input_path: str) -> Any:
        path = Path(input_path)
        if path.is_dir():
            parquet_files = list(path.rglob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files in {input_path}")
            df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        else:
            df = pd.read_parquet(input_path)
        return self.ingest_from_pandas(df)

    def aggregate_regional_metrics(self, df: Any) -> Any:
        return (
            df.withColumn("timestamp", F.to_timestamp("timestamp"))
            .groupBy("region", "category")
            .agg(
                F.count("transaction_id").alias("txn_count"),
                F.sum("price_usd").alias("total_revenue"),
                F.avg("price_usd").alias("avg_price"),
                F.sum("is_fraud").alias("fraud_count"),
                F.sum("is_defect").alias("defect_count"),
                F.avg("market_index").alias("avg_market_index"),
            )
        )

    def detect_anomalies(self, df: Any, price_threshold: float = 3.0) -> Any:
        stats = df.agg(
            F.mean("price_usd").alias("mean_price"),
            F.stddev("price_usd").alias("std_price"),
        )
        return (
            df.crossJoin(stats)
            .withColumn(
                "is_price_anomaly",
                F.when(
                    F.abs(F.col("price_usd") - F.col("mean_price"))
                    > F.lit(price_threshold) * F.coalesce(F.col("std_price"), F.lit(1.0)),
                    F.lit(1),
                ).otherwise(F.lit(0)),
            )
            .drop("mean_price", "std_price")
        )

    def write_parquet(self, df: Any, output_path: str, mode: str = "overwrite") -> None:
        try:
            df.write.mode(mode).parquet(output_path)
        except Exception:
            out = Path(output_path)
            out.mkdir(parents=True, exist_ok=True)
            pdf = df.toPandas()
            pdf.to_parquet(out / "data.parquet", index=False)

    def process_batch_pipeline(self, input_parquet: str, output_parquet: str, pdf: pd.DataFrame | None = None) -> dict[str, Any]:
        schema_cols = [f.name for f in self.SCHEMA.fields]
        if self.use_pandas_fallback and pdf is not None:
            return self._pandas_batch_pipeline(pdf, output_parquet, schema_cols)
        try:
            if pdf is not None:
                if "timestamp" in pdf.columns and pd.api.types.is_datetime64_any_dtype(pdf["timestamp"]):
                    pdf = pdf.copy()
                    pdf["timestamp"] = pdf["timestamp"].astype(str)
                df = self.ingest_from_pandas(pdf, columns=schema_cols)
            else:
                df = self.ingest_parquet_batch(input_parquet)
            df = self.detect_anomalies(df)
            aggregated = self.aggregate_regional_metrics(df)
            self.write_parquet(aggregated, output_parquet)
            row_count = aggregated.count()
            return {"status": "success", "aggregated_rows": row_count, "output": output_parquet, "engine": "spark"}
        except Exception as exc:
            if pdf is not None:
                result = self._pandas_batch_pipeline(pdf, output_parquet, schema_cols)
                result["spark_fallback_reason"] = str(exc)
                return result
            raise

    def _pandas_batch_pipeline(self, pdf: pd.DataFrame, output_parquet: str, schema_cols: list[str]) -> dict[str, Any]:
        df = pdf[[c for c in schema_cols if c in pdf.columns]].copy()
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype(str)
        mean_p = df["price_usd"].mean()
        std_p = df["price_usd"].std() or 1.0
        df["is_price_anomaly"] = (df["price_usd"] - mean_p).abs() > 3.0 * std_p
        df["is_price_anomaly"] = df["is_price_anomaly"].astype(int)
        aggregated = (
            df.groupby(["region", "category"], as_index=False)
            .agg(
                txn_count=("transaction_id", "count"),
                total_revenue=("price_usd", "sum"),
                avg_price=("price_usd", "mean"),
                fraud_count=("is_fraud", "sum"),
                defect_count=("is_defect", "sum"),
                avg_market_index=("market_index", "mean"),
            )
        )
        out = Path(output_parquet)
        out.mkdir(parents=True, exist_ok=True)
        aggregated.to_parquet(out / "data.parquet", index=False)
        return {
            "status": "success",
            "aggregated_rows": len(aggregated),
            "output": str(out),
            "engine": "pandas_fallback",
        }

    def simulate_streaming_batch(self, json_records: list[dict], output_dir: str) -> str:
        if self.use_pandas_fallback:
            return self._pandas_streaming_batch(json_records, output_dir)
        stream_dir = Path(output_dir) / "stream_input"
        stream_dir.mkdir(parents=True, exist_ok=True)
        batch_file = stream_dir / "batch_001.json"
        with open(batch_file, "w", encoding="utf-8") as f:
            for record in json_records:
                f.write(json.dumps(record, default=str) + "\n")
        try:
            df = self.spark.read.schema(self.SCHEMA).json(str(stream_dir))
            df = self.detect_anomalies(df)
            agg = self.aggregate_regional_metrics(df)
            out = str(Path(output_dir) / "stream_aggregated.parquet")
            self.write_parquet(agg, out)
            return out
        except Exception:
            return self._pandas_streaming_batch(json_records, output_dir)

    def _pandas_streaming_batch(self, json_records: list[dict], output_dir: str) -> str:
        pdf = pd.DataFrame(json_records)
        result = self._pandas_batch_pipeline(
            pdf, str(Path(output_dir) / "stream_aggregated"), [f.name for f in self.SCHEMA.fields]
        )
        return result["output"]

    def stop(self) -> None:
        self.spark.stop()
