"""Parquet efficient columnar file archiving."""

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetArchive:
    """Efficient columnar Parquet read/write with partitioning support."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write(self, df: pd.DataFrame, name: str, partition_cols: list[str] | None = None) -> str:
        path = self.base_path / name
        path.mkdir(parents=True, exist_ok=True)
        if partition_cols:
            df.to_parquet(path, engine="pyarrow", partition_cols=partition_cols, index=False)
        else:
            df.to_parquet(path / "data.parquet", engine="pyarrow", index=False)
        return str(path)

    def read(self, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        path = self.base_path / name
        if (path / "data.parquet").exists():
            return pd.read_parquet(path / "data.parquet", columns=columns)
        return pd.read_parquet(path, columns=columns)

    def read_with_filter(self, name: str, filters: list[tuple] | None = None) -> pd.DataFrame:
        path = self.base_path / name
        return pd.read_parquet(path, filters=filters)

    def get_schema(self, name: str) -> dict[str, Any]:
        path = self.base_path / name
        parquet_file = path / "data.parquet" if (path / "data.parquet").exists() else next(path.glob("*.parquet"))
        schema = pq.read_schema(parquet_file)
        return {field.name: str(field.type) for field in schema}

    def merge_partitions(self, name: str) -> pd.DataFrame:
        path = self.base_path / name
        dataset = pq.ParquetDataset(path)
        table = dataset.read()
        return table.to_pandas()

    def write_arrow_table(self, table: pa.Table, name: str) -> str:
        path = self.base_path / name / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="snappy")
        return str(path)

    def archive_transactions(self, df: pd.DataFrame, partition_by_region: bool = True) -> str:
        if partition_by_region and "region" in df.columns:
            return self.write(df, "transactions", partition_cols=["region"])
        return self.write(df, "transactions")
