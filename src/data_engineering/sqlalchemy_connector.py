"""SQLAlchemy connector for enterprise transactional relational databases."""

from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class EnterpriseDBConnector:
    """Interface with historical enterprise transactional relational databases."""

    def __init__(self, db_url: str, echo: bool = False):
        self.db_url = db_url
        self.engine: Engine = create_engine(db_url, echo=echo)

    def read_transactions(self, limit: int | None = None) -> pd.DataFrame:
        query = "SELECT * FROM transactions"
        if limit:
            query += f" LIMIT {limit}"
        return pd.read_sql(query, self.engine)

    def read_warehouses(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM warehouses", self.engine)

    def join_transaction_warehouse(self) -> pd.DataFrame:
        query = """
            SELECT t.*, w.region AS warehouse_region, w.capacity, w.lat, w.lon
            FROM transactions t
            LEFT JOIN warehouses w ON t.warehouse_id = w.warehouse_id
        """
        return pd.read_sql(query, self.engine)

    def get_fraud_samples(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM transactions WHERE is_fraud = 1", self.engine)

    def get_regional_summary(self) -> pd.DataFrame:
        query = """
            SELECT region,
                   COUNT(*) AS txn_count,
                   SUM(price_usd) AS total_revenue,
                   AVG(price_usd) AS avg_price,
                   SUM(is_fraud) AS fraud_count,
                   SUM(is_defect) AS defect_count
            FROM transactions
            GROUP BY region
        """
        return pd.read_sql(query, self.engine)

    def execute_custom(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return pd.DataFrame(result.fetchall(), columns=result.keys())

    def write_predictions(self, df: pd.DataFrame, table: str = "predictions") -> None:
        df.to_sql(table, self.engine, if_exists="replace", index=False)

    def close(self) -> None:
        self.engine.dispose()
