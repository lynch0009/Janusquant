"""Shared DuckDB write helpers for local research tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from typing import Any, Iterable, Sequence

import pandas as pd
from pandas.api import types as pd_types

from backtest.db.duckdb import DuckDBConfig
from backtest.utils import json_ready, records_to_frame


@dataclass(frozen=True)
class DuckDBWriteSummary:
    table: str
    rows_written: int
    mode: str


def quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def normalize_duckdb_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    if hasattr(value, "generation_time") and type(value).__name__ == "ObjectId":
        return str(value)
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(json_ready(value), ensure_ascii=False, default=str)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (datetime, date)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def normalize_duckdb_frame(records_or_frame: Iterable[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    frame = records_or_frame.copy() if isinstance(records_or_frame, pd.DataFrame) else records_to_frame(records_or_frame)
    if frame.empty:
        return frame
    for column in frame.columns:
        frame[column] = frame[column].map(normalize_duckdb_value)
        if column.lower().endswith("date") or column in {"date", "dt", "fetchedAt", "updated_at", "computedAt"}:
            converted = pd.to_datetime(frame[column], errors="coerce")
            if converted.notna().any() or frame[column].isna().all():
                frame[column] = converted
    return frame


def table_exists(db: DuckDBConfig, table: str) -> bool:
    frame = db.fetch_df(
        "select count(*) as count from information_schema.tables where table_schema = 'main' and table_name = ?",
        [table],
    )
    return bool(frame["count"].iloc[0])


def infer_duckdb_type(series: pd.Series) -> str:
    if pd_types.is_datetime64_any_dtype(series):
        return "timestamp"
    if pd_types.is_bool_dtype(series):
        return "boolean"
    if pd_types.is_integer_dtype(series):
        return "bigint"
    if pd_types.is_float_dtype(series):
        return "double"

    values = series.dropna()
    if values.empty:
        return "varchar"
    if values.map(lambda value: isinstance(value, bool)).all():
        return "boolean"
    if values.map(lambda value: isinstance(value, int) and not isinstance(value, bool)).all():
        return "bigint"
    if values.map(lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)).all():
        return "double"
    if values.map(lambda value: isinstance(value, (datetime, date, pd.Timestamp))).all():
        return "timestamp"
    return "varchar"


def ensure_table_columns(db: DuckDBConfig, table: str, frame: pd.DataFrame) -> None:
    if frame.empty or not table_exists(db, table):
        return
    existing = {row[0] for row in db.connection.execute(f"describe {quote_identifier(table)}").fetchall()}
    for column in frame.columns:
        if column not in existing:
            column_type = infer_duckdb_type(frame[column])
            db.execute(f"alter table {quote_identifier(table)} add column {quote_identifier(column)} {column_type}")


def replace_table(db: DuckDBConfig, table: str, records_or_frame: Iterable[dict[str, Any]] | pd.DataFrame) -> DuckDBWriteSummary:
    frame = normalize_duckdb_frame(records_or_frame)
    try:
        db.execute("begin transaction")
        with db.registered_frame("_duckdb_write_frame", frame):
            db.execute(f"create or replace table {quote_identifier(table)} as select * from _duckdb_write_frame")
        db.execute("commit")
    except Exception:
        db.execute("rollback")
        raise
    return DuckDBWriteSummary(table=table, rows_written=len(frame), mode="replace")


def append_frame(db: DuckDBConfig, table: str, records_or_frame: Iterable[dict[str, Any]] | pd.DataFrame) -> DuckDBWriteSummary:
    frame = normalize_duckdb_frame(records_or_frame)
    if frame.empty:
        return DuckDBWriteSummary(table=table, rows_written=0, mode="append")
    if not table_exists(db, table):
        return replace_table(db, table, frame)
    ensure_table_columns(db, table, frame)
    with db.registered_frame("_duckdb_write_frame", frame):
        db.execute(f"insert into {quote_identifier(table)} by name select * from _duckdb_write_frame")
    return DuckDBWriteSummary(table=table, rows_written=len(frame), mode="append")


def upsert_frame(
    db: DuckDBConfig,
    table: str,
    records_or_frame: Iterable[dict[str, Any]] | pd.DataFrame,
    *,
    key_columns: Sequence[str],
) -> DuckDBWriteSummary:
    frame = normalize_duckdb_frame(records_or_frame)
    if frame.empty:
        return DuckDBWriteSummary(table=table, rows_written=0, mode="upsert")
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing DuckDB upsert key columns for {table}: {missing}")
    frame = frame.drop_duplicates(subset=list(key_columns), keep="last").reset_index(drop=True)
    if not table_exists(db, table):
        return replace_table(db, table, frame)
    ensure_table_columns(db, table, frame)
    key_match = " and ".join(
        f"target.{quote_identifier(column)} is not distinct from source.{quote_identifier(column)}"
        for column in key_columns
    )
    try:
        db.execute("begin transaction")
        with db.registered_frame("_duckdb_write_frame", frame):
            db.execute(
                f"delete from {quote_identifier(table)} as target "
                f"using _duckdb_write_frame as source where {key_match}"
            )
            db.execute(f"insert into {quote_identifier(table)} by name select * from _duckdb_write_frame")
        db.execute("commit")
    except Exception:
        db.execute("rollback")
        raise
    return DuckDBWriteSummary(table=table, rows_written=len(frame), mode="upsert")
