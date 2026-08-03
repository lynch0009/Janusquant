"""日 K 入库字段与标准化工具。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from backtest.db import DuckDBConfig
from backtest.db.duckdb_write import upsert_frame
from backtest.db.precision import normalize_amount, normalize_price, normalize_price_series, normalize_volume


DAY_KLINE_QUERY_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
DAY_KLINE_RENAME_MAP = {
    "open": "o",
    "high": "h",
    "low": "l",
    "close": "c",
    "preclose": "prec",
    "volume": "v",
    "amount": "a",
}
DAY_KLINE_PRICE_COLUMNS = ("o", "h", "l", "c", "prec")
DAY_KLINE_INT_COLUMNS = ("v", "a")
DAY_KLINE_FLOAT_COLUMNS = ("turn", "pctChg")
DAY_KLINE_BOOL_COLUMNS = ("isST", "tradestatus")


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "y", "yes"})


def normalize_day_kline_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "adjustflag" in working.columns:
        working = working.drop(columns=["adjustflag"])

    if "code" in working.columns:
        working["code"] = working["code"].astype(str).str.strip().str.lower()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()

    working = working.rename(columns={key: value for key, value in DAY_KLINE_RENAME_MAP.items() if key in working.columns})

    for column in DAY_KLINE_PRICE_COLUMNS:
        if column in working.columns:
            working[column] = normalize_price_series(working[column])
    for column in DAY_KLINE_INT_COLUMNS:
        if column in working.columns:
            normalizer = normalize_volume if column == "v" else normalize_amount
            working[column] = working[column].map(normalizer)
    for column in DAY_KLINE_FLOAT_COLUMNS:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce").astype(float)
    for column in DAY_KLINE_BOOL_COLUMNS:
        if column in working.columns:
            working[column] = coerce_bool_series(working[column])
    return working


def build_baostock_day_doc(row: pd.Series, trade_date: datetime) -> dict[str, Any]:
    pct_value = pd.to_numeric(row.get("pctChg"), errors="coerce")
    turn_value = pd.to_numeric(row.get("turn"), errors="coerce")
    return {
        "code": str(row["code"]).strip().lower(),
        "date": trade_date,
        "o": normalize_price(row.get("o")),
        "h": normalize_price(row.get("h")),
        "l": normalize_price(row.get("l")),
        "c": normalize_price(row.get("c")),
        "prec": normalize_price(row.get("prec")),
        "v": normalize_volume(row.get("v")),
        "a": normalize_amount(row.get("a")),
        "turn": None if pd.isna(turn_value) else float(turn_value),
        "pctChg": None if pd.isna(pct_value) else float(pct_value),
        "tradestatus": bool(row.get("tradestatus", False)),
        "isST": bool(row.get("isST", False)),
    }


def write_day_kline_docs(db: DuckDBConfig, table: str, docs: list[dict[str, Any]] | pd.DataFrame) -> int:
    summary = upsert_frame(db, table, docs, key_columns=("code", "date"))
    return int(summary.rows_written)


def build_day_kline_update(doc: dict[str, Any]) -> dict[str, Any]:
    return doc


def build_day_kline_updates(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(docs)
