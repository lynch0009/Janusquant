"""Mongo basic_info 股票池加载工具。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import pandas as pd

from backtest.db import DuckDBConfig, quote_identifier
from backtest.utils import is_hs_a_share_code, normalize_internal_code, parse_basic_date


@dataclass(frozen=True)
class BasicStockWindow:
    code: str
    code_name: str
    ipo_date: datetime | None
    out_date: datetime | None

    def to_tuple(self) -> tuple[str, datetime | None, datetime | None]:
        return self.code, self.ipo_date, self.out_date


def _doc_to_stock(doc: dict[str, Any]) -> BasicStockWindow | None:
    raw_code = str(doc.get("code", "")).strip()
    if not raw_code:
        return None
    try:
        code = normalize_internal_code(raw_code)
    except ValueError:
        return None
    return BasicStockWindow(
        code=code,
        code_name=str(doc.get("code_name", "")).strip(),
        ipo_date=parse_basic_date(doc.get("ipoDate")),
        out_date=parse_basic_date(doc.get("outDate")),
    )


def _is_supported_stock(item: BasicStockWindow, *, hs_only: bool) -> bool:
    if hs_only:
        return is_hs_a_share_code(item.code)
    return item.code.startswith(("sh.", "sz.", "bj."))


def load_stock_windows(
    collection,
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    active_on: datetime | None = None,
    hs_only: bool = True,
) -> list[BasicStockWindow]:
    query: dict[str, Any] = {}
    if end_date is not None:
        query["ipoDate"] = {"$lte": end_date}
    if start_date is not None:
        query["$or"] = [{"outDate": None}, {"outDate": {"$gte": start_date}}]
    if active_on is not None:
        query["ipoDate"] = {"$lte": active_on}
        query["$or"] = [{"outDate": None}, {"outDate": {"$gte": active_on}}]

    cursor = collection.find(
        query,
        {"_id": 0, "code": 1, "code_name": 1, "ipoDate": 1, "outDate": 1},
    ).sort("code", 1)

    stocks: list[BasicStockWindow] = []
    for doc in cursor:
        item = _doc_to_stock(doc)
        if item is None or not _is_supported_stock(item, hs_only=hs_only):
            continue
        stocks.append(item)
    return stocks


def load_stock_windows_by_codes(collection, codes: Sequence[str]) -> list[BasicStockWindow]:
    normalized_codes = [normalize_internal_code(code) for code in codes]
    if not normalized_codes:
        return []

    cursor = collection.find(
        {"code": {"$in": normalized_codes}},
        {"_id": 0, "code": 1, "code_name": 1, "ipoDate": 1, "outDate": 1},
    )
    doc_map = {normalize_internal_code(str(doc.get("code", ""))): doc for doc in cursor if doc.get("code")}

    stocks: list[BasicStockWindow] = []
    for code in normalized_codes:
        doc = doc_map.get(code)
        if doc is None:
            stocks.append(BasicStockWindow(code=code, code_name="", ipo_date=None, out_date=None))
            continue
        item = _doc_to_stock(doc)
        stocks.append(item or BasicStockWindow(code=code, code_name="", ipo_date=None, out_date=None))
    return stocks


def _frame_to_stock_windows(frame: pd.DataFrame, *, hs_only: bool = True) -> list[BasicStockWindow]:
    stocks: list[BasicStockWindow] = []
    for row in frame.to_dict("records"):
        item = _doc_to_stock(row)
        if item is None or not _is_supported_stock(item, hs_only=hs_only):
            continue
        stocks.append(item)
    return stocks


def load_stock_windows_duckdb(
    db: DuckDBConfig,
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    active_on: datetime | None = None,
    hs_only: bool = True,
) -> list[BasicStockWindow]:
    clauses: list[str] = []
    params: list[Any] = []
    if end_date is not None:
        clauses.append("ipoDate <= ?")
        params.append(end_date)
    if start_date is not None:
        clauses.append("(outDate is null or outDate >= ?)")
        params.append(start_date)
    if active_on is not None:
        clauses = ["ipoDate <= ?", "(outDate is null or outDate >= ?)"]
        params = [active_on, active_on]
    where_sql = f"where {' and '.join(clauses)}" if clauses else ""
    frame = db.fetch_df(
        f"""
        select code, code_name, ipoDate, outDate
        from {quote_identifier("A_stock_market_basic_info")}
        {where_sql}
        order by code
        """,
        params,
    )
    return _frame_to_stock_windows(frame, hs_only=hs_only)


def load_stock_windows_by_codes_duckdb(db: DuckDBConfig, codes: Sequence[str]) -> list[BasicStockWindow]:
    normalized_codes = [normalize_internal_code(code) for code in codes]
    if not normalized_codes:
        return []
    frame = db.fetch_df(
        f"""
        select code, code_name, ipoDate, outDate
        from {quote_identifier("A_stock_market_basic_info")}
        where code in ({", ".join("?" for _ in normalized_codes)})
        """,
        normalized_codes,
    )
    doc_map = {normalize_internal_code(str(row.get("code", ""))): row for row in frame.to_dict("records") if row.get("code")}
    stocks: list[BasicStockWindow] = []
    for code in normalized_codes:
        doc = doc_map.get(code)
        if doc is None:
            stocks.append(BasicStockWindow(code=code, code_name="", ipo_date=None, out_date=None))
            continue
        item = _doc_to_stock(doc)
        stocks.append(item or BasicStockWindow(code=code, code_name="", ipo_date=None, out_date=None))
    return stocks
