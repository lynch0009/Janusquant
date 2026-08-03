from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd


SOURCE = "baostock"
YEAR_TYPE = "operate"

DATE_FIELDS = (
    "dividPreNoticeDate",
    "dividAgmPumDate",
    "dividPlanAnnounceDate",
    "dividPlanDate",
    "dividRegistDate",
    "dividOperateDate",
    "dividPayDate",
    "dividStockMarketDate",
)
FLOAT_FIELDS = (
    "dividCashPsBeforeTax",
    "dividCashPsAfterTax",
    "dividStocksPs",
    "dividReserveToStockPs",
)
EVENT_KEY_FLOAT_FIELDS = FLOAT_FIELDS
EVENT_KEY_DATE_FIELD = "dividOperateDate"
IMPLEMENTATION_DATE_FIELDS = (
    "dividOperateDate",
    "dividPayDate",
    "dividRegistDate",
    "dividStockMarketDate",
)


def normalize_event_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_event_float(value: Any) -> float | None:
    text = normalize_event_text(value)
    if not text:
        return None
    parsed = pd.to_numeric(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)


def normalize_event_date(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, datetime):
        return datetime(value.year, value.month, value.day).strftime("%Y-%m-%d")
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def normalize_event_year(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_dividend_event_key(doc: dict[str, Any], *, query_year: int | None = None) -> str:
    """Build a stable key for one operate-year dividend event.

    The ex-dividend/operate date is included to distinguish multiple same-year
    dividend events that have identical economic terms.
    """

    resolved_query_year = normalize_event_year(query_year if query_year is not None else doc.get("queryYear"))
    parts: list[Any] = [
        normalize_event_text(doc.get("code")),
        resolved_query_year,
        normalize_event_date(doc.get(EVENT_KEY_DATE_FIELD)),
        normalize_event_text(doc.get("dividCashStock")),
    ]
    parts.extend(normalize_event_float(doc.get(field)) for field in EVENT_KEY_FLOAT_FIELDS)
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def build_legacy_dividend_event_key(doc: dict[str, Any], *, query_year: int | None = None) -> str:
    """Build the pre-operate-date event key for matching existing documents."""

    resolved_query_year = normalize_event_year(query_year if query_year is not None else doc.get("queryYear"))
    parts: list[Any] = [
        normalize_event_text(doc.get("code")),
        resolved_query_year,
        normalize_event_text(doc.get("dividCashStock")),
    ]
    parts.extend(normalize_event_float(doc.get(field)) for field in EVENT_KEY_FLOAT_FIELDS)
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def infer_operate_query_year(doc: dict[str, Any]) -> int | None:
    """Infer the operate year for existing records that predate queryYear."""

    existing_query_year = normalize_event_year(doc.get("queryYear"))
    if existing_query_year is not None:
        return existing_query_year

    for field in IMPLEMENTATION_DATE_FIELDS:
        year = _date_year(doc.get(field))
        if year is not None:
            return year

    years = [_date_year(doc.get(field)) for field in DATE_FIELDS]
    valid_years = [year for year in years if year is not None]
    if not valid_years:
        return None
    return max(valid_years)


def _date_year(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.year
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.year
