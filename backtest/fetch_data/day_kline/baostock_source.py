from __future__ import annotations

from datetime import datetime
from typing import Any

import baostock as bs
import pandas as pd

from backtest.db import DuckDBConfig
from backtest.fetch_data.baostock_utils import BaostockQueryError, fetch_query_dataframe
from backtest.fetch_data.day_kline_common import DAY_KLINE_QUERY_FIELDS, build_baostock_day_doc, normalize_day_kline_frame, write_day_kline_docs
from backtest.utils import is_bj_code, to_trade_datetime
from backtest.utils.log import log_event

from .constants import (
    DAY_COLLECTION,
    DEFAULT_MAX_FALLBACK_MISSING_DAYS,
    DEFAULT_MAX_FALLBACK_MISSING_STOCKS,
)
from .missing import merge_dates_to_ranges
from .models import DateRange


def fetch_baostock_range(code: str, date_range: DateRange) -> pd.DataFrame:
    try:
        frame, _ = fetch_query_dataframe(
            bs.query_history_k_data_plus,
            code,
            DAY_KLINE_QUERY_FIELDS,
            start_date=date_range.start.strftime("%Y-%m-%d"),
            end_date=date_range.end.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="3",
            context=f"{code} {date_range.count_label} baostock day kline",
        )
    except BaostockQueryError as exc:
        log_event(
            "warning",
            "baostock_fallback_query_failed",
            code=code,
            start=date_range.start.strftime("%Y-%m-%d"),
            end=date_range.end.strftime("%Y-%m-%d"),
            error=str(exc),
        )
        return pd.DataFrame()
    return normalize_day_kline_frame(frame)


def fallback_missing_dates_with_baostock(
    cfg: DuckDBConfig,
    missing_by_code: dict[str, list[datetime]],
    trade_day_positions: dict[datetime, int],
    *,
    batch_size: int = 2000,
    max_missing_stocks: int = DEFAULT_MAX_FALLBACK_MISSING_STOCKS,
    max_missing_days: int = DEFAULT_MAX_FALLBACK_MISSING_DAYS,
    dry_run: bool = False,
) -> dict[str, Any]:
    docs_to_write: list[dict[str, Any]] = []
    total_missing_stocks = len(missing_by_code)
    total_missing_days = sum(len(set(values)) for values in missing_by_code.values())
    summary = {
        "fallback_docs": 0,
        "unresolved_days": 0,
        "normal_suspend_days": 0,
        "skipped_by_threshold": 0,
        "write_batches": 0,
        "resolved_dates_by_code": {},
        "unresolved_dates_by_code": {},
    }
    if total_missing_stocks > max_missing_stocks or total_missing_days > max_missing_days:
        summary["unresolved_days"] = total_missing_days
        summary["skipped_by_threshold"] = 1
        summary["unresolved_dates_by_code"] = {
            code: [date.strftime("%Y-%m-%d") for date in sorted(set(dates))]
            for code, dates in sorted(missing_by_code.items())
        }
        log_event(
            "warning",
            "baostock_fallback_skipped_by_threshold",
            missing_stocks=total_missing_stocks,
            missing_days=total_missing_days,
            max_missing_stocks=max_missing_stocks,
            max_missing_days=max_missing_days,
        )
        return summary

    for code, missing_dates in sorted(missing_by_code.items()):
        for date_range in merge_dates_to_ranges(missing_dates, trade_day_positions):
            frame = fetch_baostock_range(code, date_range)
            if frame.empty:
                unresolved_count = len([date for date in set(missing_dates) if date_range.start <= date <= date_range.end])
                summary["unresolved_days"] += unresolved_count
                summary["unresolved_dates_by_code"].setdefault(code, []).extend(
                    date.strftime("%Y-%m-%d")
                    for date in sorted(set(missing_dates))
                    if date_range.start <= date <= date_range.end
                )
                log_event(
                    "warning",
                    "unresolved_missing",
                    code=code,
                    start=date_range.start.strftime("%Y-%m-%d"),
                    end=date_range.end.strftime("%Y-%m-%d"),
                    missing_days=unresolved_count,
                    reason="baostock_empty",
                )
                continue

            fetched_dates: set[datetime] = set()
            fallback_docs_in_range = 0
            suspend_days_in_range = 0
            for _, row in frame.iterrows():
                trade_date = to_trade_datetime(row["date"])
                fetched_dates.add(trade_date)
                summary["resolved_dates_by_code"].setdefault(code, []).append(trade_date.strftime("%Y-%m-%d"))
                doc = build_baostock_day_doc(row, trade_date)
                docs_to_write.append(doc)
                if doc["tradestatus"]:
                    # Baostock 返回真实交易日数据时打印 warning，说明 xtquant 这一天确实有缺口并已补。
                    fallback_docs_in_range += 1
                    summary["fallback_docs"] += 1
                else:
                    # Baostock 明确标为停牌，属于正常无交易，不作为 unresolved 缺失处理。
                    suspend_days_in_range += 1
                    summary["normal_suspend_days"] += 1
                if len(docs_to_write) >= batch_size:
                    if not dry_run:
                        write_day_kline_docs(cfg, DAY_COLLECTION, docs_to_write)
                        summary["write_batches"] += 1
                    docs_to_write.clear()

            if fallback_docs_in_range:
                log_event(
                    "warning",
                    "baostock_fallback",
                    code=code,
                    start=date_range.start.strftime("%Y-%m-%d"),
                    end=date_range.end.strftime("%Y-%m-%d"),
                    missing_days=fallback_docs_in_range,
                    source="baostock_fallback",
                )

            unresolved_dates = [
                date
                for date in set(missing_dates)
                if date_range.start <= date <= date_range.end and date not in fetched_dates
            ]
            if unresolved_dates:
                summary["unresolved_days"] += len(unresolved_dates)
                summary["unresolved_dates_by_code"].setdefault(code, []).extend(
                    date.strftime("%Y-%m-%d") for date in sorted(unresolved_dates)
                )
                log_event(
                    "warning",
                    "unresolved_missing",
                    code=code,
                    start=date_range.start.strftime("%Y-%m-%d"),
                    end=date_range.end.strftime("%Y-%m-%d"),
                    missing_days=len(unresolved_dates),
                    normal_suspend_days=suspend_days_in_range,
                )

    if docs_to_write:
        if not dry_run:
            write_day_kline_docs(cfg, DAY_COLLECTION, docs_to_write)
            summary["write_batches"] += 1
    for key in ("resolved_dates_by_code", "unresolved_dates_by_code"):
        summary[key] = {
            code: sorted(set(dates))
            for code, dates in sorted(summary[key].items())
            if dates
        }
    return summary
