from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Sequence

from backtest.utils import date_text, to_trade_datetime

from .constants import FALLBACK_WINDOW_TRADE_DAYS
from .models import DateRange, StockMeta


def merge_dates_to_ranges(dates: Iterable[datetime], trade_day_positions: dict[datetime, int]) -> list[DateRange]:
    unique_dates = sorted(set(dates))
    if not unique_dates:
        return []
    ranges: list[DateRange] = []
    start = unique_dates[0]
    end = unique_dates[0]
    for trade_date in unique_dates[1:]:
        if trade_day_positions.get(trade_date) == trade_day_positions.get(end, -10) + 1:
            end = trade_date
            continue
        ranges.append(DateRange(start, end))
        start = trade_date
        end = trade_date
    ranges.append(DateRange(start, end))
    return ranges


def build_missing_by_code(
    expected_dates_by_code: dict[str, set[datetime]],
    written_dates_by_code: dict[str, set[datetime]],
    invalid_dates: dict[str, list[datetime]],
) -> dict[str, list[datetime]]:
    missing_by_code: dict[str, list[datetime]] = {}
    for code, expected in expected_dates_by_code.items():
        valid = written_dates_by_code.get(code, set())
        # 缺行 + 非停牌异常行合并成待 fallback 集合；后续再由 Baostock 区分真实缺失和停牌。
        missing = sorted((expected - valid) | set(invalid_dates.get(code, [])))
        if missing:
            missing_by_code[code] = missing
    return missing_by_code


def filter_xt_results_to_update_window(
    expected_dates_by_code: dict[str, set[datetime]],
    docs: Sequence[dict[str, Any]],
    invalid_dates: dict[str, list[datetime]],
) -> tuple[list[dict[str, Any]], dict[str, list[datetime]]]:
    """批量拉取会覆盖批内最早起点；写库前再按每只股票自己的增量窗口过滤。"""

    filtered_docs = [
        doc
        for doc in docs
        if to_trade_datetime(doc["date"]) in expected_dates_by_code.get(str(doc.get("code", "")).lower(), set())
    ]
    filtered_invalid: dict[str, list[datetime]] = {}
    for code, dates in invalid_dates.items():
        expected = expected_dates_by_code.get(code, set())
        valid_dates = [date for date in dates if date in expected]
        if valid_dates:
            filtered_invalid[code] = valid_dates
    return filtered_docs, filtered_invalid


def add_written_doc_stats(
    docs: Sequence[dict[str, Any]],
    written_dates_by_code: dict[str, set[datetime]],
    updated_by_date: dict[str, int],
) -> int:
    for doc in docs:
        code = str(doc["code"])
        trade_date = to_trade_datetime(doc["date"])
        written_dates_by_code.setdefault(code, set()).add(trade_date)
        key = date_text(trade_date)
        updated_by_date[key] = updated_by_date.get(key, 0) + 1
    return len(docs)


def add_fallback_counts_by_date(
    counts: dict[str, int],
    fallback_summary: dict[str, Any],
) -> dict[str, int]:
    merged = dict(counts)
    for dates in fallback_summary.get("resolved_dates_by_code", {}).values():
        for date in dates:
            merged[str(date)] = merged.get(str(date), 0) + 1
    return dict(sorted(merged.items()))


def split_missing_by_fallback_window(
    missing_by_code: dict[str, list[datetime]],
    trade_dates: Sequence[datetime],
    *,
    fallback_window_trade_days: int = FALLBACK_WINDOW_TRADE_DAYS,
) -> tuple[dict[str, list[datetime]], dict[str, list[datetime]]]:
    recent_dates = set(trade_dates[-max(1, int(fallback_window_trade_days)) :])
    recent: dict[str, list[datetime]] = {}
    historical: dict[str, list[datetime]] = {}
    for code, dates in missing_by_code.items():
        recent_dates_for_code = sorted(date for date in dates if date in recent_dates)
        historical_dates_for_code = sorted(date for date in dates if date not in recent_dates)
        if recent_dates_for_code:
            recent[code] = recent_dates_for_code
        if historical_dates_for_code:
            historical[code] = historical_dates_for_code
    return recent, historical


def build_historical_missing_rows(historical_missing_by_code: dict[str, list[datetime]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for code, dates in sorted(historical_missing_by_code.items()):
        for trade_date in sorted(set(dates)):
            rows.append(
                {
                    "code": code,
                    "date": date_text(trade_date),
                    "reason": "outside_fallback_window",
                }
            )
    return rows


def missing_codes_on_latest_trade_date(
    missing_by_code: dict[str, list[datetime]],
    fallback_summary: dict[str, Any],
    latest_trade_date: datetime,
) -> list[str]:
    latest_text = latest_trade_date.strftime("%Y-%m-%d")
    resolved_by_code = fallback_summary.get("resolved_dates_by_code", {})
    result: list[str] = []
    for code, dates in sorted(missing_by_code.items()):
        if latest_trade_date not in set(dates):
            continue
        if latest_text in set(resolved_by_code.get(code, [])):
            continue
        result.append(code)
    return result
