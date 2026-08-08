from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

import baostock as bs
import pandas as pd

from backtest.fetch_data.baostock_utils import (
    BaostockQueryError,
    fetch_query_dataframe,
)
from backtest.fetch_data.day_kline_common import coerce_bool_series
from backtest.utils import is_st_name, normalize_internal_code, to_trade_datetime, to_xt_code
from backtest.utils.log import log_event

from .constants import FIXED_DAY_KLINE_INDEX_CODES


def _historical_dates_before(
    raw_dates: Sequence[datetime],
    latest_trade_date: datetime,
) -> list[datetime]:
    return sorted(
        {
            to_trade_datetime(trade_date)
            for trade_date in raw_dates
            if to_trade_datetime(trade_date) < latest_trade_date
        }
    )


def _parse_historical_st_frame(
    code: str,
    historical_dates: list[datetime],
    frame: pd.DataFrame,
) -> dict[datetime, bool]:
    start_date = historical_dates[0]
    end_date = historical_dates[-1]
    if frame.empty or "date" not in frame.columns or "isST" not in frame.columns:
        raise RuntimeError(
            f"historical ST status is empty or incomplete for {code} "
            f"{start_date:%Y-%m-%d}->{end_date:%Y-%m-%d}"
        )

    parsed_dates = frame["date"].map(to_trade_datetime)
    parsed_status = coerce_bool_series(frame["isST"])
    code_status = {
        trade_date: bool(is_st)
        for trade_date, is_st in zip(parsed_dates.tolist(), parsed_status.tolist())
    }
    missing_dates = [trade_date for trade_date in historical_dates if trade_date not in code_status]
    if missing_dates:
        raise RuntimeError(
            f"historical ST status missing expected dates for {code}: "
            f"missing_count={len(missing_dates)}, "
            f"missing_sample={[date.strftime('%Y-%m-%d') for date in missing_dates[:10]]}"
        )
    return {trade_date: code_status[trade_date] for trade_date in historical_dates}


def fetch_current_st_codes(
    xtdata_client: Any,
    codes: Sequence[str],
) -> set[str]:
    normalized_codes = sorted({normalize_internal_code(code) for code in codes})
    stock_codes = [code for code in normalized_codes if code not in FIXED_DAY_KLINE_INDEX_CODES]
    if not stock_codes:
        return set()

    xt_codes = [to_xt_code(code) for code in stock_codes]
    try:
        details = xtdata_client.get_instrument_detail_list(xt_codes, False)
    except Exception as exc:
        raise RuntimeError(f"failed to fetch QMT instrument details for ST status: {exc}") from exc
    if not isinstance(details, dict):
        raise RuntimeError(
            "failed to fetch QMT instrument details for ST status: "
            f"expected dict, got {type(details).__name__}"
        )
    return current_st_codes_from_details(stock_codes, details)


def current_st_codes_from_details(
    codes: Sequence[str],
    details_by_xt_code: dict[str, dict[str, Any]],
) -> set[str]:
    normalized_codes = sorted({normalize_internal_code(code) for code in codes})
    stock_codes = [code for code in normalized_codes if code not in FIXED_DAY_KLINE_INDEX_CODES]
    if not stock_codes:
        return set()

    missing_codes: list[str] = []
    blank_name_codes: list[str] = []
    current_st_codes: set[str] = set()
    for code in stock_codes:
        xt_code = to_xt_code(code)
        detail = details_by_xt_code.get(xt_code)
        if not isinstance(detail, dict):
            missing_codes.append(code)
            continue
        name = str(detail.get("InstrumentName") or "").strip()
        if not name:
            blank_name_codes.append(code)
            continue
        if is_st_name(name):
            current_st_codes.add(code)

    if missing_codes or blank_name_codes:
        raise RuntimeError(
            "incomplete QMT instrument details for ST status: "
            f"missing_count={len(missing_codes)}, blank_name_count={len(blank_name_codes)}, "
            f"missing_sample={missing_codes[:10]}, blank_name_sample={blank_name_codes[:10]}"
        )

    log_event(
        "info",
        "qmt current st codes fetched",
        stock_count=len(stock_codes),
        st_count=len(current_st_codes),
    )
    return current_st_codes


def fetch_historical_st_status(
    expected_dates_by_code: dict[str, list[datetime]],
    latest_trade_date: datetime,
    *,
    max_attempts: int = 3,
) -> dict[str, dict[datetime, bool]]:
    latest_trade_date = to_trade_datetime(latest_trade_date)
    status_by_code: dict[str, dict[datetime, bool]] = {}

    stock_units: list[dict[str, Any]] = []
    stock_dates: dict[str, list[datetime]] = {}
    for raw_code, raw_dates in sorted(expected_dates_by_code.items()):
        code = normalize_internal_code(raw_code)
        historical_dates = _historical_dates_before(raw_dates, latest_trade_date)
        if not historical_dates:
            continue
        if code in FIXED_DAY_KLINE_INDEX_CODES:
            status_by_code[code] = {trade_date: False for trade_date in historical_dates}
            continue

        start_date = historical_dates[0]
        end_date = historical_dates[-1]
        stock_dates[code] = historical_dates
        stock_units.append(
            {
                "code": code,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "context": f"{code} historical ST status {start_date:%Y-%m-%d}->{end_date:%Y-%m-%d}",
            }
        )

    for unit in stock_units:
        code = unit["code"]
        try:
            frame, _ = fetch_query_dataframe(
                bs.query_history_k_data_plus,
                code,
                "date,code,tradestatus,isST",
                start_date=unit["start_date"],
                end_date=unit["end_date"],
                frequency="d",
                adjustflag="3",
                retry_times=max_attempts,
                context=unit["context"],
            )
        except BaostockQueryError as exc:
            raise RuntimeError(
                f"failed to fetch historical ST status for {code} "
                f"{unit['start_date']}->{unit['end_date']}: {exc}"
            ) from exc
        status_by_code[code] = _parse_historical_st_frame(code, stock_dates[code], frame)

    log_event(
        "info",
        "baostock historical st status fetched",
        stock_count=len(status_by_code),
        status_count=sum(len(values) for values in status_by_code.values()),
    )
    return status_by_code


def build_st_status_by_code(
    xtdata_client: Any,
    expected_dates_by_code: dict[str, list[datetime]],
    latest_trade_date: datetime,
    *,
    current_details_by_xt_code: dict[str, dict[str, Any]] | None = None,
    latest_state_by_code: dict[str, dict[str, Any]] | None = None,
    max_attempts: int = 3,
) -> dict[str, dict[datetime, bool]]:
    latest_trade_date = to_trade_datetime(latest_trade_date)
    normalized_expected_dates = {
        normalize_internal_code(raw_code): sorted(
            {to_trade_datetime(trade_date) for trade_date in raw_dates}
        )
        for raw_code, raw_dates in expected_dates_by_code.items()
    }
    latest_codes = [
        code
        for code, dates in normalized_expected_dates.items()
        if latest_trade_date in dates
    ]
    current_st_codes = (
        current_st_codes_from_details(latest_codes, current_details_by_xt_code)
        if current_details_by_xt_code is not None
        else fetch_current_st_codes(xtdata_client, latest_codes)
    )
    normalized_latest_state = {
        normalize_internal_code(raw_code): state
        for raw_code, state in (latest_state_by_code or {}).items()
    }

    status_by_code: dict[str, dict[datetime, bool]] = {}
    historical_query_dates_by_code: dict[str, list[datetime]] = {}
    carried_stock_count = 0
    changed_stock_count = 0
    missing_baseline_stock_count = 0

    for code, expected_dates in normalized_expected_dates.items():
        if code in FIXED_DAY_KLINE_INDEX_CODES:
            status_by_code[code] = {trade_date: False for trade_date in expected_dates}
            continue

        if latest_trade_date not in expected_dates:
            historical_query_dates_by_code[code] = expected_dates
            continue

        current_is_st = code in current_st_codes
        latest_state = normalized_latest_state.get(code)
        has_baseline = (
            latest_state is not None
            and "isST" in latest_state
            and latest_state.get("isST") is not None
        )
        if has_baseline and bool(latest_state["isST"]) == current_is_st:
            status_by_code[code] = {
                trade_date: current_is_st for trade_date in expected_dates
            }
            carried_stock_count += 1
            continue

        historical_dates = [
            trade_date for trade_date in expected_dates if trade_date < latest_trade_date
        ]
        if historical_dates:
            historical_query_dates_by_code[code] = historical_dates
        status_by_code.setdefault(code, {})[latest_trade_date] = current_is_st
        if has_baseline:
            changed_stock_count += 1
        else:
            missing_baseline_stock_count += 1

    if historical_query_dates_by_code:
        historical_status = fetch_historical_st_status(
            historical_query_dates_by_code,
            latest_trade_date,
            max_attempts=max_attempts,
        )
        for code, values in historical_status.items():
            status_by_code.setdefault(code, {}).update(values)

    for raw_code in latest_codes:
        code = normalize_internal_code(raw_code)
        status_by_code.setdefault(code, {})[latest_trade_date] = (
            False if code in FIXED_DAY_KLINE_INDEX_CODES else code in current_st_codes
        )

    log_event(
        "info",
        "st status query plan finished",
        carried_stock_count=carried_stock_count,
        changed_stock_count=changed_stock_count,
        missing_baseline_stock_count=missing_baseline_stock_count,
        baostock_query_stock_count=len(historical_query_dates_by_code),
    )
    return status_by_code
