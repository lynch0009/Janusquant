from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

import baostock as bs

from backtest.fetch_data.baostock_utils import BaostockQueryError, fetch_query_dataframe
from backtest.fetch_data.day_kline_common import coerce_bool_series
from backtest.utils import is_st_name, normalize_internal_code, to_trade_datetime, to_xt_code
from backtest.utils.log import log_event

from .constants import FIXED_DAY_KLINE_INDEX_CODES


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
) -> dict[str, dict[datetime, bool]]:
    latest_trade_date = to_trade_datetime(latest_trade_date)
    status_by_code: dict[str, dict[datetime, bool]] = {}

    for raw_code, raw_dates in sorted(expected_dates_by_code.items()):
        code = normalize_internal_code(raw_code)
        if code in FIXED_DAY_KLINE_INDEX_CODES:
            historical_dates = {
                to_trade_datetime(trade_date)
                for trade_date in raw_dates
                if to_trade_datetime(trade_date) < latest_trade_date
            }
            if historical_dates:
                status_by_code[code] = {trade_date: False for trade_date in historical_dates}
            continue

        historical_dates = sorted(
            {
                to_trade_datetime(trade_date)
                for trade_date in raw_dates
                if to_trade_datetime(trade_date) < latest_trade_date
            }
        )
        if not historical_dates:
            continue

        start_date = historical_dates[0]
        end_date = historical_dates[-1]
        try:
            frame, _ = fetch_query_dataframe(
                bs.query_history_k_data_plus,
                code,
                "date,code,tradestatus,isST",
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="3",
                context=f"{code} historical ST status {start_date:%Y-%m-%d}->{end_date:%Y-%m-%d}",
            )
        except BaostockQueryError as exc:
            raise RuntimeError(
                f"failed to fetch historical ST status for {code} "
                f"{start_date:%Y-%m-%d}->{end_date:%Y-%m-%d}: {exc}"
            ) from exc

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
        status_by_code[code] = {trade_date: code_status[trade_date] for trade_date in historical_dates}

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
) -> dict[str, dict[datetime, bool]]:
    latest_trade_date = to_trade_datetime(latest_trade_date)
    latest_codes = [
        code
        for code, dates in expected_dates_by_code.items()
        if latest_trade_date in {to_trade_datetime(trade_date) for trade_date in dates}
    ]
    current_st_codes = (
        current_st_codes_from_details(latest_codes, current_details_by_xt_code)
        if current_details_by_xt_code is not None
        else fetch_current_st_codes(xtdata_client, latest_codes)
    )
    status_by_code = fetch_historical_st_status(expected_dates_by_code, latest_trade_date)

    for raw_code in latest_codes:
        code = normalize_internal_code(raw_code)
        status_by_code.setdefault(code, {})[latest_trade_date] = (
            False if code in FIXED_DAY_KLINE_INDEX_CODES else code in current_st_codes
        )
    return status_by_code


# 暂时弃用：pywencai 当前接口会返回 HTTP 403，保留原标准化与查询思路供后续恢复。
#
# def normalize_pywencai_st_codes(frame) -> tuple[set[str], int]:
#     if frame is None or getattr(frame, "empty", True):
#         raise ValueError("pywencai ST frame is empty")
#
#     code_column = None
#     for candidate in ("股票代码", "代码", "code"):
#         if candidate in frame.columns:
#             code_column = candidate
#             break
#     if code_column is None:
#         raise ValueError("pywencai ST frame missing code column: 股票代码/代码/code")
#
#     codes: set[str] = set()
#     skipped = 0
#     for raw_code in frame[code_column].tolist():
#         try:
#             codes.add(normalize_internal_code(str(raw_code).strip()))
#         except ValueError:
#             skipped += 1
#     if not codes:
#         raise ValueError("pywencai ST frame has no supported A-share codes")
#     return codes, skipped
#
#
# def fetch_current_st_codes_from_pywencai_once(
#     pywencai_module: Any | None = None,
# ) -> tuple[set[str], int]:
#     if pywencai_module is None:
#         import pywencai as pywencai_module
#
#     frame = pywencai_module.get(query="所有st股票", loop=True)
#     return normalize_pywencai_st_codes(frame)
