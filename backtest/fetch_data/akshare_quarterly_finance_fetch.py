# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.db import DuckDBConfig, quote_identifier, upsert_frame
from backtest.fetch_data.stock_universe import BasicStockWindow, load_stock_windows_duckdb, load_stock_windows_by_codes_duckdb
from backtest.utils import parse_basic_date
from backtest.utils.frame_utils import sort_frame
from backtest.utils.log import log_event
from backtest.utils.quarter_utils import (
    format_quarter,
    iter_quarter_pairs,
    quarter_end,
    quarter_from_date,
    resolve_incremental_target_quarter,
)
from backtest.utils.security_code import (
    normalize_internal_code,
    plain_code,
    split_code_list,
    to_akshare_em_symbol,
    to_akshare_indicator_symbol,
)


COLLECTION_NAME = "A_stock_market_akshare_quarterly_finance"
BASIC_INFO_COLLECTION = "A_stock_market_basic_info"
SOURCE = "eastmoney"

PROFIT_FIELDS = (
    "SECUCODE",
    "SECURITY_CODE",
    "SECURITY_NAME_ABBR",
    "REPORT_DATE",
    "REPORT_TYPE",
    "REPORT_DATE_NAME",
    "NOTICE_DATE",
    "UPDATE_DATE",
    "TOTAL_OPERATE_INCOME",
    "OPERATE_INCOME",
    "NETPROFIT",
    "PARENT_NETPROFIT",
    "BASIC_EPS",
    "DILUTED_EPS",
    "DEDUCT_PARENT_NETPROFIT",
)
INDICATOR_FIELDS = (
    "SECUCODE",
    "SECURITY_CODE",
    "SECURITY_NAME_ABBR",
    "REPORT_DATE",
    "EPSJB",
    "TOTALOPERATEREVE",
    "GROSS_PROFIT",
    "PARENTNETPROFIT",
    "DEDU_PARENT_PROFIT",
    "TOTALOPERATEREVETZ",
    "PARENTNETPROFITTZ",
    "DPNP_YOY_RATIO",
    "YYZSRGDHBZC",
    "NETPROFITRPHBZC",
    "KFJLRGDHBZC",
    "ROE_DILUTED",
    "GROSS_PROFIT_RATIO",
    "NET_PROFIT_RATIO",
    "SEASON_LABEL",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch AkShare Eastmoney single-quarter finance data into DuckDB."
    )
    parser.add_argument("--mode", choices=["range", "incremental"], default="incremental")
    parser.add_argument("--start-date", help="Range mode start statDate, e.g. 2025-01-01.")
    parser.add_argument("--end-date", help="Range mode end statDate, e.g. 2026-03-31.")
    parser.add_argument("--today", help="Incremental mode current date, default: today.")
    parser.add_argument(
        "--lookback-quarters",
        type=int,
        default=0,
        help=(
            "Incremental mode compatibility option: also refresh latest N reported quarters. "
            "Default 0 means only fill missing quarters."
        ),
    )
    parser.add_argument(
        "--refresh-quarters",
        type=int,
        default=None,
        help="Incremental mode: also refresh latest N reported quarters for revised data.",
    )
    parser.add_argument(
        "--max-missing-quarters",
        type=int,
        default=2,
        help="Incremental mode: skip stocks whose missing gap is larger than this value.",
    )
    parser.add_argument(
        "--backfill-new-stocks",
        action="store_true",
        help="Incremental mode: allow stocks with no existing DuckDB records to backfill from IPO.",
    )
    parser.add_argument("--codes", default=None, help="Comma separated stock codes, e.g. sz.300308,300502.")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N stocks after filtering.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and normalize only; do not write DuckDB.")
    return parser.parse_args()


def build_missing_quarters(
    existing_stat_dates: set[datetime],
    ipo_date: datetime | None,
    target_year: int,
    target_quarter: int,
    *,
    backfill_new_stocks: bool = False,
) -> list[tuple[int, int]]:
    if not existing_stat_dates and not backfill_new_stocks:
        return []

    if ipo_date is not None:
        start_year, start_quarter = quarter_from_date(ipo_date)
    elif existing_stat_dates:
        start_year, start_quarter = quarter_from_date(min(existing_stat_dates))
    else:
        return []

    if (start_year, start_quarter) > (target_year, target_quarter):
        return []

    existing_quarters = {quarter_from_date(stat_date) for stat_date in existing_stat_dates}
    return [
        (year, quarter)
        for year, quarter in iter_quarter_pairs(start_year, start_quarter, target_year, target_quarter)
        if (year, quarter) not in existing_quarters
    ]


def normalize_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text or text in {"-", "--", "None", "nan", "NaN"}:
            return None
        multiplier = 1.0
        if text.endswith("%"):
            text = text[:-1]
        if text.endswith("亿"):
            multiplier = 100000000.0
            text = text[:-1]
        elif text.endswith("万"):
            multiplier = 10000.0
            text = text[:-1]
        try:
            number = float(text) * multiplier
        except ValueError:
            return None
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

    if math.isnan(number) or math.isinf(number):
        return None
    return number


def normalize_percent_to_decimal(value: Any) -> float | None:
    number = normalize_number(value)
    if number is None:
        return None
    return number / 100.0


def first_present(row: dict[str, Any], fields: tuple[str, ...] | list[str]) -> Any:
    for field in fields:
        if field not in row:
            continue
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def max_available_date(*values: Any) -> datetime | None:
    dates = [item for item in (parse_basic_date(value) for value in values) if item is not None]
    if not dates:
        return None
    return max(dates)


def stock_window_to_dict(stock: BasicStockWindow) -> dict[str, Any]:
    return {
        "code": stock.code,
        "code_name": stock.code_name,
        "ipoDate": stock.ipo_date,
        "outDate": stock.out_date,
    }


def load_existing_stat_dates_map(db: DuckDBConfig, codes: list[str]) -> dict[str, set[datetime]]:
    if not codes:
        return {}
    frame = db.fetch_df(
        f"""
        select code, statDate
        from {quote_identifier(COLLECTION_NAME)}
        where code in ({", ".join("?" for _ in codes)}) and source = ? and statDate is not null
        """,
        [*codes, SOURCE],
    )
    result: dict[str, set[datetime]] = {}
    for code, group in frame.groupby("code"):
        code = str(code).strip()
        if not code:
            continue
        result[code] = {
            stat_date
            for stat_date in (parse_basic_date(value) for value in group["statDate"].tolist())
            if stat_date is not None
        }
    return result


def select_existing_columns(frame: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    existing_fields = [field for field in fields if field in frame.columns]
    return frame[existing_fields].copy()


def prepare_source_frame(frame: pd.DataFrame, fields: tuple[str, ...], source_label: str) -> pd.DataFrame:
    selected = select_existing_columns(frame, fields)
    if selected.empty or "REPORT_DATE" not in selected.columns:
        return pd.DataFrame()
    selected["statDate"] = pd.to_datetime(selected["REPORT_DATE"], errors="coerce")
    selected = selected.dropna(subset=["statDate"])
    selected = selected.drop_duplicates(subset=["statDate"], keep="last")
    renamed = {
        column: f"{source_label}_{column}"
        for column in selected.columns
        if column != "statDate"
    }
    return selected.rename(columns=renamed)


def merge_quarterly_frames(profit_frame: pd.DataFrame, indicator_frame: pd.DataFrame) -> pd.DataFrame:
    profit = prepare_source_frame(profit_frame, PROFIT_FIELDS, "profit")
    indicator = prepare_source_frame(indicator_frame, INDICATOR_FIELDS, "indicator")
    if profit.empty and indicator.empty:
        return pd.DataFrame()
    if profit.empty:
        return sort_frame(indicator, sort_field="statDate")
    if indicator.empty:
        return sort_frame(profit, sort_field="statDate")
    merged = pd.merge(profit, indicator, on="statDate", how="outer", validate="one_to_one")
    return sort_frame(merged, sort_field="statDate")


def fetch_akshare_frames(ak, code: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    errors: list[str] = []
    profit_frame = pd.DataFrame()
    indicator_frame = pd.DataFrame()

    try:
        profit_frame = ak.stock_profit_sheet_by_quarterly_em(symbol=to_akshare_em_symbol(code))
    except Exception as exc:  # noqa: BLE001 - single-stock failures should not stop the batch.
        errors.append(f"profit:{type(exc).__name__}:{exc}")

    try:
        indicator_frame = ak.stock_financial_analysis_indicator_em(
            symbol=to_akshare_indicator_symbol(code),
            indicator="按单季度",
        )
    except Exception as exc:  # noqa: BLE001 - single-stock failures should not stop the batch.
        errors.append(f"indicator:{type(exc).__name__}:{exc}")

    return profit_frame, indicator_frame, errors


def report_missing_columns(code: str, profit_frame: pd.DataFrame, indicator_frame: pd.DataFrame) -> None:
    missing_profit = [field for field in PROFIT_FIELDS if field not in profit_frame.columns]
    missing_indicator = [field for field in INDICATOR_FIELDS if field not in indicator_frame.columns]
    if missing_profit:
        log_event("warning", "akshare_quarterly_missing_profit_columns", code=code, fields=missing_profit)
    if missing_indicator:
        log_event("warning", "akshare_quarterly_missing_indicator_columns", code=code, fields=missing_indicator)


def normalize_doc(row: dict[str, Any], stock: dict[str, Any], fetched_at: datetime) -> dict[str, Any]:
    stat_date = parse_basic_date(row.get("statDate"))
    if stat_date is None:
        raise ValueError("statDate is required")
    fiscal_year, fiscal_quarter = quarter_from_date(stat_date)

    notice_date = parse_basic_date(first_present(row, ("profit_NOTICE_DATE",)))
    update_date = parse_basic_date(first_present(row, ("profit_UPDATE_DATE",)))
    available_date = max_available_date(notice_date, update_date) or stat_date
    code = normalize_internal_code(stock["code"])

    doc: dict[str, Any] = {
        "code": code,
        "plain_code": plain_code(code),
        "akshare_symbol": to_akshare_em_symbol(code),
        "indicator_symbol": to_akshare_indicator_symbol(code),
        "code_name": first_present(
            row,
            ("profit_SECURITY_NAME_ABBR", "indicator_SECURITY_NAME_ABBR"),
        )
        or stock.get("code_name", ""),
        "source": SOURCE,
        "statDate": stat_date,
        "fiscalYear": fiscal_year,
        "fiscalQuarter": fiscal_quarter,
        "reportType": first_present(row, ("profit_REPORT_TYPE", "indicator_SEASON_LABEL")),
        "reportDateName": first_present(row, ("profit_REPORT_DATE_NAME", "indicator_SEASON_LABEL")),
        "noticeDate": notice_date,
        "updateDate": update_date,
        "availableDate": available_date,
        "fetchedAt": fetched_at,
        "revenue_single": normalize_number(
            first_present(row, ("indicator_TOTALOPERATEREVE", "profit_TOTAL_OPERATE_INCOME"))
        ),
        "operate_revenue_single": normalize_number(first_present(row, ("profit_OPERATE_INCOME",))),
        "net_profit_single": normalize_number(first_present(row, ("profit_NETPROFIT",))),
        "parent_net_profit_single": normalize_number(
            first_present(row, ("indicator_PARENTNETPROFIT", "profit_PARENT_NETPROFIT"))
        ),
        "deduct_parent_net_profit_single": normalize_number(
            first_present(row, ("indicator_DEDU_PARENT_PROFIT", "profit_DEDUCT_PARENT_NETPROFIT"))
        ),
        "basic_eps_single": normalize_number(first_present(row, ("profit_BASIC_EPS", "indicator_EPSJB"))),
        "diluted_eps_single": normalize_number(first_present(row, ("profit_DILUTED_EPS",))),
        "gross_profit_single": normalize_number(first_present(row, ("indicator_GROSS_PROFIT",))),
        "revenue_yoy": normalize_percent_to_decimal(first_present(row, ("indicator_TOTALOPERATEREVETZ",))),
        "parent_net_profit_yoy": normalize_percent_to_decimal(first_present(row, ("indicator_PARENTNETPROFITTZ",))),
        "deduct_parent_net_profit_yoy": normalize_percent_to_decimal(first_present(row, ("indicator_DPNP_YOY_RATIO",))),
        "revenue_qoq": normalize_percent_to_decimal(first_present(row, ("indicator_YYZSRGDHBZC",))),
        "parent_net_profit_qoq": normalize_percent_to_decimal(first_present(row, ("indicator_NETPROFITRPHBZC",))),
        "deduct_parent_net_profit_qoq": normalize_percent_to_decimal(first_present(row, ("indicator_KFJLRGDHBZC",))),
        "roe_diluted": normalize_percent_to_decimal(first_present(row, ("indicator_ROE_DILUTED",))),
        "gross_margin": normalize_percent_to_decimal(first_present(row, ("indicator_GROSS_PROFIT_RATIO",))),
        "net_margin": normalize_percent_to_decimal(first_present(row, ("indicator_NET_PROFIT_RATIO",))),
        "raw_profit_columns": [field for field in PROFIT_FIELDS if f"profit_{field}" in row],
        "raw_indicator_columns": [field for field in INDICATOR_FIELDS if f"indicator_{field}" in row],
    }

    return {key: value for key, value in doc.items() if value is not None}


def build_docs_for_stock(ak, stock: dict[str, Any], fetched_at: datetime) -> tuple[list[dict[str, Any]], list[str]]:
    code = stock["code"]
    profit_frame, indicator_frame, errors = fetch_akshare_frames(ak, code)
    if errors:
        log_event("error", "akshare_quarterly_fetch_error", code=code, errors=errors)

    if profit_frame.empty and indicator_frame.empty:
        log_event("warning", "akshare_quarterly_skipped_empty", code=code)
        return [], errors

    report_missing_columns(code, profit_frame, indicator_frame)
    if not profit_frame.empty and not indicator_frame.empty and len(profit_frame) != len(indicator_frame):
        log_event(
            "warning",
            "akshare_quarterly_frame_length_mismatch",
            code=code,
            profit_rows=len(profit_frame),
            indicator_rows=len(indicator_frame),
        )

    merged = merge_quarterly_frames(profit_frame, indicator_frame)
    docs: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        try:
            docs.append(normalize_doc(row, stock, fetched_at))
        except Exception as exc:  # noqa: BLE001 - keep the batch running and make the bad row visible.
            log_event("error", "akshare_quarterly_normalize_error", code=code, error=exc, row=row)
    return docs, errors


def filter_range_docs(docs: list[dict[str, Any]], start_date: datetime, end_date: datetime) -> list[dict[str, Any]]:
    return [doc for doc in docs if start_date <= doc["statDate"] <= end_date]


def filter_docs_by_quarters(docs: list[dict[str, Any]], quarters: set[tuple[int, int]]) -> list[dict[str, Any]]:
    if not quarters:
        return []
    target_dates = {quarter_end(year, quarter) for year, quarter in quarters}
    matched = [doc for doc in docs if doc["statDate"] in target_dates]
    return sorted(matched, key=lambda item: item["statDate"])


def latest_doc_quarters(docs: list[dict[str, Any]], count: int) -> set[tuple[int, int]]:
    if count <= 0:
        return set()
    sorted_docs = sorted(docs, key=lambda item: item["statDate"], reverse=True)
    return {
        quarter_from_date(doc["statDate"])
        for doc in sorted_docs[:count]
    }


def resolve_refresh_quarters(args: argparse.Namespace) -> int:
    value = args.refresh_quarters if args.refresh_quarters is not None else args.lookback_quarters
    return max(0, int(value or 0))


def resolve_stocks(args: argparse.Namespace, db: DuckDBConfig, *, start_date: datetime | None, end_date: datetime | None, today: datetime | None) -> list[dict[str, Any]]:
    codes = split_code_list(args.codes)
    if codes:
        stock_windows = load_stock_windows_by_codes_duckdb(db, codes)
    elif args.mode == "range":
        if start_date is None or end_date is None:
            raise ValueError("range mode requires --start-date and --end-date")
        stock_windows = load_stock_windows_duckdb(db, start_date=start_date, end_date=end_date)
    else:
        if today is None:
            raise ValueError("incremental mode requires today")
        stock_windows = load_stock_windows_duckdb(db, active_on=today)

    stocks = [stock_window_to_dict(stock) for stock in stock_windows]
    if args.limit is not None:
        stocks = stocks[: max(0, args.limit)]
    return stocks


def run(args: argparse.Namespace) -> None:
    import akshare as ak

    if args.mode == "range":
        start_date = parse_basic_date(args.start_date)
        end_date = parse_basic_date(args.end_date)
        today = None
        if start_date is None or end_date is None:
            raise ValueError("range mode requires valid --start-date and --end-date")
    else:
        start_date = None
        end_date = None
        today = parse_basic_date(args.today) or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    fetched_at = datetime.now()
    db = DuckDBConfig()
    stocks = resolve_stocks(args, db, start_date=start_date, end_date=end_date, today=today)
    target_year = None
    target_quarter = None
    refresh_quarters = 0
    existing_stat_dates_map: dict[str, set[datetime]] = {}
    if args.mode == "incremental":
        assert today is not None
        target_year, target_quarter = resolve_incremental_target_quarter(today)
        refresh_quarters = resolve_refresh_quarters(args)
        existing_stat_dates_map = load_existing_stat_dates_map(db, [stock["code"] for stock in stocks])

    log_event(
        "info",
        "akshare_quarterly_fetch_start",
        mode=args.mode,
        collection=COLLECTION_NAME,
        stock_count=len(stocks),
        dry_run=args.dry_run,
        start_date=start_date,
        end_date=end_date,
        today=today,
        target_quarter=format_quarter(target_year, target_quarter) if target_year is not None and target_quarter is not None else None,
        lookback_quarters=args.lookback_quarters,
        refresh_quarters=refresh_quarters,
        akshare_version=getattr(ak, "__version__", "unknown"),
    )

    pending_docs: list[dict[str, Any]] = []
    fetched_stock_count = 0
    skipped_up_to_date_count = 0
    skipped_no_history_count = 0
    skipped_large_gap_count = 0
    skipped_empty_count = 0
    error_stock_count = 0
    matched_doc_count = 0
    missing_quarter_count = 0
    refreshed_quarter_count = 0
    batch_count = 0
    dry_run_examples: list[dict[str, Any]] = []
    skipped_gap_examples: list[dict[str, Any]] = []

    for index, stock in enumerate(stocks, start=1):
        code = stock["code"]
        missing_quarters: list[tuple[int, int]] = []
        if args.mode == "incremental":
            assert target_year is not None and target_quarter is not None
            existing_stat_dates = existing_stat_dates_map.get(code, set())
            missing_quarters = build_missing_quarters(
                existing_stat_dates,
                stock.get("ipoDate"),
                target_year,
                target_quarter,
                backfill_new_stocks=args.backfill_new_stocks,
            )
            if not existing_stat_dates and not args.backfill_new_stocks:
                skipped_no_history_count += 1
                log_event(
                    "info",
                    "akshare_quarterly_stock_skipped_no_history",
                    index=index,
                    total=len(stocks),
                    code=code,
                    target_quarter=format_quarter(target_year, target_quarter),
                )
                continue
            if (
                args.max_missing_quarters >= 0
                and len(missing_quarters) > args.max_missing_quarters
            ):
                skipped_large_gap_count += 1
                gap_item = {
                    "code": code,
                    "code_name": stock.get("code_name", ""),
                    "last_stat_date": max(existing_stat_dates).strftime("%Y-%m-%d") if existing_stat_dates else None,
                    "missing_quarters": [format_quarter(year, quarter) for year, quarter in missing_quarters],
                }
                skipped_gap_examples.append(gap_item)
                log_event(
                    "warning",
                    "akshare_quarterly_stock_skipped_large_gap",
                    index=index,
                    total=len(stocks),
                    **gap_item,
                )
                continue
            if not missing_quarters and refresh_quarters <= 0:
                skipped_up_to_date_count += 1
                log_event(
                    "info",
                    "akshare_quarterly_stock_skipped_up_to_date",
                    index=index,
                    total=len(stocks),
                    code=code,
                    last_stat_date=max(existing_stat_dates) if existing_stat_dates else None,
                    target_quarter=format_quarter(target_year, target_quarter),
                )
                continue

        log_event("info", "akshare_quarterly_stock_start", index=index, total=len(stocks), code=code)
        docs, errors = build_docs_for_stock(ak, stock, fetched_at)
        fetched_stock_count += 1
        if errors:
            error_stock_count += 1
        if not docs:
            skipped_empty_count += 1
            continue

        if args.mode == "range":
            assert start_date is not None and end_date is not None
            docs = filter_range_docs(docs, start_date, end_date)
        else:
            missing_quarter_set = set(missing_quarters)
            refresh_quarter_set = latest_doc_quarters(docs, refresh_quarters)
            docs = filter_docs_by_quarters(docs, missing_quarter_set | refresh_quarter_set)
            missing_quarter_count += len(missing_quarter_set)
            refreshed_quarter_count += len(refresh_quarter_set - missing_quarter_set)

        matched_doc_count += len(docs)
        if args.dry_run:
            dry_run_examples.extend(docs[: max(0, 5 - len(dry_run_examples))])
        else:
            pending_docs.extend(docs)
            if len(pending_docs) >= args.batch_size:
                upsert_frame(db, COLLECTION_NAME, pending_docs, key_columns=("code", "statDate", "source"))
                pending_docs.clear()
                batch_count += 1

        log_event(
            "info",
            "akshare_quarterly_stock_done",
            index=index,
            total=len(stocks),
            code=code,
            docs=len(docs),
            missing_quarters=[format_quarter(year, quarter) for year, quarter in missing_quarters],
        )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if not args.dry_run:
        if pending_docs:
            upsert_frame(db, COLLECTION_NAME, pending_docs, key_columns=("code", "statDate", "source"))
            batch_count += 1

    summary = {
        "mode": args.mode,
        "collection": COLLECTION_NAME,
        "source": SOURCE,
        "dry_run": args.dry_run,
        "stock_count": len(stocks),
        "fetched_stock_count": fetched_stock_count,
        "skipped_up_to_date_count": skipped_up_to_date_count,
        "skipped_no_history_count": skipped_no_history_count,
        "skipped_large_gap_count": skipped_large_gap_count,
        "skipped_empty_count": skipped_empty_count,
        "error_stock_count": error_stock_count,
        "matched_doc_count": matched_doc_count,
        "missing_quarter_count": missing_quarter_count,
        "refreshed_quarter_count": refreshed_quarter_count,
        "batch_count": batch_count,
        "dry_run_examples": dry_run_examples[:5],
        "skipped_gap_examples": skipped_gap_examples[:50],
    }
    if args.mode == "incremental":
        summary["today"] = today
        summary["target_quarter"] = format_quarter(target_year, target_quarter) if target_year is not None and target_quarter is not None else None
        summary["refresh_quarters"] = refresh_quarters
        summary["max_missing_quarters"] = args.max_missing_quarters
        summary["backfill_new_stocks"] = args.backfill_new_stocks

    log_event(
        "info",
        "akshare_quarterly_fetch_done",
        **{k: v for k, v in summary.items() if k not in {"dry_run_examples", "skipped_gap_examples"}},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
