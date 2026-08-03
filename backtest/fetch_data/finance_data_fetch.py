from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import reduce
from numbers import Number
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.db import DuckDBConfig, quote_identifier, upsert_frame
from backtest.fetch_data.baostock_utils import (
    DEFAULT_RETRY_SLEEP_SECONDS,
    DEFAULT_RETRY_TIMES,
    fetch_query_dataframe,
    login_with_retry,
    safe_logout,
)
from backtest.fetch_data.stock_universe import load_stock_windows_duckdb
from backtest.utils import (
    format_quarter,
    iter_quarter_pairs,
    iter_quarters,
    next_quarter,
    quarter_end,
    quarter_from_date,
    resolve_incremental_target_quarter,
)
from backtest.utils.log import log_event


FINANCE_COLLECTION = "A_stock_market_finance_data"
BASIC_INFO_COLLECTION = "A_stock_market_basic_info"
KEY_COLUMNS = ["code", "pubDate", "statDate"]
INT_FIELDS = {"totalShare", "liqaShare"}
PROFIT_INTERFACE = "query_profit_data"
QUERY_INTERFACES = (
    PROFIT_INTERFACE,
    "query_operation_data",
    "query_growth_data",
    "query_balance_data",
    "query_cash_flow_data",
    "query_dupont_data",
)
RETRY_TIMES = DEFAULT_RETRY_TIMES
RETRY_SLEEP_SECONDS = DEFAULT_RETRY_SLEEP_SECONDS
DEFAULT_MAX_MISSING_QUARTERS = 3
DEFAULT_QUERY_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_QUERY_WORKERS = 4


def parse_date(value: str) -> datetime:
    dt = pd.to_datetime(value)
    return datetime(dt.year, dt.month, dt.day)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch finance data from Baostock and write to DuckDB.")
    parser.add_argument("--mode", choices=["range", "incremental"], default="incremental")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--today")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--max-missing-quarters",
        type=int,
        default=DEFAULT_MAX_MISSING_QUARTERS,
        help="Incremental mode: skip stocks whose missing quarter count is larger than this value. -1 disables the limit.",
    )
    parser.add_argument(
        "--query-timeout-seconds",
        type=float,
        default=DEFAULT_QUERY_TIMEOUT_SECONDS,
        help="Timeout for one Baostock interface query subprocess.",
    )
    parser.add_argument(
        "--max-query-workers",
        type=int,
        default=DEFAULT_MAX_QUERY_WORKERS,
        help="Maximum concurrent Baostock interface query subprocesses.",
    )
    return parser.parse_args()


def load_existing_stat_dates_map(db: DuckDBConfig, codes: list[str]) -> dict[str, set[datetime]]:
    if not codes:
        return {}
    frame = db.fetch_df(
        f"""
        select code, statDate
        from {quote_identifier(FINANCE_COLLECTION)}
        where code in ({", ".join("?" for _ in codes)}) and statDate is not null
        """,
        codes,
    )
    result: dict[str, set[datetime]] = {}
    for code, group in frame.groupby("code"):
        code = str(code).strip()
        if not code:
            continue
        result[code] = {
            parse_date(value)
            for value in group["statDate"].tolist()
            if value is not None
        }
    return result


def _baostock_query_worker(
    interface_name: str,
    code: str,
    year: int,
    quarter: int,
    result_queue,
) -> None:
    try:
        import baostock as bs

        login_with_retry(retry_times=1, retry_sleep_seconds=0.0, quiet=True, progress=None)
        query_func = getattr(bs, interface_name)
        frame, _retries_used = fetch_query_dataframe(
            query_func,
            code=code,
            year=year,
            quarter=quarter,
            retry_times=1,
            retry_sleep_seconds=0.0,
            quiet=True,
            progress=None,
            context=f"{code} {year}Q{quarter} {interface_name}",
        )
        result_queue.put(
            {
                "status": "ok",
                "columns": list(frame.columns),
                "records": frame.to_dict("records"),
            }
        )
    except Exception as exc:  # noqa: BLE001 - subprocess boundary must serialize all failures.
        result_queue.put(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        safe_logout(quiet=True)


def _run_query_attempt(
    interface_name: str,
    code: str,
    year: int,
    quarter: int,
    *,
    timeout_seconds: float,
    worker_func=_baostock_query_worker,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=worker_func,
        args=(interface_name, code, year, quarter, result_queue),
    )
    process.daemon = True
    process.start()
    process.join(max(0.001, float(timeout_seconds)))
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(5)
        return {"status": "timeout"}

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {"status": "error", "error_type": "NoResult", "error": f"exitcode={process.exitcode}"}


def _frame_from_query_payload(payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(payload.get("records", []), columns=payload.get("columns", []))


def fetch_one_interface(
    interface_name: str,
    code: str,
    year: int,
    quarter: int,
    *,
    query_timeout_seconds: float,
) -> tuple[pd.DataFrame, bool]:
    label = f"{code} {year}Q{quarter} {interface_name}"
    for attempt in range(1, RETRY_TIMES + 1):
        start_time = time.monotonic()
        log_event(
            "info",
            "baostock_finance_query_start",
            code=code,
            year=year,
            quarter=quarter,
            interface=interface_name,
            attempt=attempt,
            timeout_seconds=query_timeout_seconds,
        )
        payload = _run_query_attempt(
            interface_name,
            code,
            year,
            quarter,
            timeout_seconds=query_timeout_seconds,
        )
        elapsed_seconds = time.monotonic() - start_time
        status = payload.get("status")
        if status == "ok":
            frame = _frame_from_query_payload(payload)
            event = "baostock_finance_query_empty" if frame.empty else "baostock_finance_query_done"
            log_event(
                "info",
                event,
                code=code,
                year=year,
                quarter=quarter,
                interface=interface_name,
                attempt=attempt,
                rows=len(frame),
                elapsed_seconds=elapsed_seconds,
            )
            return frame, True

        if status == "timeout":
            log_event(
                "warning",
                "baostock_finance_query_timeout",
                code=code,
                year=year,
                quarter=quarter,
                interface=interface_name,
                attempt=attempt,
                elapsed_seconds=elapsed_seconds,
                timeout_seconds=query_timeout_seconds,
            )
        else:
            log_event(
                "warning",
                "baostock_finance_query_error",
                code=code,
                year=year,
                quarter=quarter,
                interface=interface_name,
                attempt=attempt,
                elapsed_seconds=elapsed_seconds,
                error_type=payload.get("error_type"),
                error=payload.get("error"),
            )

        if attempt < RETRY_TIMES:
            time.sleep(max(0.0, RETRY_SLEEP_SECONDS))

    log_event(
        "error",
        "baostock_finance_query_failed",
        code=code,
        year=year,
        quarter=quarter,
        interface=interface_name,
        retry_times=RETRY_TIMES,
        label=label,
    )
    return pd.DataFrame(), False


def fetch_one_quarter(
    code: str,
    year: int,
    quarter: int,
    *,
    max_query_workers: int,
    query_timeout_seconds: float,
) -> pd.DataFrame:
    profit_frame, profit_ok = fetch_one_interface(
        PROFIT_INTERFACE,
        code,
        year,
        quarter,
        query_timeout_seconds=query_timeout_seconds,
    )
    if not profit_ok or profit_frame.empty:
        return pd.DataFrame()

    frames = [profit_frame.drop_duplicates(subset=KEY_COLUMNS, keep="last")]
    other_interfaces = [item for item in QUERY_INTERFACES if item != PROFIT_INTERFACE]
    with ThreadPoolExecutor(max_workers=max(1, int(max_query_workers))) as executor:
        futures = {
            executor.submit(
                fetch_one_interface,
                interface_name,
                code,
                year,
                quarter,
                query_timeout_seconds=query_timeout_seconds,
            ): interface_name
            for interface_name in other_interfaces
        }
        for future in as_completed(futures):
            interface_name = futures[future]
            try:
                frame, ok = future.result()
            except Exception as exc:  # noqa: BLE001 - keep quarter fetch resilient.
                log_event(
                    "error",
                    "baostock_finance_query_future_error",
                    code=code,
                    year=year,
                    quarter=quarter,
                    interface=interface_name,
                    error_type=type(exc).__name__,
                    error=exc,
                )
                continue
            if ok and not frame.empty:
                frames.append(frame.drop_duplicates(subset=KEY_COLUMNS, keep="last"))

    if not frames:
        return pd.DataFrame()

    return reduce(
        lambda left, right: pd.merge(left, right, on=KEY_COLUMNS, how="outer", validate="one_to_one"),
        frames,
    )


def normalize_doc(row: dict[str, Any]) -> dict[str, Any]:
    doc = dict(row)
    doc["code"] = str(doc["code"])
    doc["pubDate"] = pd.to_datetime(doc["pubDate"]).to_pydatetime()
    doc["statDate"] = pd.to_datetime(doc["statDate"]).to_pydatetime()

    for field in INT_FIELDS:
        if field in doc:
            value = doc[field]
            if value not in (None, ""):
                try:
                    number = float(value) if not isinstance(value, Number) else float(value)
                    if abs(number - round(number)) <= 1e-9:
                        doc[field] = int(round(number))
                except (TypeError, ValueError):
                    pass

    for field, value in list(doc.items()):
        if pd.isna(value):
            doc[field] = ""
    return doc


def build_missing_quarters(
    existing_stat_dates: set[datetime],
    ipo_date: datetime | None,
    target_year: int,
    target_quarter: int,
) -> list[tuple[int, int]]:
    target_stat_date = quarter_end(target_year, target_quarter)
    existing_quarters = {quarter_from_date(stat_date) for stat_date in existing_stat_dates}
    if quarter_from_date(target_stat_date) in existing_quarters:
        return []

    if existing_stat_dates:
        start_year, start_quarter = next_quarter(*quarter_from_date(max(existing_stat_dates)))
    elif ipo_date is not None:
        start_year, start_quarter = quarter_from_date(ipo_date)
    else:
        return []

    if (start_year, start_quarter) > (target_year, target_quarter):
        return []

    return list(iter_quarter_pairs(start_year, start_quarter, target_year, target_quarter))


def exceeds_incremental_gap_limit(missing_quarters: list[tuple[int, int]], max_missing_quarters: int) -> bool:
    return max_missing_quarters >= 0 and len(missing_quarters) > max_missing_quarters


def run_range_mode(args: argparse.Namespace) -> None:
    if not args.start_date or not args.end_date:
        raise ValueError("range mode requires --start-date and --end-date")

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)

    db = DuckDBConfig()

    stocks = load_stock_windows_duckdb(db, start_date=start_date, end_date=end_date)
    quarters = list(iter_quarters(start_date, end_date))

    log_event(
        "info",
        "baostock_finance_fetch_start",
        mode="range",
        collection=FINANCE_COLLECTION,
        start_date=start_date,
        end_date=end_date,
        stock_count=len(stocks),
        quarter_count=len(quarters),
        max_query_workers=args.max_query_workers,
        query_timeout_seconds=args.query_timeout_seconds,
    )

    pending_docs: list[dict[str, Any]] = []
    matched_count = 0
    batch_count = 0
    quarter_count = 0
    code_count = 0

    for stock in stocks:
        code = stock.code
        ipo_date = stock.ipo_date
        out_date = stock.out_date
        code_count += 1
        log_event("info", "baostock_finance_stock_range_start", index=code_count, total=len(stocks), code=code)

        for year, quarter in quarters:
            current_quarter_end = quarter_end(year, quarter)
            if ipo_date is not None and current_quarter_end < ipo_date:
                continue
            if out_date is not None and current_quarter_end > out_date:
                continue

            quarter_count += 1
            quarter_frame = fetch_one_quarter(
                code,
                year,
                quarter,
                max_query_workers=args.max_query_workers,
                query_timeout_seconds=args.query_timeout_seconds,
            )
            if quarter_frame.empty:
                continue

            for row in quarter_frame.to_dict("records"):
                doc = normalize_doc(row)
                if doc["statDate"] < start_date or doc["statDate"] > end_date:
                    continue

                matched_count += 1
                pending_docs.append(doc)

            if len(pending_docs) >= args.batch_size:
                upsert_frame(db, FINANCE_COLLECTION, pending_docs, key_columns=("code", "pubDate", "statDate"))
                pending_docs.clear()
                batch_count += 1

    if pending_docs:
        upsert_frame(db, FINANCE_COLLECTION, pending_docs, key_columns=("code", "pubDate", "statDate"))
        batch_count += 1

    summary = {
        "mode": "range",
        "collection": FINANCE_COLLECTION,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "stock_count": len(stocks),
        "quarter_count": quarter_count,
        "matched_count": matched_count,
        "batch_count": batch_count,
    }
    log_event("info", "baostock_finance_fetch_done", **summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def run_incremental_mode(args: argparse.Namespace) -> None:
    today = parse_date(args.today) if args.today else datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    target_year, target_quarter = resolve_incremental_target_quarter(today)

    db = DuckDBConfig()

    stocks = load_stock_windows_duckdb(db, active_on=today)
    codes = [item.code for item in stocks]
    existing_stat_dates_map = load_existing_stat_dates_map(db, codes)

    log_event(
        "info",
        "baostock_finance_fetch_start",
        mode="incremental",
        collection=FINANCE_COLLECTION,
        today=today,
        target_quarter=format_quarter(target_year, target_quarter),
        active_stock_count=len(stocks),
        max_missing_quarters=args.max_missing_quarters,
        max_query_workers=args.max_query_workers,
        query_timeout_seconds=args.query_timeout_seconds,
    )

    pending_docs: list[dict[str, Any]] = []
    updated_stock_count = 0
    skipped_up_to_date_count = 0
    skipped_gap_count = 0
    fetched_quarter_count = 0
    batch_count = 0
    missing_gap_rows: list[dict[str, Any]] = []

    for index, stock in enumerate(stocks, start=1):
        code = stock.code
        code_name = stock.code_name
        ipo_date = stock.ipo_date
        existing_stat_dates = existing_stat_dates_map.get(code, set())

        missing_quarters = build_missing_quarters(
            existing_stat_dates,
            ipo_date,
            target_year,
            target_quarter,
        )

        if not missing_quarters:
            skipped_up_to_date_count += 1
            log_event(
                "info",
                "baostock_finance_stock_skipped_up_to_date",
                index=index,
                total=len(stocks),
                code=code,
                code_name=code_name,
                last_stat_date=max(existing_stat_dates) if existing_stat_dates else None,
                target_quarter=format_quarter(target_year, target_quarter),
            )
            continue

        if exceeds_incremental_gap_limit(missing_quarters, args.max_missing_quarters):
            skipped_gap_count += 1
            gap_item = {
                "code": code,
                "code_name": code_name,
                "last_stat_date": max(existing_stat_dates).strftime("%Y-%m-%d") if existing_stat_dates else None,
                "missing_quarters": [format_quarter(year, quarter) for year, quarter in missing_quarters],
            }
            missing_gap_rows.append(gap_item)
            log_event(
                "warning",
                "baostock_finance_stock_skipped_large_gap",
                index=index,
                total=len(stocks),
                **gap_item,
            )
            continue

        log_event(
            "info",
            "baostock_finance_stock_update_start",
            index=index,
            total=len(stocks),
            code=code,
            code_name=code_name,
            missing_quarters=[format_quarter(year, quarter) for year, quarter in missing_quarters],
        )

        updated_stock_count += 1
        for year, quarter in missing_quarters:
            quarter_frame = fetch_one_quarter(
                code,
                year,
                quarter,
                max_query_workers=args.max_query_workers,
                query_timeout_seconds=args.query_timeout_seconds,
            )
            fetched_quarter_count += 1
            if quarter_frame.empty:
                continue

            target_stat_date = quarter_end(year, quarter)
            for row in quarter_frame.to_dict("records"):
                doc = normalize_doc(row)
                if doc["statDate"] != target_stat_date:
                    continue
                pending_docs.append(doc)

            if len(pending_docs) >= args.batch_size:
                upsert_frame(db, FINANCE_COLLECTION, pending_docs, key_columns=("code", "pubDate", "statDate"))
                pending_docs.clear()
                batch_count += 1

    if pending_docs:
        upsert_frame(db, FINANCE_COLLECTION, pending_docs, key_columns=("code", "pubDate", "statDate"))
        batch_count += 1

    summary = {
        "mode": "incremental",
        "collection": FINANCE_COLLECTION,
        "today": today.strftime("%Y-%m-%d"),
        "target_quarter": format_quarter(target_year, target_quarter),
        "active_stock_count": len(stocks),
        "updated_stock_count": updated_stock_count,
        "skipped_up_to_date_count": skipped_up_to_date_count,
        "skipped_gap_count": skipped_gap_count,
        "max_missing_quarters": args.max_missing_quarters,
        "max_query_workers": args.max_query_workers,
        "query_timeout_seconds": args.query_timeout_seconds,
        "fetched_quarter_count": fetched_quarter_count,
        "batch_count": batch_count,
        "gap_examples": missing_gap_rows[:50],
    }
    log_event(
        "info",
        "baostock_finance_fetch_done",
        **{k: v for k, v in summary.items() if k != "gap_examples"},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    args = parse_args()

    if args.mode == "incremental":
        run_incremental_mode(args)
        return

    run_range_mode(args)


if __name__ == "__main__":
    main()
