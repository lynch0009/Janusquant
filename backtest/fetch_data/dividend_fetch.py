from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import baostock as bs
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.db import DuckDBConfig, quote_identifier, upsert_frame
from backtest.fetch_data.baostock_utils import (
    BaostockQueryError,
    fetch_query_dataframe,
    login_with_retry,
    safe_logout,
)
from backtest.fetch_data.dividend_event_key import (
    DATE_FIELDS,
    FLOAT_FIELDS,
    SOURCE,
    YEAR_TYPE,
    build_dividend_event_key,
    build_legacy_dividend_event_key,
)
from backtest.utils import is_index_code
from backtest.utils.log import log_event


COLLECTION_NAME = "A_stock_market_dividend_data"
BASIC_INFO_COLLECTION = "A_stock_market_basic_info"
DEFAULT_START_YEAR = 2010
GROUP_MAX_DATE_FIELDS = {
    "maxDividPreNoticeDate": "dividPreNoticeDate",
    "maxDividAgmPumDate": "dividAgmPumDate",
    "maxDividPlanAnnounceDate": "dividPlanAnnounceDate",
    "maxDividPlanDate": "dividPlanDate",
    "maxDividRegistDate": "dividRegistDate",
    "maxDividOperateDate": "dividOperateDate",
    "maxDividPayDate": "dividPayDate",
    "maxDividStockMarketDate": "dividStockMarketDate",
}


@dataclass
class SyncSummary:
    mode: str
    year_type: str
    eligible_code_count: int
    processed_code_count: int
    target_year_count: int
    fetched_year_count: int
    empty_year_count: int
    failed_year_count: int
    inserted_doc_count: int
    upserted_doc_count: int
    deleted_stale_doc_count: int
    insert_batch_count: int
    update_batch_count: int
    retried_query_count: int


@dataclass(frozen=True)
class StockQueryWindow:
    code: str
    ipo_date: datetime | None
    out_date: datetime | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Baostock dividend data into DuckDB.")
    parser.add_argument("--mode", choices=["full", "update"], required=True, default="update")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=datetime.now().year)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--relogin-delay-seconds", type=float, default=60)
    parser.add_argument("--progress-every-codes", type=int, default=20)
    return parser.parse_args()


def ensure_indexes(db: DuckDBConfig) -> list[str]:
    created: list[str] = []
    index_statements = [
        (f"idx__{COLLECTION_NAME}__code_1__dividOperateDate_1", "code, dividOperateDate"),
        (f"idx__{COLLECTION_NAME}__code_1__dividRegistDate_1", "code, dividRegistDate"),
        (f"idx__{COLLECTION_NAME}__code_1__dividPayDate_1", "code, dividPayDate"),
        (f"idx__{COLLECTION_NAME}__source_1__yearType_1__code_1__eventKey_1", "source, yearType, code, eventKey"),
    ]
    for name, columns in index_statements:
        try:
            db.execute(f"create index if not exists {quote_identifier(name)} on {quote_identifier(COLLECTION_NAME)}({columns})")
            created.append(name)
        except Exception as exc:  # noqa: BLE001 - index creation must not block data sync.
            log_event("warning", "duckdb_dividend_index_create_failed", index=name, error=exc)
    return created


def is_sh_sz_stock(code: str) -> bool:
    normalized = str(code).strip().lower()
    return normalized.startswith("sh.") or normalized.startswith("sz.")


def load_stock_query_windows(db: DuckDBConfig) -> list[StockQueryWindow]:
    frame = db.fetch_df(
        f"""
        select code, ipoDate, outDate
        from {quote_identifier(BASIC_INFO_COLLECTION)}
        order by code
        """
    )
    rows: list[StockQueryWindow] = []
    for doc in frame.to_dict("records"):
        code = str(doc.get("code", "")).strip()
        if not code or not is_sh_sz_stock(code) or is_index_code(code):
            continue
        rows.append(
            StockQueryWindow(
                code=code,
                ipo_date=doc.get("ipoDate"),
                out_date=doc.get("outDate"),
            )
        )
    return rows


def resolve_year_window(
    item: StockQueryWindow,
    *,
    global_start_year: int,
    global_end_year: int,
    existing_last_year: int | None = None,
    resume_next_year: bool,
) -> tuple[int, int] | None:
    start_year = global_start_year
    if item.ipo_date is not None:
        start_year = max(start_year, item.ipo_date.year)

    end_year = global_end_year
    if item.out_date is not None:
        end_year = min(end_year, item.out_date.year)

    if resume_next_year and existing_last_year is not None:
        start_year = max(start_year, existing_last_year + 1)
    elif existing_last_year is not None:
        start_year = max(start_year, existing_last_year)

    if start_year > end_year:
        return None
    return start_year, end_year


def load_last_year_map(db: DuckDBConfig) -> dict[str, int]:
    available_columns = {
        row[0]
        for row in db.connection.execute(f"describe {quote_identifier(COLLECTION_NAME)}").fetchall()
    }
    fields = [field for field in GROUP_MAX_DATE_FIELDS.values() if field in available_columns]
    if not fields:
        return {}
    frame = db.fetch_df(
        f"""
        select code, {", ".join(quote_identifier(field) for field in fields)}
        from {quote_identifier(COLLECTION_NAME)}
        """
    )
    last_year_map: dict[str, int] = {}
    for code, group in frame.groupby("code"):
        code = str(code).strip()
        if not code:
            continue
        valid_dates = []
        for field in fields:
            valid_dates.extend(pd.to_datetime(group[field], errors="coerce").dropna().tolist())
        if not valid_dates:
            continue
        last_year_map[code] = pd.Timestamp(max(valid_dates)).year
    return last_year_map


def fetch_dividend_frame(
    code: str,
    year: int,
    *,
    max_retries: int,
    relogin_delay_seconds: float,
) -> tuple[pd.DataFrame, str | None, int]:
    try:
        frame, retries_used = fetch_query_dataframe(
            bs.query_dividend_data,
            code=code,
            year=str(year),
            yearType=YEAR_TYPE,
            retry_times=max(1, max_retries),
            retry_sleep_seconds=relogin_delay_seconds,
            relogin_sleep_seconds=relogin_delay_seconds,
            quiet=True,
            progress=lambda message: log_event("info", "baostock retry", message=message),
            context=f"{code} {year} query_dividend_data",
        )
        return frame, None, retries_used
    except BaostockQueryError as exc:
        return pd.DataFrame(), str(exc), exc.retries_used


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_datetime(value: Any) -> datetime | None:
    text = normalize_text(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def normalize_float(value: Any) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    parsed = pd.to_numeric(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)


def build_doc_dedup_key(doc: dict[str, Any]) -> tuple[Any, ...]:
    return doc.get("source"), doc.get("yearType"), doc.get("eventKey")


def build_upsert_filter(doc: dict[str, Any]) -> dict[str, Any]:
    event_key = doc["eventKey"]
    legacy_event_key = build_legacy_dividend_event_key(doc)
    filter_doc: dict[str, Any] = {
        "code": doc["code"],
        "source": doc["source"],
        "yearType": doc["yearType"],
    }
    if legacy_event_key == event_key:
        filter_doc["eventKey"] = event_key
        return filter_doc
    filter_doc["$or"] = [
        {"eventKey": event_key},
        {"eventKey": legacy_event_key, "dividOperateDate": doc.get("dividOperateDate")},
    ]
    return filter_doc


def frame_to_docs(frame: pd.DataFrame, *, query_year: int, fetched_at: datetime) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    docs: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for row in frame.to_dict("records"):
        doc: dict[str, Any] = {
            "code": normalize_text(row.get("code")),
            "dividCashStock": normalize_text(row.get("dividCashStock")),
            "source": SOURCE,
            "yearType": YEAR_TYPE,
            "queryYear": int(query_year),
            "fetchedAt": fetched_at,
        }
        for field in DATE_FIELDS:
            doc[field] = normalize_datetime(row.get(field))
        for field in FLOAT_FIELDS:
            doc[field] = normalize_float(row.get(field))
        doc["eventKey"] = build_dividend_event_key(doc)

        dedup_key = build_doc_dedup_key(doc)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        docs.append(doc)
    return docs


def flush_update_batch(db: DuckDBConfig, docs: list[dict[str, Any]]) -> tuple[int, int]:
    if not docs:
        return 0, 0
    affected = len(docs)
    upsert_frame(db, COLLECTION_NAME, docs, key_columns=("source", "yearType", "code", "eventKey"))
    docs.clear()
    return affected, 1


def run_full_sync(
    db: DuckDBConfig,
    stocks: list[StockQueryWindow],
    existing_last_year_map: dict[str, int],
    *,
    start_year: int,
    end_year: int,
    batch_size: int,
    max_retries: int,
    relogin_delay_seconds: float,
    progress_every_codes: int,
) -> SyncSummary:
    pending_docs: list[dict[str, Any]] = []
    processed_code_count = 0
    target_year_count = 0
    fetched_year_count = 0
    empty_year_count = 0
    failed_year_count = 0
    upserted_doc_count = 0
    update_batch_count = 0
    retried_query_count = 0
    fetched_at = datetime.now()

    login_with_retry(
        retry_times=max(1, max_retries),
        retry_sleep_seconds=relogin_delay_seconds,
        quiet=True,
        progress=lambda message: log_event("info", "baostock retry", message=message),
    )
    log_event("info", "baostock login success")
    try:
        total_codes = len(stocks)
        for code_index, item in enumerate(stocks, start=1):
            code = item.code
            year_window = resolve_year_window(
                item,
                global_start_year=start_year,
                global_end_year=end_year,
                existing_last_year=existing_last_year_map.get(code),
                resume_next_year=True,
            )
            if year_window is None:
                continue

            code_start_year, code_end_year = year_window
            processed_code_count += 1
            if code_index == 1 or code_index % max(1, progress_every_codes) == 0 or code_index == total_codes:
                log_event(
                    "info",
                    f"full sync progress {code_index}/{total_codes}: code={code}, "
                    f"start_year={code_start_year}, end_year={code_end_year}"
                )

            for year in range(code_start_year, code_end_year + 1):
                target_year_count += 1
                frame, error_msg, retries_used = fetch_dividend_frame(
                    code,
                    year,
                    max_retries=max(1, max_retries),
                    relogin_delay_seconds=relogin_delay_seconds,
                )
                retried_query_count += retries_used

                if error_msg is not None:
                    failed_year_count += 1
                    log_event("warning", f"full sync query failed: code={code}, year={year}, error={error_msg}")
                    continue

                if frame.empty:
                    empty_year_count += 1
                    continue

                fetched_year_count += 1
                pending_docs.extend(frame_to_docs(frame, query_year=year, fetched_at=fetched_at))

                if len(pending_docs) >= max(1, batch_size):
                    affected = len(pending_docs)
                    _affected, batches = flush_update_batch(db, pending_docs)
                    upserted_doc_count += affected
                    update_batch_count += batches
                    log_event(
                        "info",
                        f"full sync flush #{update_batch_count}: upserted_docs={upserted_doc_count}, "
                        f"fetched_years={fetched_year_count}, failed_years={failed_year_count}"
                    )

        affected = len(pending_docs)
        _affected, batches = flush_update_batch(db, pending_docs)
        upserted_doc_count += affected
        update_batch_count += batches
    finally:
        safe_logout()
        log_event("info", "baostock logout complete")

    return SyncSummary(
        mode="full",
        year_type=YEAR_TYPE,
        eligible_code_count=len(stocks),
        processed_code_count=processed_code_count,
        target_year_count=target_year_count,
        fetched_year_count=fetched_year_count,
        empty_year_count=empty_year_count,
        failed_year_count=failed_year_count,
        inserted_doc_count=0,
        upserted_doc_count=upserted_doc_count,
        deleted_stale_doc_count=0,
        insert_batch_count=0,
        update_batch_count=update_batch_count,
        retried_query_count=retried_query_count,
    )


def run_update_sync(
    db: DuckDBConfig,
    stocks: list[StockQueryWindow],
    last_year_map: dict[str, int],
    *,
    start_year: int,
    end_year: int,
    batch_size: int,
    max_retries: int,
    relogin_delay_seconds: float,
    progress_every_codes: int,
) -> SyncSummary:
    processed_code_count = 0
    target_year_count = 0
    fetched_year_count = 0
    empty_year_count = 0
    failed_year_count = 0
    upserted_doc_count = 0
    update_batch_count = 0
    retried_query_count = 0
    fetched_at = datetime.now()

    login_with_retry(
        retry_times=max(1, max_retries),
        retry_sleep_seconds=relogin_delay_seconds,
        quiet=True,
        progress=lambda message: log_event("info", "baostock retry", message=message),
    )
    log_event("info", "baostock login success")
    try:
        total_codes = len(stocks)
        for code_index, item in enumerate(stocks, start=1):
            code = item.code
            year_window = resolve_year_window(
                item,
                global_start_year=start_year,
                global_end_year=end_year,
                existing_last_year=last_year_map.get(code),
                resume_next_year=False,
            )
            if year_window is None:
                continue

            code_start_year, code_end_year = year_window
            processed_code_count += 1

            if code_index == 1 or code_index % max(1, progress_every_codes) == 0 or code_index == total_codes:
                log_event(
                    "info",
                    f"update sync progress {code_index}/{total_codes}: code={code}, "
                    f"start_year={code_start_year}, end_year={code_end_year}"
                )

            pending_docs: list[dict[str, Any]] = []
            for year in range(code_start_year, code_end_year + 1):
                target_year_count += 1
                frame, error_msg, retries_used = fetch_dividend_frame(
                    code,
                    year,
                    max_retries=max(1, max_retries),
                    relogin_delay_seconds=relogin_delay_seconds,
                )
                retried_query_count += retries_used

                if error_msg is not None:
                    failed_year_count += 1
                    log_event("warning", f"update sync query failed: code={code}, year={year}, error={error_msg}")
                    continue

                docs = frame_to_docs(frame, query_year=year, fetched_at=fetched_at)
                if frame.empty:
                    empty_year_count += 1
                else:
                    fetched_year_count += 1

                pending_docs.extend(docs)

            if pending_docs:
                for start in range(0, len(pending_docs), max(1, batch_size)):
                    batch = pending_docs[start:start + max(1, batch_size)]
                    affected = len(batch)
                    _affected, batches = flush_update_batch(db, batch)
                    upserted_doc_count += affected
                    update_batch_count += batches
    finally:
        safe_logout()
        log_event("info", "baostock logout complete")

    return SyncSummary(
        mode="update",
        year_type=YEAR_TYPE,
        eligible_code_count=len(stocks),
        processed_code_count=processed_code_count,
        target_year_count=target_year_count,
        fetched_year_count=fetched_year_count,
        empty_year_count=empty_year_count,
        failed_year_count=failed_year_count,
        inserted_doc_count=0,
        upserted_doc_count=upserted_doc_count,
        deleted_stale_doc_count=0,
        insert_batch_count=0,
        update_batch_count=update_batch_count,
        retried_query_count=retried_query_count,
    )


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start-year cannot be greater than end-year")

    db = DuckDBConfig()

    created_indexes = ensure_indexes(db)
    stocks = load_stock_query_windows(db)
    existing_last_year_map = load_last_year_map(db)
    log_event(
        "info",
        f"dividend sync start: mode={args.mode}, codes={len(stocks)}, year_type={YEAR_TYPE}, "
        f"start_year={args.start_year}, end_year={args.end_year}"
    )

    if args.mode == "full":
        summary = run_full_sync(
            db,
            stocks,
            existing_last_year_map,
            start_year=args.start_year,
            end_year=args.end_year,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            relogin_delay_seconds=args.relogin_delay_seconds,
            progress_every_codes=args.progress_every_codes,
        )
    else:
        summary = run_update_sync(
            db,
            stocks,
            existing_last_year_map,
            start_year=args.start_year,
            end_year=args.end_year,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            relogin_delay_seconds=args.relogin_delay_seconds,
            progress_every_codes=args.progress_every_codes,
        )

    payload = {
        "collection": COLLECTION_NAME,
        "created_or_confirmed_indexes": created_indexes,
        "summary": asdict(summary),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
