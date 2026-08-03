from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.db import DuckDBConfig, quote_identifier, upsert_frame
from backtest.utils import normalize_internal_code
from backtest.feature.minervini_fundamental_feature import (
    FEATURE_FIELDS,
    FEATURE_VERSION,
    SOURCE_COLLECTION,
    SOURCE_FIELDS,
    TARGET_COLLECTION,
    build_minervini_fundamental_features,
)
from backtest.utils.datetime_utils import to_pydatetime
from backtest.utils.log import log_event


DEFAULT_BATCH_SIZE = 300
DEFAULT_LOOKBACK_QUARTERS = 8
WARMUP_QUARTERS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Minervini fundamental feature records from AkShare quarterly facts.")
    parser.add_argument("--mode", choices=["full", "incremental"], default="incremental")
    parser.add_argument("--start-date", help="Target statDate start for full mode, e.g. 2020-01-01.")
    parser.add_argument("--end-date", help="Target statDate end for full mode, e.g. 2026-03-31.")
    parser.add_argument("--codes", help="Comma separated stock codes, e.g. sz.300308,300502.")
    parser.add_argument("--lookback-quarters", type=int, default=DEFAULT_LOOKBACK_QUARTERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, help="Only process first N codes after sorting.")
    parser.add_argument("--dry-run", action="store_true", help="Build features but do not write DuckDB.")
    return parser.parse_args()


def parse_date(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    normalized = to_pydatetime(parsed)
    if isinstance(normalized, datetime):
        return datetime(normalized.year, normalized.month, normalized.day)
    return None


def split_codes(raw_codes: str | None) -> list[str]:
    if not raw_codes:
        return []
    codes: list[str] = []
    for item in str(raw_codes).split(","):
        item = item.strip()
        if item:
            codes.append(normalize_internal_code(item))
    return sorted(set(codes))


def batched(values: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("--batch-size must be positive")
    return [values[index : index + size] for index in range(0, len(values), size)]


def date_range_query(start_date: datetime | None, end_date: datetime | None) -> dict[str, Any]:
    stat_query: dict[str, datetime] = {}
    if start_date is not None:
        stat_query["$gte"] = start_date
    if end_date is not None:
        stat_query["$lte"] = end_date
    return {"statDate": stat_query} if stat_query else {}


def warmup_start(start_date: datetime | None) -> datetime | None:
    if start_date is None:
        return None
    return to_pydatetime(pd.Timestamp(start_date) - pd.DateOffset(months=3 * WARMUP_QUARTERS))


def latest_stat_dates(db: DuckDBConfig, count: int) -> list[datetime]:
    if count <= 0:
        raise ValueError("--lookback-quarters must be positive")
    values = [
        row[0]
        for row in db.connection.execute(
            f"select distinct statDate from {quote_identifier(SOURCE_COLLECTION)} where statDate is not null"
        ).fetchall()
    ]
    dates = sorted([to_pydatetime(pd.Timestamp(value)) for value in values if pd.notna(value)], reverse=True)
    return dates[:count]


def _where_from_query(read_query: dict[str, Any]) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for field, condition in read_query.items():
        if isinstance(condition, dict):
            if "$gte" in condition:
                clauses.append(f"{quote_identifier(field)} >= ?")
                params.append(condition["$gte"])
            if "$lte" in condition:
                clauses.append(f"{quote_identifier(field)} <= ?")
                params.append(condition["$lte"])
            if "$in" in condition:
                values = list(condition["$in"])
                if not values:
                    clauses.append("1 = 0")
                else:
                    clauses.append(f"{quote_identifier(field)} in ({', '.join('?' for _ in values)})")
                    params.extend(values)
        else:
            clauses.append(f"{quote_identifier(field)} = ?")
            params.append(condition)
    return clauses, params


def load_codes(db: DuckDBConfig, *, requested_codes: list[str], read_query: dict[str, Any], limit: int | None) -> list[str]:
    if requested_codes:
        codes = requested_codes
    else:
        clauses, params = _where_from_query(read_query)
        where_sql = f" where {' and '.join(clauses)}" if clauses else ""
        codes = [
            str(row[0])
            for row in db.connection.execute(
                f"select distinct code from {quote_identifier(SOURCE_COLLECTION)}{where_sql} order by code",
                params,
            ).fetchall()
        ]
    if limit is not None:
        codes = codes[: max(0, limit)]
    return codes


def load_source_frame(db: DuckDBConfig, *, codes: list[str], read_query: dict[str, Any]) -> pd.DataFrame:
    if not codes:
        return pd.DataFrame()
    query = dict(read_query)
    query["code"] = {"$in": codes}
    clauses, params = _where_from_query(query)
    fields = [quote_identifier(field) for field in SOURCE_FIELDS]
    return db.fetch_df(
        f"""
        select {', '.join(fields)}
        from {quote_identifier(SOURCE_COLLECTION)}
        where {' and '.join(clauses)}
        order by code, statDate
        """,
        params,
    )


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [item for item in value if item is not None]
    if value is pd.NA:
        return None
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value) and not isinstance(value, list):
        return None
    return value


def frame_to_docs(frame: pd.DataFrame) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        doc: dict[str, Any] = {}
        for key, value in row.items():
            if key not in FEATURE_FIELDS:
                continue
            cleaned = clean_value(value)
            if cleaned is not None:
                doc[key] = cleaned
        docs.append(doc)
    return docs


def summarize_features(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "feature_rows": 0,
            "feature_codes": 0,
            "missing_core_rows": 0,
            "yoy_extreme_rows": 0,
        }
    return {
        "feature_rows": len(frame),
        "feature_codes": frame["code"].nunique() if "code" in frame.columns else 0,
        "missing_core_rows": int(frame["missing_core_fields"].map(bool).sum()) if "missing_core_fields" in frame.columns else 0,
        "yoy_extreme_rows": int(frame["yoy_extreme_flag"].fillna(False).sum()) if "yoy_extreme_flag" in frame.columns else 0,
    }


def resolve_mode_dates(db: DuckDBConfig, args: argparse.Namespace) -> tuple[dict[str, Any], datetime | None, datetime | None, list[datetime] | None]:
    if args.mode == "full":
        target_start = parse_date(args.start_date)
        target_end = parse_date(args.end_date)
        read_start = warmup_start(target_start)
        read_query = date_range_query(read_start, target_end)
        return read_query, target_start, target_end, None

    target_dates = latest_stat_dates(db, args.lookback_quarters)
    read_dates = latest_stat_dates(db, args.lookback_quarters + WARMUP_QUARTERS)
    read_query = {"statDate": {"$in": read_dates}}
    return read_query, None, None, target_dates


def run(args: argparse.Namespace) -> None:
    started_at = datetime.now()
    db = DuckDBConfig()

    requested_codes = split_codes(args.codes)
    read_query, target_start, target_end, target_stat_dates = resolve_mode_dates(db, args)
    codes = load_codes(db, requested_codes=requested_codes, read_query=read_query, limit=args.limit)

    log_event(
        "info",
        "minervini_fundamental_feature_build_start",
        mode=args.mode,
        codes=len(codes),
        dry_run=args.dry_run,
        target_start=target_start,
        target_end=target_end,
        target_stat_dates=len(target_stat_dates or []),
        feature_version=FEATURE_VERSION,
    )

    total_source_rows = 0
    total_valid_notice_rows = 0
    total_feature_rows = 0
    total_missing_core_rows = 0
    total_yoy_extreme_rows = 0
    total_write_batches = 0
    computed_at = datetime.now()

    code_batches = batched(codes, args.batch_size)
    for batch_number, code_batch in enumerate(code_batches, start=1):
        source_frame = load_source_frame(db, codes=code_batch, read_query=read_query)
        total_source_rows += len(source_frame)
        valid_notice_rows = int(pd.to_datetime(source_frame.get("noticeDate"), errors="coerce").notna().sum()) if not source_frame.empty else 0
        total_valid_notice_rows += valid_notice_rows

        feature_frame = build_minervini_fundamental_features(
            source_frame,
            computed_at=computed_at,
            feature_version=FEATURE_VERSION,
            write_start_date=target_start,
            write_end_date=target_end,
            target_stat_dates=target_stat_dates,
        )
        stats = summarize_features(feature_frame)
        total_feature_rows += int(stats["feature_rows"])
        total_missing_core_rows += int(stats["missing_core_rows"])
        total_yoy_extreme_rows += int(stats["yoy_extreme_rows"])

        if not args.dry_run and not feature_frame.empty:
            docs = frame_to_docs(feature_frame)
            if docs:
                upsert_frame(db, TARGET_COLLECTION, docs, key_columns=("code", "statDate", "featureVersion"))
                total_write_batches += 1

        log_event(
            "info",
            "minervini_fundamental_feature_batch_done",
            batch=f"{batch_number}/{len(code_batches)}",
            batch_codes=len(code_batch),
            source_rows=len(source_frame),
            valid_notice_rows=valid_notice_rows,
            feature_rows=stats["feature_rows"],
            missing_core_rows=stats["missing_core_rows"],
            yoy_extreme_rows=stats["yoy_extreme_rows"],
            pending_writes=0,
        )

    elapsed = (datetime.now() - started_at).total_seconds()
    log_event(
        "info",
        "minervini_fundamental_feature_build_done",
        mode=args.mode,
        codes=len(codes),
        source_rows=total_source_rows,
        valid_notice_rows=total_valid_notice_rows,
        skipped_missing_notice_rows=total_source_rows - total_valid_notice_rows,
        feature_rows=total_feature_rows,
        missing_core_rows=total_missing_core_rows,
        yoy_extreme_rows=total_yoy_extreme_rows,
        write_batches=0 if args.dry_run else total_write_batches,
        dry_run=args.dry_run,
        elapsed=f"{elapsed:.2f}s",
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
