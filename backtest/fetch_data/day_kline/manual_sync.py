from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

import pandas as pd

from backtest.db import DuckDBConfig
from backtest.fetch_data.baostock_utils import fetch_trade_calendar, login_with_retry, safe_logout
from backtest.fetch_data.day_kline_common import build_baostock_day_doc, write_day_kline_docs
from backtest.utils import is_bj_code, to_trade_datetime, to_xt_code
from backtest.utils.log import log_event

from .baostock_source import fetch_baostock_range
from .constants import DAY_COLLECTION, MAX_DAILY_SYNC_BATCH_SIZE
from .missing import build_missing_by_code, filter_xt_results_to_update_window, merge_dates_to_ranges
from .models import StockMeta
from .st_status import build_st_status_by_code
from .universe import (
    build_expected_dates_by_code,
    build_manual_code_start_map,
    latest_available_day_kline_date,
    load_previous_state_before_start_map,
    load_manual_universe,
    resolve_requested_codes,
)
from .xt_details import enrich_stock_metas_with_xt_details, fetch_xt_detail_map
from .xtquant_source import fetch_xt_market_data, iter_xt_sync_batches, xt_market_data_to_docs

ManualSource = Literal["xtquant", "baostock"]


def _metas_for_codes(codes: list[str], universe) -> list[StockMeta]:
    metas: list[StockMeta] = []
    for code in codes:
        meta = universe[code]
        metas.append(
            StockMeta(
                code=meta.code,
                xt_code=to_xt_code(meta.code),
                code_name=meta.code_name,
                ipo_date=meta.ipo_date,
                out_date=meta.out_date,
            )
        )
    return metas


def _flush_manual_docs(cfg: DuckDBConfig, docs: list[dict[str, Any]], *, dry_run: bool) -> int:
    planned = len(docs)
    if not docs:
        return 0
    if not dry_run:
        write_day_kline_docs(cfg, DAY_COLLECTION, docs)
    docs.clear()
    return planned


def _run_baostock_source(
    *,
    cfg: DuckDBConfig,
    expected_dates_by_code: dict[str, set[datetime]],
    trade_day_positions: dict[datetime, int],
    batch_size: int,
    dry_run: bool,
) -> tuple[dict[str, set[datetime]], dict[str, Any]]:
    docs_to_write: list[dict[str, Any]] = []
    written_dates_by_code: dict[str, set[datetime]] = {}
    skipped_bj_rows = 0
    planned_total = 0
    write_batches = 0
    for code, missing_dates in expected_dates_by_code.items():
        if is_bj_code(code):
            skipped_bj_rows += len(missing_dates)
            continue
        for date_range in merge_dates_to_ranges(sorted(missing_dates), trade_day_positions):
            frame = fetch_baostock_range(code, date_range)
            if frame.empty:
                continue
            for row in frame.to_dict("records"):
                trade_date = to_trade_datetime(row["date"])
                if trade_date not in expected_dates_by_code.get(code, set()):
                    continue
                doc = build_baostock_day_doc(pd.Series(row), trade_date)
                written_dates_by_code.setdefault(code, set()).add(trade_date)
                docs_to_write.append(doc)
                if len(docs_to_write) >= batch_size:
                    planned_total += _flush_manual_docs(cfg, docs_to_write, dry_run=dry_run)
                    if not dry_run:
                        write_batches += 1

    remaining = _flush_manual_docs(cfg, docs_to_write, dry_run=dry_run)
    planned_total += remaining
    if remaining and not dry_run:
        write_batches += 1
    return written_dates_by_code, {
        "planned_total": planned_total,
        "skipped_bj_rows": skipped_bj_rows,
        "write_batches": write_batches,
    }


def _run_xtquant_source(
    *,
    cfg: DuckDBConfig,
    xtdata_client,
    metas: list[StockMeta],
    code_start_map: dict[str, datetime],
    expected_dates_by_code: dict[str, set[datetime]],
    previous_state_before_start_map: dict[str, dict[str, Any]],
    st_status_by_code: dict[str, dict[datetime, bool]],
    end_date: datetime,
    batch_size: int,
    xt_batch_size: int,
    dry_run: bool,
) -> tuple[dict[str, set[datetime]], dict[str, list[datetime]], dict[str, Any]]:
    written_dates_by_code: dict[str, set[datetime]] = {}
    all_invalid_dates: dict[str, list[datetime]] = {}
    planned_total = 0
    write_batches = 0
    for batch_start, batch_metas in iter_xt_sync_batches(metas, code_start_map, xt_batch_size=xt_batch_size):
        market_data = fetch_xt_market_data(xtdata_client, [meta.xt_code for meta in batch_metas], batch_start, end_date)
        docs, invalid_dates = xt_market_data_to_docs(
            market_data,
            batch_metas,
            previous_state_before_start_map,
            st_status_by_code=st_status_by_code,
        )
        docs, invalid_dates = filter_xt_results_to_update_window(expected_dates_by_code, docs, invalid_dates)
        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            if not batch:
                continue
            if not dry_run:
                write_day_kline_docs(cfg, DAY_COLLECTION, batch)
                write_batches += 1
            planned_total += len(batch)
            for doc in batch:
                written_dates_by_code.setdefault(str(doc["code"]), set()).add(to_trade_datetime(doc["date"]))
        for code, dates in invalid_dates.items():
            all_invalid_dates.setdefault(code, []).extend(dates)
    return written_dates_by_code, all_invalid_dates, {
        "planned_total": planned_total,
        "write_batches": write_batches,
    }


def run_manual_day_kline_sync(
    *,
    source: ManualSource,
    stocks: str,
    start_date: datetime,
    end_date: datetime,
    xtdata_client=None,
    cfg: DuckDBConfig | None = None,
    batch_size: int = 2000,
    xt_batch_size: int = 300,
    max_attempts: int = 3,
    dry_run: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    if source not in {"xtquant", "baostock"}:
        raise ValueError("source must be one of: xtquant, baostock")
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    cfg = cfg or DuckDBConfig()
    batch_size = max(1, min(int(batch_size), MAX_DAILY_SYNC_BATCH_SIZE))
    xt_batch_size = max(1, int(xt_batch_size))
    now = now or datetime.now()
    candidate_end_date = latest_available_day_kline_date(now)
    requested_end_date = end_date
    end_date = min(end_date, candidate_end_date)
    universe = load_manual_universe(cfg, requested_end_date)
    codes = resolve_requested_codes(stocks, universe)
    base_summary: dict[str, Any] = {
        "source": source,
        "requested_codes": len(codes),
        "active_codes": 0,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "requested_end_date": requested_end_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "candidate_end_date": candidate_end_date.strftime("%Y-%m-%d"),
        "market_close_time": "15:45:00",
        "end_date_truncated": end_date < requested_end_date,
        "trade_days": 0,
        "planned_total": 0,
        "written_total": 0,
        "skipped_bj_rows": 0,
        "missing_days": 0,
        "missing_by_code": {},
        "write_batches": 0,
        "dry_run": dry_run,
    }
    log_event(
        "info",
        "manual daily kline cutoff resolved",
        source=source,
        beijing_now=now,
        market_close_time="15:45:00",
        requested_end_date=requested_end_date.strftime("%Y-%m-%d"),
        effective_end_date=end_date.strftime("%Y-%m-%d"),
        end_date_truncated=end_date < requested_end_date,
    )
    if start_date > end_date:
        log_event(
            "info",
            "manual daily kline skipped before market close",
            source=source,
            start_date=start_date.strftime("%Y-%m-%d"),
            effective_end_date=end_date.strftime("%Y-%m-%d"),
        )
        base_summary["skip_reason"] = "no_closed_day_in_requested_range"
        return base_summary

    if source == "xtquant" and xtdata_client is None:
        from .daily_sync import default_xtdata

        xtdata_client = default_xtdata
        if xtdata_client is None:
            raise RuntimeError("xtquant is not available in current Python environment")

    login_with_retry()
    try:
        code_start_map = build_manual_code_start_map(codes, universe, start_date, end_date)
        calendar_start_date = min(code_start_map.values()) if code_start_map else start_date
        log_event(
            "info",
            "manual daily kline trade calendar query",
            start_date=calendar_start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        trade_dates = fetch_trade_calendar(calendar_start_date, end_date)
        trade_day_positions = {trade_date: index for index, trade_date in enumerate(trade_dates)}
        metas = _metas_for_codes(codes, universe)
        expected_dates_by_code = build_expected_dates_by_code(
            metas,
            code_start_map,
            trade_dates,
            end_date,
        )

        summary = dict(base_summary)
        summary["active_codes"] = len(code_start_map)
        summary["trade_days"] = len(trade_dates)
        if not code_start_map:
            log_event("info", "manual daily kline no codes need update", requested_codes=len(codes), source=source)
            return summary

        log_event(
            "info",
            "manual daily kline sync start",
            source=source,
            active_codes=len(code_start_map),
            dry_run=dry_run,
        )

        if source == "baostock":
            written_dates_by_code, source_summary = _run_baostock_source(
                cfg=cfg,
                expected_dates_by_code=expected_dates_by_code,
                trade_day_positions=trade_day_positions,
                batch_size=batch_size,
                dry_run=dry_run,
            )
            missing_by_code = build_missing_by_code(expected_dates_by_code, written_dates_by_code, {})

        else:
            previous_state_before_start_map = load_previous_state_before_start_map(
                cfg,
                code_start_map,
            )
            active_metas = [meta for meta in metas if meta.code in code_start_map]
            active_stock_codes = {
                meta.code
                for meta in active_metas
                if not universe[meta.code].is_index
            }
            active_xt_codes = [
                meta.xt_code
                for meta in active_metas
                if not universe[meta.code].is_index
            ]
            current_details_by_xt_code = fetch_xt_detail_map(
                xtdata_client,
                active_xt_codes,
            )
            active_metas, detail_stats = enrich_stock_metas_with_xt_details(
                active_metas,
                current_details_by_xt_code,
                active_codes=active_stock_codes,
            )
            latest_trade_date = trade_dates[-1] if trade_dates else end_date
            stock_expected_dates = {
                code: dates
                for code, dates in expected_dates_by_code.items()
                if code in active_stock_codes
            }
            st_status_by_code = build_st_status_by_code(
                xtdata_client,
                stock_expected_dates,
                latest_trade_date + timedelta(days=1),
                current_details_by_xt_code=current_details_by_xt_code,
                max_attempts=max_attempts,
            )
            for meta in active_metas:
                if universe[meta.code].is_index:
                    st_status_by_code[meta.code] = {
                        trade_date: False
                        for trade_date in expected_dates_by_code.get(meta.code, set())
                    }
            written_dates_by_code, all_invalid_dates, source_summary = _run_xtquant_source(
                cfg=cfg,
                xtdata_client=xtdata_client,
                metas=active_metas,
                code_start_map=code_start_map,
                expected_dates_by_code=expected_dates_by_code,
                previous_state_before_start_map=previous_state_before_start_map,
                st_status_by_code=st_status_by_code,
                end_date=end_date,
                batch_size=batch_size,
                xt_batch_size=xt_batch_size,
                dry_run=dry_run,
            )
            source_summary.update(detail_stats)
            missing_by_code = build_missing_by_code(expected_dates_by_code, written_dates_by_code, all_invalid_dates)

        summary.update(source_summary)
        summary["written_total"] = int(source_summary.get("planned_total", 0))
        summary["missing_days"] = sum(len(set(values)) for values in missing_by_code.values())
        summary["missing_by_code"] = {
            code: [date.strftime("%Y-%m-%d") for date in sorted(set(dates))]
            for code, dates in sorted(missing_by_code.items())
            if dates
        }
        log_event(
            "info",
            "manual daily kline sync finished",
            **{k: v for k, v in summary.items() if k != "missing_by_code"},
        )
        return summary
    finally:
        safe_logout()
