from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from backtest.db import DuckDBConfig
from backtest.fetch_data.baostock_utils import fetch_trade_calendar, login_with_retry, safe_logout
from backtest.fetch_data.day_kline_common import build_baostock_day_doc, write_day_kline_docs
from backtest.utils import is_bj_code, is_st_name, to_trade_datetime, to_xt_code
from backtest.utils.log import log_event

from .baostock_source import fetch_baostock_range
from .constants import DAY_COLLECTION, DEFAULT_LOCAL_ROOT, MAX_MONGO_BATCH_SIZE
from .local_source import build_local_doc, load_finance_index, read_local_csv
from .missing import build_missing_by_code, filter_xt_results_to_update_window, merge_dates_to_ranges
from .models import StockMeta
from .universe import (
    build_manual_code_start_map,
    build_symbol_code_map,
    load_latest_state_map,
    load_manual_universe,
    resolve_requested_codes,
)
from .xtquant_source import fetch_xt_market_data, iter_xt_sync_batches, xt_market_data_to_docs

ManualSource = Literal["xtquant", "baostock", "local"]


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


def _trade_codes_for_date(code_start_map, universe, trade_date: datetime) -> list[str]:
    return [
        code
        for code, start in code_start_map.items()
        if trade_date >= start and (universe[code].out_date is None or trade_date <= universe[code].out_date)
    ]


def _run_local_source(
    *,
    cfg: DuckDBConfig,
    universe,
    codes: list[str],
    code_start_map: dict[str, datetime],
    latest_state_map: dict[str, dict[str, Any]],
    trade_dates: list[datetime],
    end_date: datetime,
    local_root: str | Path,
    batch_size: int,
    dry_run: bool,
) -> tuple[dict[str, set[datetime]], dict[str, Any]]:
    finance_index = load_finance_index(cfg, codes, end_date)
    symbol_map = build_symbol_code_map(universe)
    isst_state = {
        code: bool(latest_state_map.get(code, {}).get("isST", is_st_name(universe[code].code_name)))
        for code in codes
    }
    docs_to_write: list[dict[str, Any]] = []
    written_dates_by_code: dict[str, set[datetime]] = {}
    local_unmatched_total = 0
    planned_total = 0
    local_root_path = Path(local_root)

    for trade_date in trade_dates:
        trade_codes = _trade_codes_for_date(code_start_map, universe, trade_date)
        if not trade_codes:
            continue
        file_path = local_root_path / f"{trade_date:%Y%m%d}.csv"
        local_frame, unmatched_count = read_local_csv(file_path, symbol_map) if file_path.exists() else (pd.DataFrame(), 0)
        local_unmatched_total += unmatched_count
        local_rows_by_code = local_frame.set_index("code", drop=False).to_dict("index") if not local_frame.empty else {}
        for code in trade_codes:
            row = local_rows_by_code.get(code)
            if row is None:
                continue
            doc = build_local_doc(pd.Series(row), code, trade_date, isst_state.get(code, False), finance_index)
            written_dates_by_code.setdefault(code, set()).add(trade_date)
            docs_to_write.append(doc)
            if len(docs_to_write) >= batch_size:
                planned_total += _flush_manual_docs(cfg, docs_to_write, dry_run=dry_run)

    planned_total += _flush_manual_docs(cfg, docs_to_write, dry_run=dry_run)
    return written_dates_by_code, {
        "local_unmatched_codes": local_unmatched_total,
        "planned_total": planned_total,
    }


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

    planned_total += _flush_manual_docs(cfg, docs_to_write, dry_run=dry_run)
    return written_dates_by_code, {
        "planned_total": planned_total,
        "skipped_bj_rows": skipped_bj_rows,
    }


def _run_xtquant_source(
    *,
    cfg: DuckDBConfig,
    xtdata_client,
    universe,
    codes: list[str],
    code_start_map: dict[str, datetime],
    expected_dates_by_code: dict[str, set[datetime]],
    latest_state_map: dict[str, dict[str, Any]],
    end_date: datetime,
    batch_size: int,
    xt_batch_size: int,
    dry_run: bool,
) -> tuple[dict[str, set[datetime]], dict[str, list[datetime]], dict[str, Any]]:
    metas = _metas_for_codes(codes, universe)
    written_dates_by_code: dict[str, set[datetime]] = {}
    all_invalid_dates: dict[str, list[datetime]] = {}
    planned_total = 0
    for batch_start, batch_metas in iter_xt_sync_batches(metas, code_start_map, xt_batch_size=xt_batch_size):
        market_data = fetch_xt_market_data(xtdata_client, [meta.xt_code for meta in batch_metas], batch_start, end_date)
        docs, invalid_dates = xt_market_data_to_docs(market_data, batch_metas, latest_state_map)
        docs, invalid_dates = filter_xt_results_to_update_window(expected_dates_by_code, docs, invalid_dates)
        if docs:
            if not dry_run:
                write_day_kline_docs(cfg, DAY_COLLECTION, docs)
            planned_total += len(docs)
            for doc in docs:
                written_dates_by_code.setdefault(str(doc["code"]), set()).add(to_trade_datetime(doc["date"]))
        for code, dates in invalid_dates.items():
            all_invalid_dates.setdefault(code, []).extend(dates)
    return written_dates_by_code, all_invalid_dates, {"planned_total": planned_total}


def run_manual_day_kline_sync(
    *,
    source: ManualSource,
    stocks: str,
    start_date: datetime,
    end_date: datetime,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    xtdata_client=None,
    cfg: DuckDBConfig | None = None,
    batch_size: int = 2000,
    xt_batch_size: int = 300,
    dry_run: bool = False,
) -> dict[str, Any]:
    if source not in {"xtquant", "baostock", "local"}:
        raise ValueError("source must be one of: xtquant, baostock, local")
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    if source == "xtquant" and xtdata_client is None:
        from .daily_sync import default_xtdata

        xtdata_client = default_xtdata
        if xtdata_client is None:
            raise RuntimeError("xtquant is not available in current Python environment")

    cfg = cfg or DuckDBConfig()
    batch_size = max(1, min(int(batch_size), MAX_MONGO_BATCH_SIZE))
    xt_batch_size = max(1, int(xt_batch_size))
    needs_baostock_login = True
    if needs_baostock_login:
        login_with_retry()
    try:
        universe = load_manual_universe(cfg, end_date)
        codes = resolve_requested_codes(stocks, universe)
        latest_state_map = load_latest_state_map(cfg, codes, end_date)
        code_start_map = build_manual_code_start_map(codes, universe, latest_state_map, start_date, end_date)
        trade_dates = fetch_trade_calendar(min(code_start_map.values()) if code_start_map else start_date, end_date)
        trade_day_positions = {trade_date: index for index, trade_date in enumerate(trade_dates)}
        expected_dates_by_code = {
            code: {
                trade_date
                for trade_date in trade_dates
                if trade_date >= code_start_map.get(code, end_date) and trade_date <= end_date
            }
            for code in code_start_map
        }

        summary: dict[str, Any] = {
            "source": source,
            "requested_codes": len(codes),
            "active_codes": len(code_start_map),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "trade_days": len(trade_dates),
            "planned_total": 0,
            "written_total": 0,
            "skipped_bj_rows": 0,
            "missing_days": 0,
            "missing_by_code": {},
            "dry_run": dry_run,
        }
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

        if source == "local":
            written_dates_by_code, source_summary = _run_local_source(
                cfg=cfg,
                universe=universe,
                codes=codes,
                code_start_map=code_start_map,
                latest_state_map=latest_state_map,
                trade_dates=trade_dates,
                end_date=end_date,
                local_root=local_root,
                batch_size=batch_size,
                dry_run=dry_run,
            )
            missing_by_code = build_missing_by_code(expected_dates_by_code, written_dates_by_code, {})

        elif source == "baostock":
            written_dates_by_code, source_summary = _run_baostock_source(
                cfg=cfg,
                expected_dates_by_code=expected_dates_by_code,
                trade_day_positions=trade_day_positions,
                batch_size=batch_size,
                dry_run=dry_run,
            )
            missing_by_code = build_missing_by_code(expected_dates_by_code, written_dates_by_code, {})

        else:
            written_dates_by_code, all_invalid_dates, source_summary = _run_xtquant_source(
                cfg=cfg,
                xtdata_client=xtdata_client,
                universe=universe,
                codes=codes,
                code_start_map=code_start_map,
                expected_dates_by_code=expected_dates_by_code,
                latest_state_map=latest_state_map,
                end_date=end_date,
                batch_size=batch_size,
                xt_batch_size=xt_batch_size,
                dry_run=dry_run,
            )
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
        if needs_baostock_login:
            safe_logout()
