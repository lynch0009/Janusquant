from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

try:  # pragma: no cover - exercised only in a real QMT runtime.
    from xtquant import xtdata as default_xtdata
except ImportError:  # pragma: no cover - unit tests run without xtquant in some envs.
    default_xtdata = None

from backtest.db import DuckDBConfig
from backtest.fetch_data.baostock_utils import fetch_trade_calendar, login_with_retry, safe_logout
from backtest.fetch_data.day_kline_common import write_day_kline_docs
from backtest.utils import is_bj_code
from backtest.utils.log import log_event

from .baostock_source import fallback_missing_dates_with_baostock
from .basic_info import load_basic_docs, sync_incremental_basic_info
from .constants import (
    DAY_COLLECTION,
    DEFAULT_MAX_FALLBACK_MISSING_DAYS,
    DEFAULT_MAX_FALLBACK_MISSING_STOCKS,
    FALLBACK_WINDOW_TRADE_DAYS,
    FIXED_DAY_KLINE_INDEX_CODES,
    MAX_MONGO_BATCH_SIZE,
)
from .missing import (
    add_fallback_counts_by_date,
    add_written_doc_stats,
    build_historical_missing_rows,
    build_missing_by_code,
    filter_xt_results_to_update_window,
    missing_codes_on_latest_trade_date,
    split_missing_by_fallback_window,
)
from .models import DailySyncSummaryBuilder
from .report import should_write_report, write_daily_sync_report
from .st_status import build_st_status_by_code
from .universe import (
    build_code_start_map,
    build_day_kline_update_universe,
    build_expected_dates_by_code,
    load_latest_state_map,
    merge_fixed_day_kline_indexes,
)
from .xt_details import enrich_stock_metas_with_xt_details, fetch_xt_detail_map
from .xtquant_source import fetch_xt_market_data, iter_xt_sync_batches, xt_market_data_to_docs


def resolve_latest_trade_date() -> datetime:
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = today - timedelta(days=40)
    trading_days = fetch_trade_calendar(start_date, today)
    if not trading_days:
        raise ValueError("failed to resolve latest trade date")
    return trading_days[-1]


def run_daily_sync(
    *,
    xtdata_client=None,
    cfg: DuckDBConfig | None = None,
    batch_size: int = 2000,
    xt_batch_size: int = 300,
    max_fallback_missing_stocks: int = DEFAULT_MAX_FALLBACK_MISSING_STOCKS,
    max_fallback_missing_days: int = DEFAULT_MAX_FALLBACK_MISSING_DAYS,
) -> dict[str, Any]:
    xtdata_client = xtdata_client or default_xtdata
    if xtdata_client is None:
        raise RuntimeError("xtquant is not available in current Python environment")

    cfg = cfg or DuckDBConfig()
    batch_size = max(1, min(int(batch_size), MAX_MONGO_BATCH_SIZE))
    xt_batch_size = max(1, int(xt_batch_size))
    now = datetime.now()

    log_event("info", "xtquant daily kline sync start")
    # 1. 股票池只做增量比较：已有 DuckDB basic_info 是主数据源，新股才调用 detail。
    basic_result = sync_incremental_basic_info(
        cfg,
        xtdata_client,
        now=now,
        batch_size=batch_size,
    )
    log_event(
        "info",
        "basic info incremental sync finished",
        inserted=len(basic_result.inserted_docs),
        detail_failed=len(basic_result.detail_failed_codes),
        stock_pool_diff=len(basic_result.stock_pool_diff_rows),
        **basic_result.db_operation_summary,
    )

    all_basic_docs = load_basic_docs(cfg, hs_only=False)
    update_metas, universe_stats = build_day_kline_update_universe(
        all_basic_docs,
        basic_result.stock_pool_diff_rows,
        now,
    )
    update_metas, fixed_index_count = merge_fixed_day_kline_indexes(update_metas)
    bj_skipped_stock_count = sum(
        1
        for doc in all_basic_docs
        if str(doc.get("code", "")).strip() and is_bj_code(str(doc.get("code", "")))
    )
    codes = [meta.code for meta in update_metas]
    if not update_metas:
        raise RuntimeError("DuckDB basic_info has no active/listed 沪深 A-share stocks for daily kline sync")
    summary_builder = DailySyncSummaryBuilder(
        basic_result=basic_result,
        bj_skipped_stock_count=bj_skipped_stock_count,
        universe_stats=universe_stats,
        update_universe_count=len(update_metas),
        stock_count=len(codes),
        fixed_index_count=fixed_index_count,
    )

    login_with_retry()
    try:
        # 2. 交易日历仍由 Baostock 提供，用来判断 xtquant 结果是否真的缺日期。
        end_date = resolve_latest_trade_date()
        latest_state_map = load_latest_state_map(cfg, codes, end_date)
        code_start_map = build_code_start_map(update_metas, latest_state_map, end_date)
        summary_builder.active_stock_count = len(code_start_map)
        missing_by_code: dict[str, list[datetime]] = {}
        if not code_start_map:
            log_event("info", "no stocks need daily kline update", end_date=end_date.strftime("%Y-%m-%d"))
            summary = summary_builder.to_dict()
            if should_write_report(basic_result, missing_by_code, [], []):
                write_daily_sync_report(basic_result, summary, now=now, historical_missing_rows=[])
            return summary

        trade_dates = fetch_trade_calendar(min(code_start_map.values()), end_date)
        if not trade_dates:
            raise RuntimeError("no trade dates resolved for xtquant daily kline update")
        expected_dates_by_code = build_expected_dates_by_code(update_metas, code_start_map, trade_dates, end_date)
        trade_day_positions = {trade_date: index for index, trade_date in enumerate(trade_dates)}
        active_codes = set(code_start_map)
        active_xt_codes = [
            meta.xt_code
            for meta in update_metas
            if meta.code in active_codes and meta.code not in FIXED_DAY_KLINE_INDEX_CODES
        ]
        current_details_by_xt_code = fetch_xt_detail_map(xtdata_client, active_xt_codes)
        update_metas, float_volume_stats = enrich_stock_metas_with_xt_details(
            update_metas,
            current_details_by_xt_code,
            active_codes=active_codes,
        )
        universe_stats.update(float_volume_stats)
        st_status_by_code = build_st_status_by_code(
            xtdata_client,
            expected_dates_by_code,
            end_date,
            current_details_by_xt_code=current_details_by_xt_code,
        )
        log_event(
            "info",
            "daily kline update universe ready",
            hs_stock_count=len(update_metas),
            active_stock_count=len(code_start_map),
            bj_skipped_stock_count=bj_skipped_stock_count,
            fixed_index_count=fixed_index_count,
            **universe_stats,
        )

        written_dates_by_code: dict[str, set[datetime]] = {}
        updated_by_date: dict[str, int] = {}
        xtquant_docs_count = 0
        all_invalid_dates: dict[str, list[datetime]] = {}
        pending_start: datetime | None = None
        pending_docs: list[dict[str, Any]] = []

        def flush_pending_docs() -> None:
            nonlocal xtquant_docs_count, pending_docs
            if not pending_docs:
                return
            write_day_kline_docs(cfg, DAY_COLLECTION, pending_docs)
            xtquant_docs_count += add_written_doc_stats(pending_docs, written_dates_by_code, updated_by_date)
            pending_docs = []

        for batch_start, batch_metas in iter_xt_sync_batches(update_metas, code_start_map, xt_batch_size=xt_batch_size):
            # 3. 先按增量起点分组，再分批拉 xtquant，避免一只长缺口股票带着整批回拉历史。
            if not batch_metas:
                continue
            if pending_start is None:
                pending_start = batch_start
            elif batch_start != pending_start:
                flush_pending_docs()
                pending_start = batch_start
            market_data = fetch_xt_market_data(
                xtdata_client,
                [meta.xt_code for meta in batch_metas],
                batch_start,
                end_date,
            )
            docs, invalid_dates = xt_market_data_to_docs(
                market_data,
                batch_metas,
                latest_state_map,
                st_status_by_code=st_status_by_code,
            )
            docs, invalid_dates = filter_xt_results_to_update_window(
                expected_dates_by_code,
                docs,
                invalid_dates,
            )
            if docs:
                pending_docs.extend(docs)
            for code, dates in invalid_dates.items():
                all_invalid_dates.setdefault(code, []).extend(dates)
            log_event(
                "info",
                "xtquant daily kline batch processed",
                start_date=batch_start.strftime("%Y-%m-%d"),
                batch_size=len(batch_metas),
                docs=len(docs),
                invalid_days=sum(len(values) for values in invalid_dates.values()),
            )
        flush_pending_docs()

        missing_by_code = build_missing_by_code(
            expected_dates_by_code,
            written_dates_by_code,
            all_invalid_dates,
        )
        missing_days = sum(len(set(values)) for values in missing_by_code.values())
        if missing_days:
            log_event(
                "warning",
                "xtquant_missing_detected",
                stock_count=len(missing_by_code),
                missing_days=missing_days,
            )
        recent_missing_by_code, historical_missing_by_code = split_missing_by_fallback_window(
            missing_by_code,
            trade_dates,
            fallback_window_trade_days=FALLBACK_WINDOW_TRADE_DAYS,
        )
        historical_missing_rows = build_historical_missing_rows(historical_missing_by_code)
        if historical_missing_rows:
            log_event(
                "warning",
                "historical_missing_outside_fallback_window",
                stock_count=len(historical_missing_by_code),
                missing_days=len(historical_missing_rows),
                fallback_window_trade_days=FALLBACK_WINDOW_TRADE_DAYS,
            )
        # 4. 只对疑似缺口做 Baostock fallback，停牌由 Baostock tradestatus 消解。
        fallback_summary = fallback_missing_dates_with_baostock(
            cfg,
            recent_missing_by_code,
            trade_day_positions,
            batch_size=batch_size,
            max_missing_stocks=max_fallback_missing_stocks,
            max_missing_days=max_fallback_missing_days,
        )
        updated_by_date = add_fallback_counts_by_date(updated_by_date, fallback_summary)
        missing_today_codes = missing_codes_on_latest_trade_date(missing_by_code, fallback_summary, end_date)
        day_kline_updated_count = xtquant_docs_count + int(fallback_summary.get("fallback_docs", 0)) + int(
            fallback_summary.get("normal_suspend_days", 0)
        )

        summary_builder.day_kline_updated_count = day_kline_updated_count
        summary_builder.xtquant_docs_count = xtquant_docs_count
        summary_builder.day_kline_updated_by_date = updated_by_date
        summary_builder.day_kline_missing_codes_today = missing_today_codes
        summary_builder.missing_by_code = missing_by_code
        summary_builder.fallback_summary = fallback_summary
        summary_builder.historical_missing_rows = historical_missing_rows
        result = summary_builder.to_dict()
        if should_write_report(basic_result, missing_by_code, missing_today_codes, historical_missing_rows):
            write_daily_sync_report(
                basic_result,
                result,
                now=now,
                historical_missing_rows=historical_missing_rows,
            )
        log_event(
            "info",
            "xtquant daily kline sync finished",
            basic_inserted_count=result["basic_inserted_count"],
            stock_pool_xt_only_count=result["stock_pool_xt_only_count"],
            stock_pool_db_only_count=result["stock_pool_db_only_count"],
            day_kline_update_universe_count=result["day_kline_update_universe_count"],
            fixed_index_update_universe_count=result["fixed_index_update_universe_count"],
            day_kline_updated_count=result["day_kline_updated_count"],
            day_kline_missing_today_count=result["day_kline_missing_today_count"],
            report_dir=result["report_dir"],
        )
        return result
    finally:
        safe_logout()


def run_xtquant_daily_kline_sync(**kwargs) -> dict[str, Any]:
    return run_daily_sync(**kwargs)
