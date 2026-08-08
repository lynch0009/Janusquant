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
from backtest.utils import is_bj_code, is_hs_a_share_code
from backtest.utils.log import log_event

from .baostock_source import fallback_missing_dates_with_baostock
from .basic_info import load_basic_docs, project_basic_docs, sync_incremental_basic_info
from .constants import (
    DAY_COLLECTION,
    DEFAULT_MAX_FALLBACK_MISSING_DAYS,
    DEFAULT_MAX_FALLBACK_MISSING_STOCKS,
    FALLBACK_WINDOW_TRADE_DAYS,
    FIXED_DAY_KLINE_INDEX_CODES,
    MAX_DAILY_SYNC_BATCH_SIZE,
)
from .missing import (
    add_fallback_counts_by_date,
    add_written_doc_stats,
    build_historical_missing_rows,
    build_missing_by_code,
    filter_xt_results_to_update_window,
    split_missing_by_fallback_window,
    unresolved_after_fallback,
)
from .models import DailySyncSummaryBuilder
from .report import should_write_report, write_daily_sync_report
from .st_status import build_st_status_by_code
from .universe import (
    build_code_start_map,
    build_day_kline_update_universe,
    build_expected_dates_by_code,
    latest_available_day_kline_date,
    load_latest_state_map,
    load_previous_trade_date_stock_count,
    merge_fixed_day_kline_indexes,
)
from .xt_details import enrich_stock_metas_with_xt_details
from .xtquant_source import fetch_xt_market_data, iter_xt_sync_batches, xt_market_data_to_docs


def set_daily_unit_counts(
    summary_builder: DailySyncSummaryBuilder,
    *,
    expected_codes: set[str],
    unresolved_codes: set[str],
) -> None:
    basic_result = summary_builder.basic_result
    planned_codes = set(expected_codes) | set(basic_result.detail_requested_codes)
    failed_codes = (
        set(basic_result.detail_failed_codes)
        | set(unresolved_codes)
    )
    skipped_codes = (
        set(basic_result.db_only_detail_missing_codes)
        | set(basic_result.missing_ipo_codes)
    ) - failed_codes
    succeeded_codes = planned_codes - failed_codes - skipped_codes
    summary_builder.unit_planned_count = len(planned_codes)
    summary_builder.unit_attempted_count = len(planned_codes)
    summary_builder.unit_succeeded_count = len(succeeded_codes)
    summary_builder.unit_skipped_count = len(skipped_codes)
    summary_builder.unit_failed_count = len(failed_codes & planned_codes)


def run_daily_sync(
    *,
    xtdata_client=None,
    cfg: DuckDBConfig | None = None,
    batch_size: int = 2000,
    xt_batch_size: int = 300,
    max_fallback_missing_stocks: int = DEFAULT_MAX_FALLBACK_MISSING_STOCKS,
    max_fallback_missing_days: int = DEFAULT_MAX_FALLBACK_MISSING_DAYS,
    max_attempts: int = 3,
    dry_run: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    xtdata_client = xtdata_client or default_xtdata
    if xtdata_client is None:
        raise RuntimeError("xtquant is not available in current Python environment")

    cfg = cfg or DuckDBConfig()
    batch_size = max(1, min(int(batch_size), MAX_DAILY_SYNC_BATCH_SIZE))
    xt_batch_size = max(1, int(xt_batch_size))
    now = now or datetime.now()
    candidate_end_date = latest_available_day_kline_date(now)

    log_event(
        "info",
        "xtquant daily kline sync start",
        beijing_now=now,
        market_close_time="15:45:00",
        before_market_close=candidate_end_date.date() < now.date(),
        candidate_end_date=candidate_end_date.strftime("%Y-%m-%d"),
    )
    # 1. 全部沪深股票与 db_only 股票合并为一次 detail 批量请求，后续阶段复用结果。
    basic_result = sync_incremental_basic_info(
        cfg,
        xtdata_client,
        now=now,
        dry_run=dry_run,
        initialize_adjust_factor_baselines=True,
    )
    log_event(
        "info",
        "basic info incremental sync finished",
        inserted=len(basic_result.inserted_docs),
        detail_failed=len(basic_result.detail_failed_codes),
        missing_ipo=len(basic_result.missing_ipo_codes),
        stock_pool_diff=len(basic_result.stock_pool_diff_rows),
        adjust_factor_baseline_planned=basic_result.adjust_factor_baseline_planned_count,
        adjust_factor_baseline_existing=basic_result.adjust_factor_baseline_existing_count,
        adjust_factor_baseline_written=basic_result.adjust_factor_baseline_written_count,
        db_only_confirmed_delisted=len(basic_result.db_only_confirmed_delisted_codes),
        db_only_active_detail=len(basic_result.db_only_active_detail_codes),
        db_only_detail_missing=len(basic_result.db_only_detail_missing_codes),
        instrument_detail_requested=basic_result.instrument_detail_requested_count,
        instrument_detail_returned=len(basic_result.details_by_xt_code),
        **basic_result.db_operation_summary,
    )
    if basic_result.db_only_active_detail_codes or basic_result.db_only_detail_missing_codes:
        log_event(
            "warning",
            "xtquant_stock_pool_difference_preserved",
            active_detail_codes=list(basic_result.db_only_active_detail_codes),
            detail_missing_codes=list(basic_result.db_only_detail_missing_codes),
        )

    all_basic_docs = project_basic_docs(
        load_basic_docs(cfg, hs_only=False),
        basic_result,
    )
    update_metas, universe_stats = build_day_kline_update_universe(
        all_basic_docs,
        basic_result.stock_pool_diff_rows,
        now,
        excluded_db_only_codes=basic_result.db_only_detail_missing_codes,
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

    latest_state_map = load_latest_state_map(cfg, codes, candidate_end_date)
    provisional_start_map = build_code_start_map(update_metas, latest_state_map, candidate_end_date)
    calendar_start_date = candidate_end_date - timedelta(days=40)
    if provisional_start_map:
        calendar_start_date = min(calendar_start_date, min(provisional_start_map.values()))

    login_with_retry()
    try:
        # 2. 单次获取完整任务区间的交易日历，同时确定最终截止交易日和缺口日期。
        log_event(
            "info",
            "daily kline trade calendar query",
            start_date=calendar_start_date.strftime("%Y-%m-%d"),
            end_date=candidate_end_date.strftime("%Y-%m-%d"),
        )
        trade_dates = fetch_trade_calendar(calendar_start_date, candidate_end_date)
        if not trade_dates:
            raise ValueError("failed to resolve latest trade date")
        end_date = trade_dates[-1]
        log_event(
            "info",
            "daily kline trade calendar resolved",
            trade_days=len(trade_dates),
            final_trade_date=end_date.strftime("%Y-%m-%d"),
        )
        (
            previous_trade_date,
            previous_trade_date_stock_count,
        ) = load_previous_trade_date_stock_count(
            cfg,
            end_date,
        )
        summary_builder.previous_trade_date = (
            previous_trade_date.strftime("%Y-%m-%d")
            if previous_trade_date is not None
            else None
        )
        summary_builder.previous_trade_date_stock_count = previous_trade_date_stock_count
        summary_builder.updated_stock_count_diff_vs_previous_trade_date = (
            -previous_trade_date_stock_count
        )
        code_start_map = build_code_start_map(update_metas, latest_state_map, end_date)
        summary_builder.active_stock_count = len(code_start_map)
        missing_by_code: dict[str, list[datetime]] = {}
        if not code_start_map:
            log_event("info", "no stocks need daily kline update", end_date=end_date.strftime("%Y-%m-%d"))
            set_daily_unit_counts(
                summary_builder,
                expected_codes=set(),
                unresolved_codes=set(),
            )
            summary = summary_builder.to_dict()
            summary["write_batches"] = 0
            if should_write_report(basic_result, missing_by_code, [], []):
                write_daily_sync_report(basic_result, summary, now=now, historical_missing_rows=[])
            return summary

        expected_dates_by_code = build_expected_dates_by_code(update_metas, code_start_map, trade_dates, end_date)
        trade_day_positions = {trade_date: index for index, trade_date in enumerate(trade_dates)}
        active_codes = set(code_start_map)
        active_xt_codes = [
            meta.xt_code
            for meta in update_metas
            if meta.code in active_codes and meta.code not in FIXED_DAY_KLINE_INDEX_CODES
        ]
        current_details_by_xt_code = {
            code: detail
            for code, detail in basic_result.details_by_xt_code.items()
            if code in active_xt_codes
        }
        missing_detail_codes = sorted(set(active_xt_codes) - set(current_details_by_xt_code))
        if missing_detail_codes:
            log_event(
                "warning",
                "daily kline instrument details missing from bulk response",
                missing_count=len(missing_detail_codes),
                missing_codes=missing_detail_codes,
            )
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
            latest_state_by_code=latest_state_map,
            max_attempts=max_attempts,
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
        pending_docs: list[dict[str, Any]] = []
        write_batches = 0

        def write_docs(docs: list[dict[str, Any]]) -> None:
            nonlocal xtquant_docs_count, write_batches
            if not docs:
                return
            if not dry_run:
                write_day_kline_docs(cfg, DAY_COLLECTION, docs)
                write_batches += 1
            xtquant_docs_count += add_written_doc_stats(
                docs,
                written_dates_by_code,
                updated_by_date,
            )

        def buffer_docs(docs: list[dict[str, Any]]) -> None:
            nonlocal xtquant_docs_count, pending_docs, write_batches
            if not docs:
                return
            offset = 0
            if pending_docs:
                needed = batch_size - len(pending_docs)
                pending_docs.extend(docs[:needed])
                offset = min(needed, len(docs))
                if len(pending_docs) == batch_size:
                    write_docs(pending_docs)
                    pending_docs = []
            full_end = offset + ((len(docs) - offset) // batch_size) * batch_size
            for start in range(offset, full_end, batch_size):
                write_docs(docs[start : start + batch_size])
            if full_end < len(docs):
                pending_docs = docs[full_end:]

        for batch_start, batch_metas in iter_xt_sync_batches(update_metas, code_start_map, xt_batch_size=xt_batch_size):
            # 3. 先按增量起点分组，再分批拉 xtquant，避免一只长缺口股票带着整批回拉历史。
            if not batch_metas:
                continue
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
                buffer_docs(docs)
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
        write_docs(pending_docs)
        pending_docs = []

        raw_missing_by_code = build_missing_by_code(
            expected_dates_by_code,
            written_dates_by_code,
            all_invalid_dates,
        )
        raw_missing_days = sum(
            len(set(values))
            for values in raw_missing_by_code.values()
        )
        if raw_missing_days:
            log_event(
                "warning",
                "xtquant_missing_detected",
                stock_count=len(raw_missing_by_code),
                missing_days=raw_missing_days,
            )
        recent_missing_by_code, historical_missing_by_code = split_missing_by_fallback_window(
            raw_missing_by_code,
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
            dry_run=dry_run,
        )
        updated_by_date = add_fallback_counts_by_date(updated_by_date, fallback_summary)
        missing_by_code = unresolved_after_fallback(
            raw_missing_by_code,
            fallback_summary,
        )
        missing_today_codes = sorted(
            code
            for code, dates in missing_by_code.items()
            if end_date in set(dates)
        )
        day_kline_updated_count = xtquant_docs_count + int(fallback_summary.get("fallback_docs", 0)) + int(
            fallback_summary.get("normal_suspend_days", 0)
        )
        updated_latest_codes = {
            code
            for code, dates in written_dates_by_code.items()
            if end_date in dates and is_hs_a_share_code(code)
        }
        latest_text = end_date.strftime("%Y-%m-%d")
        updated_latest_codes.update(
            code
            for code, dates in fallback_summary.get("resolved_dates_by_code", {}).items()
            if latest_text in set(dates) and is_hs_a_share_code(code)
        )

        summary_builder.day_kline_updated_count = day_kline_updated_count
        summary_builder.xtquant_docs_count = xtquant_docs_count
        summary_builder.day_kline_updated_by_date = updated_by_date
        summary_builder.day_kline_missing_codes_today = missing_today_codes
        summary_builder.day_kline_updated_stock_count = len(updated_latest_codes)
        summary_builder.updated_stock_count_diff_vs_previous_trade_date = (
            len(updated_latest_codes) - previous_trade_date_stock_count
        )
        summary_builder.raw_missing_days = raw_missing_days
        summary_builder.missing_by_code = missing_by_code
        summary_builder.fallback_summary = fallback_summary
        summary_builder.historical_missing_rows = historical_missing_rows
        set_daily_unit_counts(
            summary_builder,
            expected_codes=set(expected_dates_by_code),
            unresolved_codes=set(missing_by_code),
        )
        result = summary_builder.to_dict()
        result["write_batches"] = write_batches + int(fallback_summary.get("write_batches", 0))
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
            basic_missing_ipo_count=len(result["basic_missing_ipo_codes"]),
            adjust_factor_baseline_written_count=result["adjust_factor_baseline_written_count"],
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
