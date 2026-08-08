from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .constants import FALLBACK_WINDOW_TRADE_DAYS, MIN_DAY_KLINE_START_DATE


@dataclass(frozen=True)
class StockMeta:
    code: str
    xt_code: str
    code_name: str
    ipo_date: datetime | None
    out_date: datetime | None
    float_volume: float | None = None


@dataclass(frozen=True)
class DateRange:
    start: datetime
    end: datetime

    @property
    def count_label(self) -> str:
        if self.start == self.end:
            return self.start.strftime("%Y-%m-%d")
        return f"{self.start:%Y-%m-%d}->{self.end:%Y-%m-%d}"


@dataclass(frozen=True)
class XtDayRow:
    doc: dict[str, Any] | None
    date: datetime | None
    is_suspended: bool = False
    missing_reason: str = ""


@dataclass(frozen=True)
class BasicInfoSyncResult:
    inserted_docs: tuple[dict[str, Any], ...]
    detail_failed_codes: tuple[str, ...]
    stock_pool_diff_rows: tuple[dict[str, Any], ...]
    db_operation_summary: dict[str, int]
    missing_ipo_codes: tuple[str, ...] = ()
    adjust_factor_baseline_planned_count: int = 0
    adjust_factor_baseline_existing_count: int = 0
    adjust_factor_baseline_written_count: int = 0
    detail_requested_codes: tuple[str, ...] = ()
    instrument_detail_requested_count: int = 0
    db_only_confirmed_delisted_codes: tuple[str, ...] = ()
    db_only_active_detail_codes: tuple[str, ...] = ()
    db_only_detail_missing_codes: tuple[str, ...] = ()
    confirmed_delisted_updates: tuple[dict[str, Any], ...] = ()
    details_by_xt_code: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def inserted_codes(self) -> list[str]:
        return [doc["code"] for doc in self.inserted_docs]


def empty_fallback_summary() -> dict[str, Any]:
    return {
        "fallback_docs": 0,
        "unresolved_days": 0,
        "normal_suspend_days": 0,
        "skipped_by_threshold": 0,
        "write_batches": 0,
        "resolved_dates_by_code": {},
        "unresolved_dates_by_code": {},
    }


@dataclass
class DailySyncSummaryBuilder:
    basic_result: BasicInfoSyncResult
    bj_skipped_stock_count: int
    universe_stats: dict[str, int]
    update_universe_count: int
    stock_count: int
    fixed_index_count: int = 0
    active_stock_count: int = 0
    day_kline_updated_count: int = 0
    xtquant_docs_count: int = 0
    day_kline_updated_by_date: dict[str, int] = field(default_factory=dict)
    day_kline_missing_codes_today: list[str] = field(default_factory=list)
    day_kline_updated_stock_count: int = 0
    previous_trade_date: str | None = None
    previous_trade_date_stock_count: int = 0
    updated_stock_count_diff_vs_previous_trade_date: int = 0
    raw_missing_days: int = 0
    missing_by_code: dict[str, list[datetime]] = field(default_factory=dict)
    fallback_summary: dict[str, Any] = field(default_factory=empty_fallback_summary)
    historical_missing_rows: list[dict[str, str]] = field(default_factory=list)
    report_dir: str | None = None
    unit_planned_count: int = 0
    unit_attempted_count: int = 0
    unit_succeeded_count: int = 0
    unit_skipped_count: int = 0
    unit_failed_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        historical_codes = {row["code"] for row in self.historical_missing_rows}
        return {
            "basic_inserted_codes": self.basic_result.inserted_codes,
            "basic_inserted_count": len(self.basic_result.inserted_docs),
            "basic_detail_failed_codes": list(self.basic_result.detail_failed_codes),
            "basic_missing_ipo_codes": list(self.basic_result.missing_ipo_codes),
            "adjust_factor_baseline_planned_count": self.basic_result.adjust_factor_baseline_planned_count,
            "adjust_factor_baseline_existing_count": self.basic_result.adjust_factor_baseline_existing_count,
            "adjust_factor_baseline_written_count": self.basic_result.adjust_factor_baseline_written_count,
            "stock_pool_xt_only_count": sum(
                1 for row in self.basic_result.stock_pool_diff_rows if row["diff_type"] == "xt_only"
            ),
            "stock_pool_db_only_count": sum(
                1 for row in self.basic_result.stock_pool_diff_rows if row["diff_type"] == "db_only"
            ),
            "db_only_detail_requested_count": (
                len(self.basic_result.db_only_confirmed_delisted_codes)
                + len(self.basic_result.db_only_active_detail_codes)
                + len(self.basic_result.db_only_detail_missing_codes)
            ),
            "instrument_detail_requested_count": self.basic_result.instrument_detail_requested_count,
            "db_only_confirmed_delisted_count": len(
                self.basic_result.db_only_confirmed_delisted_codes
            ),
            "db_only_confirmed_delisted_codes": list(
                self.basic_result.db_only_confirmed_delisted_codes
            ),
            "db_only_active_detail_count": len(
                self.basic_result.db_only_active_detail_codes
            ),
            "db_only_active_detail_codes": list(
                self.basic_result.db_only_active_detail_codes
            ),
            "db_only_detail_missing_count": len(
                self.basic_result.db_only_detail_missing_codes
            ),
            "db_only_detail_missing_codes": list(
                self.basic_result.db_only_detail_missing_codes
            ),
            "db_only_warning_count": (
                len(self.basic_result.db_only_active_detail_codes)
                + len(self.basic_result.db_only_detail_missing_codes)
            ),
            "day_kline_updated_count": self.day_kline_updated_count,
            "day_kline_updated_by_date": dict(sorted(self.day_kline_updated_by_date.items())),
            "day_kline_missing_codes_today": self.day_kline_missing_codes_today,
            "day_kline_missing_today_count": len(self.day_kline_missing_codes_today),
            "day_kline_updated_stock_count": self.day_kline_updated_stock_count,
            "previous_trade_date": self.previous_trade_date,
            "previous_trade_date_stock_count": self.previous_trade_date_stock_count,
            "updated_stock_count_diff_vs_previous_trade_date": (
                self.updated_stock_count_diff_vs_previous_trade_date
            ),
            "day_kline_update_universe_count": self.update_universe_count,
            "fixed_index_update_universe_count": self.fixed_index_count,
            "day_kline_min_start_date": MIN_DAY_KLINE_START_DATE.strftime("%Y-%m-%d"),
            **self.universe_stats,
            "bj_skipped_stock_count": self.bj_skipped_stock_count,
            "fallback_window_trade_days": FALLBACK_WINDOW_TRADE_DAYS,
            "fallback_summary": self.fallback_summary,
            "historical_missing_days": len(self.historical_missing_rows),
            "historical_missing_stock_count": len(historical_codes),
            "historical_missing_detail_path": None,
            "report_dir": self.report_dir,
            "stock_count": self.stock_count,
            "active_stock_count": self.active_stock_count,
            "xtquant_docs": self.xtquant_docs_count,
            "missing_stock_count": len(self.missing_by_code),
            "raw_missing_days": self.raw_missing_days,
            "resolved_fallback_days": sum(
                len(set(values))
                for values in self.fallback_summary.get("resolved_dates_by_code", {}).values()
            ),
            "normal_suspend_days": int(self.fallback_summary.get("normal_suspend_days", 0)),
            "unresolved_after_fallback_days": sum(
                len(set(values)) for values in self.missing_by_code.values()
            ),
            "missing_days": sum(len(set(values)) for values in self.missing_by_code.values()),
            "unit_planned_count": self.unit_planned_count,
            "unit_attempted_count": self.unit_attempted_count,
            "unit_succeeded_count": self.unit_succeeded_count,
            "unit_skipped_count": self.unit_skipped_count,
            "unit_failed_count": self.unit_failed_count,
        }


@dataclass(frozen=True)
class SecurityMeta:
    """统一描述一个待更新标的。股票来自 basic_info，指数来自 day_kline 历史存量。"""

    code: str
    code_name: str
    ipo_date: datetime | None
    out_date: datetime | None
    is_index: bool = False
