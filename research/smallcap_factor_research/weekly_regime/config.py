"""Configuration and constants for weekly regime research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_FEATURES = (
    "ret_1d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "amount_expand",
    "ma10",
    "ma20",
    "ma20_slope_5d",
)
TREND_QUALITY_CORE_FEATURES = (
    "pullback_from_hhv20",
    "max_drawdown_20d",
    "up_day_ratio_20d",
    "up_amount_ratio_20d",
    "down_shrink_20d",
)
TREND_QUALITY_CORE_FACTORS = (
    "current_composite_zscore",
    "pullback_from_hhv20",
    "max_drawdown_20d",
    "up_day_ratio_20d",
    "up_amount_ratio_20d",
    "down_shrink_20d",
    "rs_positive_days_20d",
)
DEFAULT_HORIZONS = (5, 10, 20, 40)
DEFAULT_TOP_NS = (1, 2, 3, 5, 10)
FACTOR_NAMES = (
    "current_composite_zscore",
    "ret20_desc",
    "ret60_desc",
    "rs20_desc",
    "amount_expand_desc",
    "smallcap_trend",
    "trend_composite",
    "trend_no_amount",
    "trend_no_cap",
)
CONDITION_NAMES = (
    "none",
    "close_gt_ma20",
    "close_gt_ma10_and_ma20",
    "ma20_slope_gt_0",
    "ret20_gt_0",
    "rs20_gt_0",
    "close_gt_ma20_and_rs20_gt_0",
    "trend_quality_all",
)
FOCUS_PERIODS = (
    ("trend_2021_02_09", "2021-02-01", "2021-09-30"),
    ("trend_2022_05_08", "2022-05-01", "2022-08-31"),
    ("trend_2025_04_2026_01", "2025-04-01", "2026-01-31"),
    ("bad_2024_04", "2024-04-01", "2024-04-30"),
    ("bad_2024_10_12", "2024-10-01", "2024-12-31"),
)
WEEKDAY_NAMES = {
    0: "monday",
    1: "tuesday",
    2: "wednesday",
    3: "thursday",
    4: "friday",
}
PRIMARY_COMPARISON_FACTORS = ("current_composite_zscore",)
PRIMARY_COMPARISON_CONDITIONS = ("none", "ma20_slope_gt_0", "close_gt_ma20", "rs20_gt_0")
PRIMARY_COMPARISON_HORIZONS = (20, 40)
PRIMARY_COMPARISON_TOP_NS = (3, 5)


@dataclass(frozen=True)
class RegimeWeeklyFillResearchConfig:
    regime_run_dir: Path
    output_dir: Path
    start_date: datetime
    end_date: datetime
    candidate_pool_size: int = 200
    price_mode: str = "qfq"
    min_listing_trade_days: int = 120
    min_close_price: float = 1.5
    index_code: str = "sz.399303"
    cache_dir: Path | None = None
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    top_ns: tuple[int, ...] = DEFAULT_TOP_NS
    bucket_count: int = 5
    weekly_fill_weekday: int = 4
    feature_set: str = "base"


def validate_weekday(value: int) -> int:
    weekday = int(value)
    if weekday not in WEEKDAY_NAMES:
        raise ValueError("weekly_fill_weekday must be within 0..4")
    return weekday


def normalize_feature_set(value: str | None) -> str:
    normalized = str(value or "base").strip().lower()
    if normalized in {"", "none", "default"}:
        normalized = "base"
    if normalized not in {"base", "trend_quality_core"}:
        raise ValueError(f"unsupported feature_set: {value}")
    return normalized


def features_for_feature_set(feature_set: str | None) -> tuple[str, ...]:
    normalized = normalize_feature_set(feature_set)
    if normalized == "trend_quality_core":
        return tuple(dict.fromkeys((*DEFAULT_FEATURES, *TREND_QUALITY_CORE_FEATURES)))
    return DEFAULT_FEATURES


def validate_config(config: RegimeWeeklyFillResearchConfig) -> None:
    if config.start_date > config.end_date:
        raise ValueError("start_date must be <= end_date")
    if int(config.candidate_pool_size) < 1:
        raise ValueError("candidate_pool_size must be >= 1")
    if int(config.min_listing_trade_days) < 0:
        raise ValueError("min_listing_trade_days must be >= 0")
    if float(config.min_close_price) < 0:
        raise ValueError("min_close_price must be >= 0")
    if int(config.bucket_count) < 2:
        raise ValueError("bucket_count must be >= 2")
    if not config.horizons or any(int(value) <= 0 for value in config.horizons):
        raise ValueError("horizons must contain positive integers")
    if not config.top_ns or any(int(value) <= 0 for value in config.top_ns):
        raise ValueError("top_ns must contain positive integers")
    validate_weekday(config.weekly_fill_weekday)
    normalize_feature_set(config.feature_set)


__all__ = [
    "CONDITION_NAMES",
    "DEFAULT_FEATURES",
    "DEFAULT_HORIZONS",
    "DEFAULT_TOP_NS",
    "FACTOR_NAMES",
    "FOCUS_PERIODS",
    "PRIMARY_COMPARISON_CONDITIONS",
    "PRIMARY_COMPARISON_FACTORS",
    "PRIMARY_COMPARISON_HORIZONS",
    "PRIMARY_COMPARISON_TOP_NS",
    "RegimeWeeklyFillResearchConfig",
    "TREND_QUALITY_CORE_FACTORS",
    "TREND_QUALITY_CORE_FEATURES",
    "WEEKDAY_NAMES",
    "features_for_feature_set",
    "normalize_feature_set",
    "validate_config",
    "validate_weekday",
]
