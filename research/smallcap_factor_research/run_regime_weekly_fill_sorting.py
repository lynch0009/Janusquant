from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.data import DuckDBDataPortal
from backtest.db import DuckDBConfig
from research.smallcap_factor_research.weekly_regime import (
    RegimeWeeklyFillResearchConfig,
    run_regime_weekly_fill_weekday_batch,
    run_regime_weekly_fill_research,
)
from research.smallcap_factor_research.weekly_regime.config import (
    DEFAULT_HORIZONS,
    DEFAULT_TOP_NS,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output_regime_weekly_fill_sorting"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEFAULT_REGIME_RUN_DIR = (
    Path("backtest")
    / "runs"
    / "output"
    / "smallcap_amount_shock_event_regime_hold_v1"
    / "20200101_20260521_20260611_223835"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research weekly fill sorting factors inside active small-cap event regime.")
    parser.add_argument("--regime-run-dir", default=str(DEFAULT_REGIME_RUN_DIR), help="Directory containing regime_state.csv.")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--candidate-pool-size", type=int, default=200)
    parser.add_argument("--price-mode", default="qfq", choices=["raw", "qfq", "hfq"])
    parser.add_argument("--min-listing-trade-days", type=int, default=120)
    parser.add_argument("--min-close-price", type=float, default=1.5)
    parser.add_argument("--index-code", default="sz.399303")
    parser.add_argument("--horizons", default=",".join(str(value) for value in DEFAULT_HORIZONS))
    parser.add_argument("--top-ns", default=",".join(str(value) for value in DEFAULT_TOP_NS))
    parser.add_argument("--bucket-count", type=int, default=5)
    parser.add_argument("--feature-set", default="base", choices=["base", "trend_quality_core"])
    parser.add_argument("--weekly-fill-weekday", type=int, default=4, help="0=Monday ... 4=Friday.")
    parser.add_argument(
        "--weekly-fill-weekdays",
        default="",
        help="Comma-separated weekdays for batch research, e.g. Monday,Tuesday,Wednesday,Thursday,Friday or 0,1,2,3,4.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--disable-cache", action="store_true")
    return parser.parse_args()


def parse_weekdays(raw_value: str) -> tuple[int, ...]:
    if raw_value is None or not str(raw_value).strip():
        return ()
    name_map = {
        "0": 0,
        "mon": 0,
        "monday": 0,
        "1": 1,
        "tue": 1,
        "tues": 1,
        "tuesday": 1,
        "2": 2,
        "wed": 2,
        "wednesday": 2,
        "3": 3,
        "thu": 3,
        "thur": 3,
        "thurs": 3,
        "thursday": 3,
        "4": 4,
        "fri": 4,
        "friday": 4,
    }
    weekdays: list[int] = []
    for token in str(raw_value).split(","):
        normalized = token.strip().lower()
        if not normalized:
            continue
        if normalized not in name_map:
            raise ValueError(f"unsupported weekday token: {token}")
        weekday = name_map[normalized]
        if weekday not in weekdays:
            weekdays.append(weekday)
    return tuple(weekdays)


def build_config(args: argparse.Namespace) -> RegimeWeeklyFillResearchConfig:
    regime_run_dir = Path(args.regime_run_dir)
    if not regime_run_dir.exists():
        raise FileNotFoundError(f"regime run dir does not exist: {regime_run_dir}")
    if int(args.candidate_pool_size) < 1:
        raise ValueError("candidate_pool_size must be >= 1")
    if int(args.weekly_fill_weekday) < 0 or int(args.weekly_fill_weekday) > 4:
        raise ValueError("weekly_fill_weekday must be within 0..4")
    horizons = tuple(int(value.strip()) for value in str(args.horizons).split(",") if value.strip())
    top_ns = tuple(int(value.strip()) for value in str(args.top_ns).split(",") if value.strip())
    if not horizons:
        raise ValueError("horizons cannot be empty")
    if not top_ns:
        raise ValueError("top_ns cannot be empty")
    return RegimeWeeklyFillResearchConfig(
        regime_run_dir=regime_run_dir,
        output_dir=Path(args.output_dir),
        start_date=datetime.strptime(args.start_date, "%Y-%m-%d"),
        end_date=datetime.strptime(args.end_date, "%Y-%m-%d"),
        candidate_pool_size=int(args.candidate_pool_size),
        price_mode=str(args.price_mode),
        min_listing_trade_days=int(args.min_listing_trade_days),
        min_close_price=float(args.min_close_price),
        index_code=str(args.index_code),
        cache_dir=None if args.disable_cache else Path(args.cache_dir),
        horizons=horizons,
        top_ns=top_ns,
        bucket_count=int(args.bucket_count),
        weekly_fill_weekday=int(args.weekly_fill_weekday),
        feature_set=str(args.feature_set),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    batch_weekdays = parse_weekdays(args.weekly_fill_weekdays)
    db_client = DuckDBConfig()
    try:
        portal = DuckDBDataPortal(db_client)
        if batch_weekdays:
            result = run_regime_weekly_fill_weekday_batch(portal, config, weekdays=batch_weekdays)
        else:
            result = run_regime_weekly_fill_research(portal, config)
        print(f"output_dir={result.output_dir.resolve()}")
        print(result.summary)
    finally:
        db_client.close()


if __name__ == "__main__":
    main()
