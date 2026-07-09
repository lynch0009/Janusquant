from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.data import CachedDuckDBDataPortal, DuckDBDataPortal, FrameCache, ResearchDailyHistoryStore
from backtest.db import DuckDBConfig
from backtest.execution import EngineConfig, SignalDrivenBacktestEngine
from backtest.execution.smallcap_rotation_executor import SmallCapRotationDailyOpenExecutor
from backtest.portfolio import EqualSlotSizer
from backtest.risk import AbsoluteLowPriceExitPolicy
from backtest.strategies.smallcap_amount_shock_reversal import SmallCapAmountShockReversalStrategy
from backtest.utils.log import log_event

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "smallcap_amount_shock_reversal"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "smallcap_amount_shock_reversal"


def parse_int_tuple(raw_value: str) -> tuple[int, ...]:
    if raw_value is None or not str(raw_value).strip():
        return ()
    return tuple(sorted({int(part.strip()) for part in str(raw_value).split(",") if part.strip()}))


def parse_optional_float(raw_value) -> float | None:
    text = str(raw_value or "").strip().lower()
    if text in {"", "none", "null", "na"}:
        return None
    return float(text)


def parse_optional_int(raw_value) -> int | None:
    text = str(raw_value or "").strip().lower()
    if text in {"", "none", "null", "na"}:
        return None
    return int(text)


def build_run_output_dir(base_dir: Path, start_date: datetime, end_date: datetime) -> Path:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{start_date:%Y%m%d}_{end_date:%Y%m%d}_{run_timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the small-cap amount-shock reversal backtest.")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-04-24")
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument("--rebalance-every", type=int, default=10)
    parser.add_argument("--candidate-pool-size", type=int, default=150)
    parser.add_argument("--min-listing-trade-days", type=int, default=120)
    parser.add_argument("--amount-keep-groups", default="5", help="Empty string disables amount shock filtering.")
    parser.add_argument("--ret-keep-groups", default="5", help="Empty string disables ret_10d group filtering and sorts by liqaMV.")
    parser.add_argument(
        "--min-research-ret-10d",
        type=float,
        default=None,
        help="Optional minimum research_ret_10d threshold, for example 0.10 means ret_10d <= -10%.",
    )
    parser.add_argument("--selection-sort", default="ret_desc", choices=["ret_desc", "cap_asc"])
    parser.add_argument("--group-count", type=int, default=5)
    parser.add_argument("--amount-fast-window", type=int, default=5)
    parser.add_argument("--amount-slow-window", type=int, default=20)
    parser.add_argument("--ret-window", type=int, default=10)
    parser.add_argument("--slippage-bps", type=float, default=30.0)
    parser.add_argument("--signal-price-mode", default="hfq", choices=["raw", "qfq", "hfq"])
    parser.add_argument("--st-lookback-trade-days", default="100")
    parser.add_argument("--min-signal-close-price", default="1.5")
    parser.add_argument("--low-price-exit-threshold", default="1.3")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--cache-version", default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    requested_end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    db_client = DuckDBConfig()
    frame_cache = None if args.disable_cache else FrameCache(DEFAULT_CACHE_DIR, version=args.cache_version)
    data_portal = DuckDBDataPortal(db_client) if frame_cache is None else CachedDuckDBDataPortal(db_client, frame_cache=frame_cache)
    trade_dates = data_portal.get_trade_calendar(start_date, requested_end_date)
    if not trade_dates:
        raise ValueError("No trade dates available in the requested date range.")
    resolved_end_date = trade_dates[-1]

    amount_keep_groups = parse_int_tuple(args.amount_keep_groups)
    ret_keep_groups = parse_int_tuple(args.ret_keep_groups)
    st_lookback_trade_days = parse_optional_int(args.st_lookback_trade_days)
    min_signal_close_price = parse_optional_float(args.min_signal_close_price)
    low_price_exit_threshold = parse_optional_float(args.low_price_exit_threshold)
    log_event(
        "info",
        "smallcap_amount_shock_reversal_run_start",
        start_date=start_date,
        resolved_end_date=resolved_end_date,
        candidate_pool_size=args.candidate_pool_size,
        top_k=args.top_k,
        max_positions=args.max_positions,
        hold_days=args.hold_days,
        rebalance_every=args.rebalance_every,
        amount_keep_groups=list(amount_keep_groups),
        ret_keep_groups=list(ret_keep_groups),
        min_research_ret_10d=args.min_research_ret_10d,
        st_lookback_trade_days=st_lookback_trade_days,
        min_signal_close_price=min_signal_close_price,
        low_price_exit_threshold=low_price_exit_threshold,
        slippage_bps=args.slippage_bps,
        signal_price_mode=args.signal_price_mode,
        cache_enabled=frame_cache is not None,
        cache_dir=str(DEFAULT_CACHE_DIR),
        cache_version=args.cache_version,
    )

    strategy = SmallCapAmountShockReversalStrategy(
        top_k=args.top_k,
        hold_days=args.hold_days,
        rebalance_every_n_trade_days=args.rebalance_every,
        min_listing_trade_days=args.min_listing_trade_days,
        candidate_pool_size=args.candidate_pool_size,
        amount_fast_window=args.amount_fast_window,
        amount_slow_window=args.amount_slow_window,
        ret_window=args.ret_window,
        group_count=args.group_count,
        amount_keep_groups=amount_keep_groups,
        ret_keep_groups=ret_keep_groups,
        min_research_ret_10d=args.min_research_ret_10d,
        selection_sort=args.selection_sort,
        signal_price_mode=args.signal_price_mode,
        st_lookback_trade_days=st_lookback_trade_days,
        min_signal_close_price=min_signal_close_price,
    )
    exit_policy = (
        AbsoluteLowPriceExitPolicy(min_low_price=low_price_exit_threshold)
        if low_price_exit_threshold is not None
        else None
    )
    config = EngineConfig(
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
        position_size_pct=1 / max(args.max_positions, 1),
        execute_on_next_trade_date=True,
        progress_logging=True,
    )
    engine = SignalDrivenBacktestEngine(
        db_client,
        strategy,
        execution_model=SmallCapRotationDailyOpenExecutor(slippage_bps=args.slippage_bps),
        config=config,
        position_sizer=EqualSlotSizer(),
        data_portal=data_portal,
        exit_policy=exit_policy,
    )

    result = engine.run(start_date, resolved_end_date, research_store=ResearchDailyHistoryStore(data_portal))
    report = result.analyze()
    cache_summary = frame_cache.summary() if frame_cache is not None else {}

    output_dir = build_run_output_dir(Path(args.output_dir), start_date, resolved_end_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.orders_frame().to_csv(output_dir / "orders.csv", index=False)
    result.trades_frame().to_csv(output_dir / "trades.csv", index=False)
    result.equity_frame().to_csv(output_dir / "equity.csv", index=False)
    result.closed_positions_frame().to_csv(output_dir / "closed_positions.csv", index=False)
    report.export(
        output_dir / "analytics",
        metadata={
            "strategy_name": "smallcap_amount_shock_reversal",
            "script_name": Path(__file__).name,
            "date_range": f"{start_date:%Y-%m-%d} -> {resolved_end_date:%Y-%m-%d}",
            "cache_summary": cache_summary,
            "parameters": {
                **vars(args),
                "amount_keep_groups": list(amount_keep_groups),
                "ret_keep_groups": list(ret_keep_groups),
                "min_research_ret_10d": args.min_research_ret_10d,
                "st_lookback_trade_days": st_lookback_trade_days,
                "min_signal_close_price": min_signal_close_price,
                "low_price_exit_threshold": low_price_exit_threshold,
                "resolved_end_date": resolved_end_date.strftime("%Y-%m-%d"),
            },
        },
    )

    print(json.dumps(report.summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
