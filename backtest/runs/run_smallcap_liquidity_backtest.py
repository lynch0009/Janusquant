from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.data import CachedMongoDataPortal, FrameCache, MongoDataPortal, ResearchDailyHistoryStore
from backtest.db import MongoDBConfig
from backtest.execution import EngineConfig, SignalDrivenBacktestEngine
from backtest.execution.smallcap_rotation_executor import SmallCapRotationDailyOpenExecutor
from backtest.portfolio import EqualSlotSizer
from backtest.risk import CloseBelowMaExitPolicy, CompositeExitPolicy, FixedStopLossExitPolicy
from backtest.strategies.smallcap_liquidity_rotation import SmallCapLiquidityRotationStrategy
from backtest.utils.log import log_event

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "smallcap_liquidity_backtest"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "smallcap_liquidity"


def build_run_output_dir(base_dir: Path, start_date: datetime, end_date: datetime) -> Path:
    """按回测区间和本次运行时间生成独立输出目录，避免覆盖历史结果。"""

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}_{run_timestamp}"
    return base_dir / run_name


def parse_int_tuple(raw_value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in str(raw_value).split(",") if part.strip()}))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the small-cap liquidity-cleaned rotation backtest.")
    parser.add_argument("--start-date", default="2025-01-01", help="Inclusive backtest start date, format YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-04-04", help="Inclusive backtest end date.")
    parser.add_argument("--initial-cash", type=float, default=1_000_000, help="Initial portfolio cash.")
    parser.add_argument("--top-k", type=int, default=10, help="Target number of holdings per rebalance.")
    parser.add_argument("--max-positions", type=int, default=10, help="Maximum simultaneous positions.")
    parser.add_argument("--hold-days", type=int, default=20, help="Holding period in trade dates.")
    parser.add_argument("--rebalance-every", type=int, default=20, help="Rebalance cadence in trade dates.")
    parser.add_argument("--min-listing-trade-days", type=int, default=120, help="Minimum listing age in trade dates.")
    parser.add_argument("--candidate-pool-size", type=int, default=100, help="Small-cap candidate pool size before liquidity filtering.")
    parser.add_argument("--liquidity-window", type=int, default=20, help="Rolling window used to compute average liquidity metrics.")
    parser.add_argument("--min-avg-amount", type=float, default=20_000_000, help="Minimum rolling average daily amount.")
    parser.add_argument(
        "--min-avg-turn",
        type=float,
        default=2.0,
        help="Minimum rolling average daily turnover in percent points. For example, 1 means 1%%. Legacy decimal inputs like 0.01 are auto-converted.",
    )
    parser.add_argument(
        "--exclude-bottom-liquidity-pct",
        type=float,
        default=0.15,
        help="Cross-sectional exclusion ratio within each trade date. For example, 0.15 removes the bottom 15%%.",
    )
    parser.add_argument("--min-close-price", type=float, default=None, help="Optional minimum close price filter.")
    parser.add_argument("--factor-filter-enabled", action="store_true", help="Enable HHV middle-group factor filter.")
    parser.add_argument("--factor-sort-enabled", action="store_true", help="Enable amount_expand factor sort.")
    parser.add_argument(
        "--amount-expand-descending",
        action="store_true",
        help="When factor sort is enabled, sort amount_expand descending instead of ascending.",
    )
    parser.add_argument("--hhv-window", type=int, default=60, help="HHV window used by distance_to_hhv factor.")
    parser.add_argument("--hhv-group-count", type=int, default=5, help="Daily cross-sectional HHV factor group count.")
    parser.add_argument("--hhv-keep-groups", default="2,3,4", help="Comma separated HHV groups to keep after factor grouping.")
    parser.add_argument("--amount-expand-fast-window", type=int, default=5, help="Fast rolling amount window for amount_expand.")
    parser.add_argument("--amount-expand-slow-window", type=int, default=20, help="Slow rolling amount window for amount_expand.")
    parser.add_argument("--slippage-bps", type=float, default=30.0, help="Buy/sell slippage in basis points.")
    parser.add_argument("--fixed-stop-loss-pct", type=float, default=None, help="Fixed intraday stop-loss percentage, for example 0.08.")
    parser.add_argument("--ma-stop-window", type=int, default=None, help="Close-below-MA stop window, for example 20.")
    parser.add_argument("--research-price-mode", default="hfq", choices=["raw", "qfq", "hfq"], help="Shared research price mode used by close-confirmed MA stops.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory used to export result tables.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable shared parquet cache and force database reads/recomputation.")
    parser.add_argument("--cache-version", default="v1", help="Manual cache version. Change it to invalidate old cached frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    requested_end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()
    hhv_keep_groups = parse_int_tuple(args.hhv_keep_groups)
    args.hhv_keep_groups = hhv_keep_groups

    db_client = MongoDBConfig()
    frame_cache = None if args.disable_cache else FrameCache(DEFAULT_CACHE_DIR, version=args.cache_version)
    data_portal = (
        MongoDataPortal(db_client)
        if frame_cache is None
        else CachedMongoDataPortal(db_client, frame_cache=frame_cache)
    )
    trade_dates = data_portal.get_trade_calendar(start_date, requested_end_date)
    if not trade_dates:
        raise ValueError("No trade dates available in the requested date range.")

    resolved_end_date = trade_dates[-1]
    log_event(
        "info",
        "smallcap_liquidity_run_start",
        start_date=start_date,
        requested_end_date=requested_end_date,
        resolved_end_date=resolved_end_date,
        initial_cash=args.initial_cash,
        top_k=args.top_k,
        max_positions=args.max_positions,
        hold_days=args.hold_days,
        rebalance_every=args.rebalance_every,
        candidate_pool_size=args.candidate_pool_size,
        liquidity_window=args.liquidity_window,
        min_avg_amount=args.min_avg_amount,
        min_avg_turn=args.min_avg_turn,
        exclude_bottom_liquidity_pct=args.exclude_bottom_liquidity_pct,
        min_close_price=args.min_close_price,
        factor_filter_enabled=args.factor_filter_enabled,
        factor_sort_enabled=args.factor_sort_enabled,
        amount_expand_descending=args.amount_expand_descending,
        hhv_window=args.hhv_window,
        hhv_group_count=args.hhv_group_count,
        hhv_keep_groups=list(hhv_keep_groups),
        amount_expand_fast_window=args.amount_expand_fast_window,
        amount_expand_slow_window=args.amount_expand_slow_window,
        slippage_bps=args.slippage_bps,
        fixed_stop_loss_pct=args.fixed_stop_loss_pct,
        ma_stop_window=args.ma_stop_window,
        research_price_mode=args.research_price_mode,
        cache_enabled=frame_cache is not None,
        cache_dir=str(DEFAULT_CACHE_DIR),
        cache_version=args.cache_version,
    )

    strategy = SmallCapLiquidityRotationStrategy(
        top_k=args.top_k,
        hold_days=args.hold_days,
        rebalance_every_n_trade_days=args.rebalance_every,
        min_listing_trade_days=args.min_listing_trade_days,
        candidate_pool_size=args.candidate_pool_size,
        liquidity_window=args.liquidity_window,
        min_avg_amount=args.min_avg_amount,
        min_avg_turn=args.min_avg_turn,
        exclude_bottom_liquidity_pct=args.exclude_bottom_liquidity_pct,
        min_close_price=args.min_close_price,
        factor_filter_enabled=args.factor_filter_enabled,
        factor_sort_enabled=args.factor_sort_enabled,
        amount_expand_descending=args.amount_expand_descending,
        hhv_window=args.hhv_window,
        hhv_group_count=args.hhv_group_count,
        hhv_keep_groups=hhv_keep_groups,
        amount_expand_fast_window=args.amount_expand_fast_window,
        amount_expand_slow_window=args.amount_expand_slow_window,
    )
    execution_model = SmallCapRotationDailyOpenExecutor(slippage_bps=args.slippage_bps)

    exit_policies = []
    if args.fixed_stop_loss_pct is not None:
        exit_policies.append(FixedStopLossExitPolicy(stop_loss_pct=args.fixed_stop_loss_pct))
    if args.ma_stop_window is not None:
        exit_policies.append(
            CloseBelowMaExitPolicy(
                ma_window=args.ma_stop_window,
                price_mode=args.research_price_mode,
            )
        )

    exit_policy = None
    if len(exit_policies) == 1:
        exit_policy = exit_policies[0]
    elif len(exit_policies) > 1:
        exit_policy = CompositeExitPolicy(exit_policies)

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
        execution_model=execution_model,
        config=config,
        position_sizer=EqualSlotSizer(),
        exit_policy=exit_policy,
        data_portal=data_portal,
    )

    result = engine.run(start_date, resolved_end_date, research_store=ResearchDailyHistoryStore(data_portal))
    report = result.analyze()
    cache_summary = frame_cache.summary() if frame_cache is not None else {}
    log_event(
        "info",
        "smallcap_liquidity_run_finish",
        final_equity=report.summary.get("final_equity"),
        total_return=report.summary.get("total_return"),
        trade_count=report.summary.get("trade_count"),
        filled_order_count=report.summary.get("filled_order_count"),
        skipped_order_count=report.summary.get("skipped_order_count"),
        cache_enabled=frame_cache is not None,
        cache_summary=cache_summary,
    )

    output_dir = build_run_output_dir(Path(args.output_dir), start_date, resolved_end_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.orders_frame().to_csv(output_dir / "orders.csv", index=False)
    result.trades_frame().to_csv(output_dir / "trades.csv", index=False)
    result.equity_frame().to_csv(output_dir / "equity.csv", index=False)
    result.closed_positions_frame().to_csv(output_dir / "closed_positions.csv", index=False)
    report.export(
        output_dir / "analytics",
        metadata={
            "strategy_name": "smallcap_liquidity_backtest",
            "script_name": Path(__file__).name,
            "date_range": f"{start_date:%Y-%m-%d} -> {resolved_end_date:%Y-%m-%d}",
            "cache_summary": cache_summary,
            "parameters": {
                **vars(args),
                "resolved_end_date": resolved_end_date.strftime("%Y-%m-%d"),
            },
        },
    )

    print(json.dumps(report.summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
