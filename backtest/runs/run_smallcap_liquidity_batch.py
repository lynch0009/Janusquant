from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

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

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "smallcap_liquidity_batch"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "smallcap_liquidity_batch"


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _slug_amount(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "_")


def _slug_float(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def _build_batch_output_dir(base_dir: Path, start_date: datetime, end_date: datetime) -> Path:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}_{run_timestamp}"
    return base_dir / run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a parameter grid for the small-cap liquidity-cleaned strategy.")
    parser.add_argument("--start-date", default="2025-01-01", help="Inclusive backtest start date, format YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-04-24", help="Inclusive backtest end date.")
    parser.add_argument("--initial-cash", type=float, default=1_000_000, help="Initial portfolio cash.")
    parser.add_argument("--top-k", type=int, default=10, help="Target number of holdings per rebalance.")
    parser.add_argument("--max-positions", type=int, default=10, help="Maximum simultaneous positions.")
    parser.add_argument("--hold-days", type=int, default=20, help="Holding period in trade dates.")
    parser.add_argument("--rebalance-every", type=int, default=20, help="Rebalance cadence in trade dates.")
    parser.add_argument("--min-listing-trade-days", type=int, default=120, help="Minimum listing age in trade dates.")
    parser.add_argument(
        "--liquidity-window",
        "--liquidity_window",
        default="20",
        help="Comma-separated rolling windows used to compute average liquidity metrics.",
    )
    parser.add_argument("--candidate-pool-sizes", default="100,200,150", help="Comma-separated candidate pool sizes.")
    parser.add_argument("--min-avg-amounts", default="30000000", help="Comma-separated average amount thresholds.")
    parser.add_argument(
        "--min-avg-turns",
        default="2",
        help="Comma-separated average turnover thresholds in percent points. For example, 1,2,3 means 1%%, 2%%, 3%%. Legacy decimal inputs like 0.01 are auto-converted.",
    )
    parser.add_argument(
        "--exclude-bottom-liquidity-pct",
        default="0.15",
        help="Comma-separated cross-sectional exclusion ratios. For example, 0.15 removes the bottom 15%%.",
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
    parser.add_argument("--research-price-mode", default="qfq", choices=["raw", "qfq", "hfq"], help="Shared research price mode used by close-confirmed MA stops.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory used to export result tables.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable shared parquet cache and force database reads/recomputation.")
    parser.add_argument("--cache-version", default="v1", help="Manual cache version. Change it to invalidate old cached frames.")
    return parser.parse_args()


def _build_exit_policy(args: argparse.Namespace):
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
    if len(exit_policies) == 1:
        return exit_policies[0]
    if len(exit_policies) > 1:
        return CompositeExitPolicy(exit_policies)
    return None


def main() -> None:
    args = parse_args()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    requested_end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()

    candidate_pool_sizes = _parse_int_list(args.candidate_pool_sizes)
    min_avg_amounts = _parse_float_list(args.min_avg_amounts)
    min_avg_turns = _parse_float_list(args.min_avg_turns)
    liquidity_windows = _parse_int_list(str(args.liquidity_window))
    exclude_bottom_liquidity_pcts = _parse_float_list(str(args.exclude_bottom_liquidity_pct))
    hhv_keep_groups = _parse_int_tuple(args.hhv_keep_groups)
    args.hhv_keep_groups = hhv_keep_groups
    if (
        not candidate_pool_sizes
        or not min_avg_amounts
        or not min_avg_turns
        or not liquidity_windows
        or not exclude_bottom_liquidity_pcts
    ):
        raise ValueError("Parameter grid cannot be empty.")

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
    batch_output_dir = _build_batch_output_dir(Path(args.output_dir), start_date, resolved_end_date)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    shared_research_store = ResearchDailyHistoryStore(data_portal)
    execution_model = SmallCapRotationDailyOpenExecutor(slippage_bps=args.slippage_bps)
    exit_policy = _build_exit_policy(args)

    summary_rows: list[dict[str, object]] = []
    param_grid = list(
        itertools.product(
            candidate_pool_sizes,
            min_avg_amounts,
            min_avg_turns,
            liquidity_windows,
            exclude_bottom_liquidity_pcts,
        )
    )
    log_event(
        "info",
        "smallcap_liquidity_batch_start",
        start_date=start_date,
        resolved_end_date=resolved_end_date,
        parameter_count=len(param_grid),
        candidate_pool_sizes=candidate_pool_sizes,
        min_avg_amounts=min_avg_amounts,
        min_avg_turns=min_avg_turns,
        liquidity_windows=liquidity_windows,
        exclude_bottom_liquidity_pcts=exclude_bottom_liquidity_pcts,
        top_k=args.top_k,
        max_positions=args.max_positions,
        factor_filter_enabled=args.factor_filter_enabled,
        factor_sort_enabled=args.factor_sort_enabled,
        amount_expand_descending=args.amount_expand_descending,
        hhv_window=args.hhv_window,
        hhv_group_count=args.hhv_group_count,
        hhv_keep_groups=list(hhv_keep_groups),
        amount_expand_fast_window=args.amount_expand_fast_window,
        amount_expand_slow_window=args.amount_expand_slow_window,
        cache_enabled=frame_cache is not None,
        cache_dir=str(DEFAULT_CACHE_DIR),
        cache_version=args.cache_version,
    )

    for run_index, (
        candidate_pool_size,
        min_avg_amount,
        min_avg_turn,
        liquidity_window,
        exclude_bottom_liquidity_pct,
    ) in enumerate(param_grid, start=1):
        run_name = (
            f"run_{run_index:03d}"
            f"_cp{candidate_pool_size}"
            f"_amt{_slug_amount(min_avg_amount)}"
            f"_turn{_slug_float(min_avg_turn)}"
            f"_lw{liquidity_window}"
            f"_ex{_slug_float(exclude_bottom_liquidity_pct)}"
        )
        run_output_dir = batch_output_dir / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        strategy = SmallCapLiquidityRotationStrategy(
            top_k=args.top_k,
            hold_days=args.hold_days,
            rebalance_every_n_trade_days=args.rebalance_every,
            min_listing_trade_days=args.min_listing_trade_days,
            candidate_pool_size=candidate_pool_size,
            liquidity_window=liquidity_window,
            min_avg_amount=min_avg_amount,
            min_avg_turn=min_avg_turn,
            exclude_bottom_liquidity_pct=exclude_bottom_liquidity_pct,
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

        result = engine.run(start_date, resolved_end_date, research_store=shared_research_store)
        report = result.analyze()
        cache_summary = frame_cache.summary() if frame_cache is not None else {}
        result.orders_frame().to_csv(run_output_dir / "orders.csv", index=False)
        result.trades_frame().to_csv(run_output_dir / "trades.csv", index=False)
        result.equity_frame().to_csv(run_output_dir / "equity.csv", index=False)
        result.closed_positions_frame().to_csv(run_output_dir / "closed_positions.csv", index=False)
        report.export(
            run_output_dir / "analytics",
            metadata={
                "strategy_name": "smallcap_liquidity_backtest",
                "script_name": Path(__file__).name,
                "run_name": run_name,
                "date_range": f"{start_date:%Y-%m-%d} -> {resolved_end_date:%Y-%m-%d}",
                "cache_summary": cache_summary,
                "parameters": {
                    **vars(args),
                    "candidate_pool_size": candidate_pool_size,
                    "min_avg_amount": min_avg_amount,
                    "min_avg_turn": min_avg_turn,
                    "liquidity_window": liquidity_window,
                    "exclude_bottom_liquidity_pct": exclude_bottom_liquidity_pct,
                    "resolved_end_date": resolved_end_date.strftime("%Y-%m-%d"),
                },
            },
        )

        summary = dict(report.summary)
        summary.update(
            {
                "run_name": run_name,
                "candidate_pool_size": candidate_pool_size,
                "min_avg_amount": min_avg_amount,
                "min_avg_turn": min_avg_turn,
                "liquidity_window": liquidity_window,
                "exclude_bottom_liquidity_pct": exclude_bottom_liquidity_pct,
            }
        )
        summary_rows.append(summary)
        log_event(
            "info",
            "smallcap_liquidity_batch_run_finish",
            run_name=run_name,
            final_equity=summary.get("final_equity"),
            total_return=summary.get("total_return"),
            max_drawdown=summary.get("max_drawdown"),
            trade_count=summary.get("trade_count"),
            cache_enabled=frame_cache is not None,
            cache_summary=cache_summary,
        )

    final_cache_summary = frame_cache.summary() if frame_cache is not None else {}
    summary_frame = pd.DataFrame(summary_rows)
    if not summary_frame.empty:
        if "total_return" in summary_frame.columns:
            summary_frame = summary_frame.sort_values(["total_return"], ascending=[False]).reset_index(drop=True)
        summary_frame.to_csv(batch_output_dir / "summary.csv", index=False)

    (batch_output_dir / "batch_parameters.json").write_text(
        json.dumps(
            {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "resolved_end_date": resolved_end_date.strftime("%Y-%m-%d"),
                "candidate_pool_sizes": candidate_pool_sizes,
                "min_avg_amounts": min_avg_amounts,
                "min_avg_turns": min_avg_turns,
                "liquidity_windows": liquidity_windows,
                "exclude_bottom_liquidity_pcts": exclude_bottom_liquidity_pcts,
                "base_parameters": vars(args),
                "cache_enabled": frame_cache is not None,
                "cache_dir": str(DEFAULT_CACHE_DIR),
                "cache_version": args.cache_version,
                "cache_summary": final_cache_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
