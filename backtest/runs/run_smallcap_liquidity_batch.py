from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.data import CachedDuckDBDataPortal, DuckDBDataPortal, FrameCache, ResearchDailyHistoryStore
from backtest.db import DuckDBConfig
from backtest.execution import EngineConfig, SignalDrivenBacktestEngine
from backtest.execution.smallcap_rotation_executor import SmallCapRotationDailyOpenExecutor
from backtest.portfolio import EqualSlotSizer
from backtest.risk import AbsoluteLowPriceExitPolicy, CloseBelowMaExitPolicy, CompositeExitPolicy, FixedStopLossExitPolicy
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


def _parse_optional_float(raw_value) -> float | None:
    text = "" if raw_value is None else str(raw_value).strip().lower()
    if text in {"", "none", "null", "na"}:
        return None
    return float(text)


def _parse_optional_int(raw_value) -> int | None:
    text = "" if raw_value is None else str(raw_value).strip().lower()
    if text in {"", "none", "null", "na"}:
        return None
    return int(text)


def _parse_optional_float_list(value: str) -> list[float | None]:
    return [_parse_optional_float(item) for item in str(value).split(",") if item.strip()]


def _dedupe_preserve_order(values: list):
    result = []
    seen = set()
    for value in values:
        key = None if value is None else float(value) if isinstance(value, float) else value
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _normalize_turn_threshold(min_avg_turn: float | None) -> float | None:
    if min_avg_turn is None:
        return None
    if 0 < min_avg_turn < 1:
        return min_avg_turn * 100.0
    return min_avg_turn


def _normalize_exclude_bottom_liquidity_pct(value: float | None) -> float | None:
    if value is None or value == 0:
        return None
    return value


def _validate_positive_int_list(name: str, values: list[int]) -> None:
    if not values:
        raise ValueError(f"{name} cannot be empty.")
    invalid = [value for value in values if value < 1]
    if invalid:
        raise ValueError(f"{name} must contain positive integers: {invalid}")


def _validate_optional_float_list(
    name: str,
    values: list[float | None],
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    allow_zero: bool = True,
) -> None:
    if not values:
        raise ValueError(f"{name} cannot be empty.")
    invalid = []
    for value in values:
        if value is None:
            continue
        if not math.isfinite(float(value)):
            invalid.append(value)
            continue
        if not allow_zero and value == 0:
            invalid.append(value)
            continue
        if min_value is not None and value < min_value:
            invalid.append(value)
            continue
        if max_value is not None and value > max_value:
            invalid.append(value)
    if invalid:
        raise ValueError(f"{name} contains invalid values: {invalid}")


def _slug_amount(value: float | None) -> str:
    if value is None:
        return "None"
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "_")


def _slug_float(value: float | None) -> str:
    if value is None:
        return "None"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def _slug_text(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _build_batch_output_dir(base_dir: Path, start_date: datetime, end_date: datetime) -> Path:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}_{run_timestamp}"
    return base_dir / run_name


def _write_summary(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    summary_frame = pd.DataFrame(summary_rows)
    if summary_frame.empty:
        return
    if "total_return" in summary_frame.columns:
        summary_frame = summary_frame.sort_values(["total_return"], ascending=[False]).reset_index(drop=True)
    summary_frame.to_csv(output_dir / "summary.csv", index=False)


def _parse_and_validate_grid(args: argparse.Namespace) -> dict[str, object]:
    if args.selection_sort == "ret_desc" and args.factor_sort_enabled:
        raise ValueError("--selection-sort ret_desc cannot be combined with --factor-sort-enabled")

    positive_int_checks = {
        "top_k": [args.top_k],
        "max_positions": [args.max_positions],
        "hold_days": [args.hold_days],
        "rebalance_every": [args.rebalance_every],
        "min_listing_trade_days": [args.min_listing_trade_days],
        "hhv_window": [args.hhv_window],
        "hhv_group_count": [args.hhv_group_count],
        "amount_expand_fast_window": [args.amount_expand_fast_window],
        "amount_expand_slow_window": [args.amount_expand_slow_window],
    }
    for name, values in positive_int_checks.items():
        _validate_positive_int_list(name, values)
    if args.amount_expand_slow_window < args.amount_expand_fast_window:
        raise ValueError("--amount-expand-slow-window must be >= --amount-expand-fast-window")
    if args.initial_cash <= 0:
        raise ValueError("--initial-cash must be positive")
    if args.slippage_bps < 0:
        raise ValueError("--slippage-bps must be non-negative")
    if args.fixed_stop_loss_pct is not None and args.fixed_stop_loss_pct <= 0:
        raise ValueError("--fixed-stop-loss-pct must be positive when provided")
    if args.ma_stop_window is not None and args.ma_stop_window < 1:
        raise ValueError("--ma-stop-window must be positive when provided")

    candidate_pool_sizes = _dedupe_preserve_order(_parse_int_list(args.candidate_pool_sizes))
    liquidity_windows = _dedupe_preserve_order(_parse_int_list(str(args.liquidity_window)))
    min_avg_amounts = _dedupe_preserve_order(_parse_optional_float_list(args.min_avg_amounts))
    min_avg_turns = _dedupe_preserve_order(
        [_normalize_turn_threshold(value) for value in _parse_optional_float_list(args.min_avg_turns)]
    )
    exclude_bottom_liquidity_pcts = _dedupe_preserve_order(
        [
            _normalize_exclude_bottom_liquidity_pct(value)
            for value in _parse_optional_float_list(str(args.exclude_bottom_liquidity_pct))
        ]
    )
    min_close_prices = _dedupe_preserve_order(_parse_optional_float_list(args.min_close_price))
    ret_windows = _dedupe_preserve_order(_parse_int_list(str(args.ret_window)))
    min_research_ret_10d_values = _dedupe_preserve_order(_parse_optional_float_list(args.min_research_ret_10d))

    _validate_positive_int_list("candidate_pool_sizes", candidate_pool_sizes)
    _validate_positive_int_list("liquidity_windows", liquidity_windows)
    _validate_positive_int_list("ret_windows", ret_windows)
    _validate_optional_float_list("min_avg_amounts", min_avg_amounts, min_value=0)
    _validate_optional_float_list("min_avg_turns", min_avg_turns, min_value=0)
    _validate_optional_float_list("exclude_bottom_liquidity_pcts", exclude_bottom_liquidity_pcts, min_value=0, max_value=1)
    _validate_optional_float_list("min_close_prices", min_close_prices, min_value=0, allow_zero=False)
    _validate_optional_float_list("min_research_ret_10d_values", min_research_ret_10d_values)

    st_lookback_trade_days = _parse_optional_int(args.st_lookback_trade_days)
    if st_lookback_trade_days is not None and st_lookback_trade_days < 0:
        raise ValueError("--st-lookback-trade-days must be non-negative")
    low_price_exit_threshold = _parse_optional_float(args.low_price_exit_threshold)
    if low_price_exit_threshold is not None and low_price_exit_threshold <= 0:
        raise ValueError("--low-price-exit-threshold must be positive when provided")
    hhv_keep_groups = _parse_int_tuple(args.hhv_keep_groups)

    args.st_lookback_trade_days = st_lookback_trade_days
    args.low_price_exit_threshold = low_price_exit_threshold
    args.hhv_keep_groups = hhv_keep_groups
    args.min_close_prices = min_close_prices
    args.ret_windows = ret_windows
    args.min_research_ret_10d_values = min_research_ret_10d_values

    param_grid = list(
        itertools.product(
            candidate_pool_sizes,
            min_avg_amounts,
            min_avg_turns,
            liquidity_windows,
            exclude_bottom_liquidity_pcts,
            min_close_prices,
            ret_windows,
            min_research_ret_10d_values,
        )
    )
    if not param_grid:
        raise ValueError("Parameter grid cannot be empty.")

    return {
        "candidate_pool_sizes": candidate_pool_sizes,
        "min_avg_amounts": min_avg_amounts,
        "min_avg_turns": min_avg_turns,
        "liquidity_windows": liquidity_windows,
        "exclude_bottom_liquidity_pcts": exclude_bottom_liquidity_pcts,
        "min_close_prices": min_close_prices,
        "ret_windows": ret_windows,
        "min_research_ret_10d_values": min_research_ret_10d_values,
        "hhv_keep_groups": hhv_keep_groups,
        "param_grid": param_grid,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a parameter grid for the small-cap liquidity-cleaned strategy.")
    parser.add_argument("--start-date", default="2020-01-01", help="Inclusive backtest start date, format YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-06-30", help="Inclusive backtest end date.")
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
    parser.add_argument(
        "--min-avg-amounts",
        default="30000000",
        help="Comma-separated average amount thresholds. Use none/null/na to disable this filter in a grid run.",
    )
    parser.add_argument(
        "--min-avg-turns",
        default="2",
        help="Comma-separated average turnover thresholds in percent points. Use none/null/na to disable this filter in a grid run. For example, 1,2,3 means 1%%, 2%%, 3%%. Legacy decimal inputs like 0.01 are auto-converted.",
    )
    parser.add_argument(
        "--exclude-bottom-liquidity-pct",
        default="0.15",
        help="Comma-separated cross-sectional exclusion ratios. Use none/null/na or 0 to disable this filter in a grid run. For example, 0.15 removes the bottom 15%%.",
    )
    parser.add_argument(
        "--min-close-price",
        "--min-close-prices",
        dest="min_close_price",
        default="1.5",
        help="Comma-separated minimum raw close price signal filters. Use none/null/na to disable this filter in a grid run.",
    )
    parser.add_argument("--st-lookback-trade-days", default="100", help="Filter stocks that were ST in the previous N trade days. Use none/null/na to disable.")
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
    parser.add_argument(
        "--ret-window",
        default="10",
        help="Comma-separated lookback windows used to compute ret_10d/research_ret_10d for selection-sort ret_desc.",
    )
    parser.add_argument(
        "--min-research-ret-10d",
        default="none",
        help="Comma-separated minimum research_ret_10d filters. Use none/null/na to disable. 0.05 means prior ret_window return <= -5%%.",
    )
    parser.add_argument(
        "--signal-price-mode",
        default="hfq",
        choices=["raw", "qfq", "hfq"],
        help="Price mode used to compute ret_10d/research_ret_10d. Raw close is still used for min-close-price.",
    )
    parser.add_argument(
        "--selection-sort",
        default="cap_asc",
        choices=["cap_asc", "ret_desc"],
        help="Final selection sort. cap_asc preserves legacy liquidity behavior; ret_desc matches amount_shock_reversal by sorting research_ret_10d descending.",
    )
    parser.add_argument("--slippage-bps", type=float, default=30.0, help="Buy/sell slippage in basis points.")
    parser.add_argument("--fixed-stop-loss-pct", type=float, default=None, help="Fixed intraday stop-loss percentage, for example 0.08.")
    parser.add_argument("--low-price-exit-threshold", default="1.3", help="Exit when intraday low touches this raw price. Use none/null/na to disable.")
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
    if args.low_price_exit_threshold is not None:
        exit_policies.append(AbsoluteLowPriceExitPolicy(min_low_price=args.low_price_exit_threshold))
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
    raw_cli_parameters = vars(args).copy()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    requested_end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()

    parsed_grid = _parse_and_validate_grid(args)
    candidate_pool_sizes = parsed_grid["candidate_pool_sizes"]
    min_avg_amounts = parsed_grid["min_avg_amounts"]
    min_avg_turns = parsed_grid["min_avg_turns"]
    liquidity_windows = parsed_grid["liquidity_windows"]
    exclude_bottom_liquidity_pcts = parsed_grid["exclude_bottom_liquidity_pcts"]
    min_close_prices = parsed_grid["min_close_prices"]
    ret_windows = parsed_grid["ret_windows"]
    min_research_ret_10d_values = parsed_grid["min_research_ret_10d_values"]
    hhv_keep_groups = parsed_grid["hhv_keep_groups"]
    param_grid = parsed_grid["param_grid"]

    db_client = DuckDBConfig()
    frame_cache = None if args.disable_cache else FrameCache(DEFAULT_CACHE_DIR, version=args.cache_version)
    data_portal = (
        DuckDBDataPortal(db_client)
        if frame_cache is None
        else CachedDuckDBDataPortal(db_client, frame_cache=frame_cache)
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
        min_close_prices=min_close_prices,
        ret_windows=ret_windows,
        min_research_ret_10d_values=min_research_ret_10d_values,
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
        signal_price_mode=args.signal_price_mode,
        selection_sort=args.selection_sort,
        st_lookback_trade_days=args.st_lookback_trade_days,
        low_price_exit_threshold=args.low_price_exit_threshold,
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
        min_close_price,
        ret_window,
        min_research_ret_10d,
    ) in enumerate(param_grid, start=1):
        run_name = (
            f"run_{run_index:03d}"
            f"_cp{candidate_pool_size}"
            f"_amt{_slug_amount(min_avg_amount)}"
            f"_turn{_slug_float(min_avg_turn)}"
            f"_lw{liquidity_window}"
            f"_ex{_slug_float(exclude_bottom_liquidity_pct)}"
            f"_px{_slug_float(min_close_price)}"
            f"_retw{ret_window}"
            f"_mret{_slug_float(min_research_ret_10d)}"
            f"_sort{_slug_text(args.selection_sort)}"
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
            min_close_price=min_close_price,
            st_lookback_trade_days=args.st_lookback_trade_days,
            factor_filter_enabled=args.factor_filter_enabled,
            factor_sort_enabled=args.factor_sort_enabled,
            amount_expand_descending=args.amount_expand_descending,
            hhv_window=args.hhv_window,
            hhv_group_count=args.hhv_group_count,
            hhv_keep_groups=hhv_keep_groups,
            amount_expand_fast_window=args.amount_expand_fast_window,
            amount_expand_slow_window=args.amount_expand_slow_window,
            ret_window=ret_window,
            min_research_ret_10d=min_research_ret_10d,
            signal_price_mode=args.signal_price_mode,
            selection_sort=args.selection_sort,
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
                    "min_close_price": min_close_price,
                    "ret_window": ret_window,
                    "min_research_ret_10d": min_research_ret_10d,
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
                "min_close_price": min_close_price,
                "min_research_ret_10d": min_research_ret_10d,
                "selection_sort": args.selection_sort,
                "ret_window": ret_window,
                "signal_price_mode": args.signal_price_mode,
            }
        )
        summary_rows.append(summary)
        _write_summary(summary_rows, batch_output_dir)
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
    _write_summary(summary_rows, batch_output_dir)

    (batch_output_dir / "batch_parameters.json").write_text(
        json.dumps(
            {
                "raw_cli_parameters": raw_cli_parameters,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "resolved_end_date": resolved_end_date.strftime("%Y-%m-%d"),
                "candidate_pool_sizes": candidate_pool_sizes,
                "min_avg_amounts": min_avg_amounts,
                "min_avg_turns": min_avg_turns,
                "liquidity_windows": liquidity_windows,
                "exclude_bottom_liquidity_pcts": exclude_bottom_liquidity_pcts,
                "min_close_prices": min_close_prices,
                "ret_windows": ret_windows,
                "min_research_ret_10d_values": min_research_ret_10d_values,
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
