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

from backtest.data import CachedDuckDBDataPortal, DuckDBDataPortal, FrameCache, ResearchDailyHistoryStore
from backtest.db import DuckDBConfig
from backtest.execution import EngineConfig, SignalDrivenBacktestEngine
from backtest.execution.smallcap_rotation_executor import SmallCapRotationDailyOpenExecutor
from backtest.portfolio import EqualSlotSizer
from backtest.risk import AbsoluteLowPriceExitPolicy
from backtest.strategies.smallcap_amount_shock_reversal import SmallCapAmountShockReversalStrategy
from backtest.utils.log import log_event

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "smallcap_amount_shock_reversal_batch"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "smallcap_amount_shock_reversal_batch"


def parse_int_tuple(raw_value: str) -> tuple[int, ...]:
    if raw_value is None or not str(raw_value).strip():
        return ()
    return tuple(sorted({int(part.strip()) for part in str(raw_value).split(",") if part.strip()}))


def parse_int_list(raw_value: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw_value).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_group_tuple_list(raw_value: str) -> list[tuple[int, ...]]:
    values = [parse_int_tuple(part) for part in str(raw_value).split(";") if part.strip()]
    if not values:
        raise ValueError("Expected at least one group definition.")
    return values


def parse_optional_float_list(raw_value: str) -> list[float | None]:
    values: list[float | None] = []
    for part in str(raw_value).split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item in {"none", "null", "na"}:
            values.append(None)
            continue
        value = float(item)
        if value < 0:
            raise ValueError("min_research_ret_10d values must be >= 0.")
        values.append(value)
    if not values:
        raise ValueError("Expected at least one min_research_ret_10d value.")
    return values


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


def groups_label(groups: tuple[int, ...]) -> str:
    return "none" if not groups else "_".join(str(group) for group in groups)


def optional_float_label(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{int(round(value * 100)):03d}"


def validate_keep_groups(name: str, groups: tuple[int, ...]) -> None:
    invalid = [group for group in groups if group < 1 or group > 5]
    if invalid:
        raise ValueError(f"{name} must only contain groups from 1 to 5, got: {invalid}")


def build_batch_output_dir(base_dir: Path, start_date: datetime, end_date: datetime) -> Path:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{start_date:%Y%m%d}_{end_date:%Y%m%d}_{run_timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the small-cap amount-shock reversal experiment batch.")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-04-24")
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--candidate-pool-sizes", default="150")
    parser.add_argument("--position-counts", default="10")
    parser.add_argument("--periods", default="20")
    parser.add_argument("--amount-keep-groups", default="5")
    parser.add_argument("--ret-keep-groups-list", default="3,4,5")
    parser.add_argument("--min-research-ret-10d-list", default="none")
    parser.add_argument("--min-listing-trade-days", type=int, default=120)
    parser.add_argument("--slippage-bps", type=float, default=30.0)
    parser.add_argument("--signal-price-mode", default="hfq", choices=["raw", "qfq", "hfq"])
    parser.add_argument("--selection-sort", default="ret_desc", choices=["ret_desc", "cap_asc"])
    parser.add_argument("--st-lookback-trade-days", default="100")
    parser.add_argument("--min-signal-close-price", default="1.5")
    parser.add_argument("--low-price-exit-threshold", default="1.3")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--cache-version", default="v1")
    return parser.parse_args()


def build_experiment_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    candidate_pool_sizes = parse_int_list(args.candidate_pool_sizes)
    position_counts = parse_int_list(args.position_counts)
    periods = parse_int_list(args.periods)
    amount_keep_groups = parse_int_tuple(args.amount_keep_groups)
    ret_keep_groups_list = parse_group_tuple_list(args.ret_keep_groups_list)
    min_research_ret_10d_list = parse_optional_float_list(args.min_research_ret_10d_list)

    validate_keep_groups("amount_keep_groups", amount_keep_groups)
    for groups in ret_keep_groups_list:
        validate_keep_groups("ret_keep_groups", groups)
    if any(value < 1 for value in candidate_pool_sizes):
        raise ValueError("candidate_pool_sizes must be >= 1.")
    if any(value < 1 for value in position_counts):
        raise ValueError("position_counts must be >= 1.")
    if any(value < 1 for value in periods):
        raise ValueError("periods must be >= 1.")

    experiments: list[dict[str, object]] = []
    for candidate_pool_size, position_count, period, ret_keep_groups, min_research_ret_10d in itertools.product(
        candidate_pool_sizes,
        position_counts,
        periods,
        ret_keep_groups_list,
        min_research_ret_10d_list,
    ):
        amount_label = groups_label(amount_keep_groups)
        ret_label = groups_label(ret_keep_groups)
        threshold_label = optional_float_label(min_research_ret_10d)
        experiments.append(
            {
                "run_name": (
                    f"cp{candidate_pool_size}_pos{position_count}_p{period}"
                    f"_amtg{amount_label}_retg{ret_label}_retmin{threshold_label}_{args.selection_sort}"
                ),
                "candidate_pool_size": int(candidate_pool_size),
                "position_count": int(position_count),
                "top_k": int(position_count),
                "max_positions": int(position_count),
                "hold_days": int(period),
                "rebalance_every": int(period),
                "period": int(period),
                "amount_keep_groups": amount_keep_groups,
                "ret_keep_groups": ret_keep_groups,
                "min_research_ret_10d": min_research_ret_10d,
                "min_research_ret_10d_label": threshold_label,
            }
        )

    run_names = [str(experiment["run_name"]) for experiment in experiments]
    if len(run_names) != len(set(run_names)):
        raise ValueError("Generated duplicate run names.")
    return experiments


def closed_position_stats(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty or "realized_return" not in frame.columns:
        return {
            "closed_position_count": 0,
            "avg_closed_return": None,
            "median_closed_return": None,
            "avg_holding_trade_days": None,
        }
    result = {
        "closed_position_count": int(len(frame)),
        "avg_closed_return": float(pd.to_numeric(frame["realized_return"], errors="coerce").mean()),
        "median_closed_return": float(pd.to_numeric(frame["realized_return"], errors="coerce").median()),
        "avg_holding_trade_days": None,
    }
    if "holding_trade_days" in frame.columns:
        result["avg_holding_trade_days"] = float(pd.to_numeric(frame["holding_trade_days"], errors="coerce").mean())
    return result


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

    batch_output_dir = build_batch_output_dir(Path(args.output_dir), start_date, resolved_end_date)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    st_lookback_trade_days = parse_optional_int(args.st_lookback_trade_days)
    min_signal_close_price = parse_optional_float(args.min_signal_close_price)
    low_price_exit_threshold = parse_optional_float(args.low_price_exit_threshold)
    shared_research_store = ResearchDailyHistoryStore(data_portal)
    execution_model = SmallCapRotationDailyOpenExecutor(slippage_bps=args.slippage_bps)
    exit_policy = (
        AbsoluteLowPriceExitPolicy(min_low_price=low_price_exit_threshold)
        if low_price_exit_threshold is not None
        else None
    )
    summary_rows: list[dict[str, object]] = []
    experiments = build_experiment_grid(args)
    log_event(
        "info",
        "smallcap_amount_shock_reversal_batch_start",
        start_date=start_date,
        resolved_end_date=resolved_end_date,
        experiment_count=len(experiments),
        candidate_pool_sizes=args.candidate_pool_sizes,
        position_counts=args.position_counts,
        periods=args.periods,
        min_research_ret_10d_list=args.min_research_ret_10d_list,
        st_lookback_trade_days=st_lookback_trade_days,
        min_signal_close_price=min_signal_close_price,
        low_price_exit_threshold=low_price_exit_threshold,
        slippage_bps=args.slippage_bps,
        signal_price_mode=args.signal_price_mode,
        cache_enabled=frame_cache is not None,
        cache_dir=str(DEFAULT_CACHE_DIR),
        cache_version=args.cache_version,
    )

    for index, experiment in enumerate(experiments, start=1):
        run_name = str(experiment["run_name"])
        amount_keep_groups = tuple(experiment["amount_keep_groups"])
        ret_keep_groups = tuple(experiment["ret_keep_groups"])
        run_output_dir = batch_output_dir / f"{index:02d}_{run_name}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        strategy = SmallCapAmountShockReversalStrategy(
            top_k=int(experiment["top_k"]),
            hold_days=int(experiment["hold_days"]),
            rebalance_every_n_trade_days=int(experiment["rebalance_every"]),
            min_listing_trade_days=args.min_listing_trade_days,
            candidate_pool_size=int(experiment["candidate_pool_size"]),
            amount_keep_groups=amount_keep_groups,
            ret_keep_groups=ret_keep_groups,
            min_research_ret_10d=experiment["min_research_ret_10d"],
            selection_sort=args.selection_sort,
            signal_price_mode=args.signal_price_mode,
            st_lookback_trade_days=st_lookback_trade_days,
            min_signal_close_price=min_signal_close_price,
        )
        config = EngineConfig(
            initial_cash=args.initial_cash,
            max_positions=int(experiment["max_positions"]),
            position_size_pct=1 / max(int(experiment["max_positions"]), 1),
            execute_on_next_trade_date=True,
            progress_logging=True,
        )
        engine = SignalDrivenBacktestEngine(
            db_client,
            strategy,
            execution_model=execution_model,
            config=config,
            position_sizer=EqualSlotSizer(),
            data_portal=data_portal,
            exit_policy=exit_policy,
        )

        result = engine.run(start_date, resolved_end_date, research_store=shared_research_store)
        report = result.analyze()
        closed_positions = result.closed_positions_frame()
        result.orders_frame().to_csv(run_output_dir / "orders.csv", index=False)
        result.trades_frame().to_csv(run_output_dir / "trades.csv", index=False)
        result.equity_frame().to_csv(run_output_dir / "equity.csv", index=False)
        closed_positions.to_csv(run_output_dir / "closed_positions.csv", index=False)
        report.export(
            run_output_dir / "analytics",
            metadata={
                "strategy_name": "smallcap_amount_shock_reversal",
                "script_name": Path(__file__).name,
                "run_name": run_name,
                "date_range": f"{start_date:%Y-%m-%d} -> {resolved_end_date:%Y-%m-%d}",
                "parameters": {
                    **vars(args),
                    "candidate_pool_size": int(experiment["candidate_pool_size"]),
                    "position_count": int(experiment["position_count"]),
                    "top_k": int(experiment["top_k"]),
                    "max_positions": int(experiment["max_positions"]),
                    "hold_days": int(experiment["hold_days"]),
                    "rebalance_every": int(experiment["rebalance_every"]),
                    "period": int(experiment["period"]),
                    "amount_keep_groups": list(amount_keep_groups),
                    "ret_keep_groups": list(ret_keep_groups),
                    "min_research_ret_10d": experiment["min_research_ret_10d"],
                    "min_research_ret_10d_label": str(experiment["min_research_ret_10d_label"]),
                    "selection_sort": args.selection_sort,
                    "st_lookback_trade_days": st_lookback_trade_days,
                    "min_signal_close_price": min_signal_close_price,
                    "low_price_exit_threshold": low_price_exit_threshold,
                    "resolved_end_date": resolved_end_date.strftime("%Y-%m-%d"),
                },
            },
        )

        row = dict(report.summary)
        row.update(
            {
                "run_name": run_name,
                "hold_days": int(experiment["hold_days"]),
                "rebalance_every": int(experiment["rebalance_every"]),
                "period": int(experiment["period"]),
                "candidate_pool_size": int(experiment["candidate_pool_size"]),
                "position_count": int(experiment["position_count"]),
                "top_k": int(experiment["top_k"]),
                "max_positions": int(experiment["max_positions"]),
                "amount_keep_groups": groups_label(amount_keep_groups),
                "ret_keep_groups": groups_label(ret_keep_groups),
                "min_research_ret_10d": experiment["min_research_ret_10d"],
                "min_research_ret_10d_label": str(experiment["min_research_ret_10d_label"]),
                "selection_sort": args.selection_sort,
                "st_lookback_trade_days": st_lookback_trade_days,
                "min_signal_close_price": min_signal_close_price,
                "low_price_exit_threshold": low_price_exit_threshold,
                **closed_position_stats(closed_positions),
            }
        )
        summary_rows.append(row)
        log_event(
            "info",
            "smallcap_amount_shock_reversal_batch_run_finish",
            run_name=run_name,
            total_return=row.get("total_return"),
            max_drawdown=row.get("max_drawdown"),
            trade_count=row.get("trade_count"),
        )

    summary_frame = pd.DataFrame(summary_rows)
    if not summary_frame.empty:
        sort_col = "total_return" if "total_return" in summary_frame.columns else "run_name"
        summary_frame = summary_frame.sort_values(sort_col, ascending=False).reset_index(drop=True)
        summary_frame.to_csv(batch_output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (batch_output_dir / "batch_parameters.json").write_text(
        json.dumps(
            {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "resolved_end_date": resolved_end_date.strftime("%Y-%m-%d"),
                "experiments": [
                    {
                        **experiment,
                        "amount_keep_groups": list(experiment["amount_keep_groups"]),
                        "ret_keep_groups": list(experiment["ret_keep_groups"]),
                    }
                    for experiment in experiments
                ],
                "base_parameters": vars(args),
                "parsed_risk_parameters": {
                    "st_lookback_trade_days": st_lookback_trade_days,
                    "min_signal_close_price": min_signal_close_price,
                    "low_price_exit_threshold": low_price_exit_threshold,
                },
                "cache_enabled": frame_cache is not None,
                "cache_dir": str(DEFAULT_CACHE_DIR),
                "cache_version": args.cache_version,
                "cache_summary": frame_cache.summary() if frame_cache is not None else {},
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
