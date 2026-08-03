from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.data import CachedDuckDBDataPortal, DuckDBDataPortal, FrameCache, ResearchDailyHistoryStore
from backtest.db import DuckDBConfig
from backtest.execution import EngineConfig, SignalDrivenBacktestEngine
from backtest.execution.smallcap_rotation_executor import SmallCapRotationDailyOpenExecutor
from backtest.portfolio import EqualSlotSizer
from backtest.risk import AbsoluteLowPriceExitPolicy, CompositeExitPolicy, FixedStopLossExitPolicy
from backtest.strategies.smallcap_amount_shock_event import EVENT_SELECTION_SORTS, SmallCapAmountShockEventStrategy
from backtest.utils.log import log_event

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "smallcap_amount_shock_event_batch"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "smallcap_amount_shock_event_batch"


def parse_optional_float(raw_value) -> float | None:
    text = "" if raw_value is None else str(raw_value).strip().lower()
    if text in {"", "none", "null", "na"}:
        return None
    return float(text)


def parse_optional_int(raw_value) -> int | None:
    text = "" if raw_value is None else str(raw_value).strip().lower()
    if text in {"", "none", "null", "na"}:
        return None
    return int(text)


def groups_label(groups: Sequence[int]) -> str:
    return "none" if not groups else "_".join(str(group) for group in groups)


def float_label(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def build_batch_output_dir(base_dir: Path, start_date: datetime, end_date: datetime, run_name: str) -> Path:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(run_name or "batch"))
    return base_dir / f"{safe_run_name}_{start_date:%Y%m%d}_{end_date:%Y%m%d}_{run_timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small-cap amount-shock event experiment batches from a JSON config.")
    parser.add_argument("--config-file", required=True, help="JSON config file describing the experiment matrix.")
    parser.add_argument("--start-date", default=None, help="Optional override for config date_range.start_date.")
    parser.add_argument("--end-date", default=None, help="Optional override for config date_range.end_date.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-version", default=None, help="Optional override for config cache_version.")
    parser.add_argument("--fixed-event-anchor-run-name", default=None)
    parser.add_argument("--disable-cache", action="store_true")
    return parser.parse_args()


def load_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config root must be a JSON object.")
    return data


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config.{key} must be an object.")
    return value


def require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list.")
    return value


def int_list(value: Any, path: str) -> list[int]:
    values = [int(item) for item in require_list(value, path)]
    if any(item < 1 for item in values):
        raise ValueError(f"{path} values must be >= 1.")
    return values


def float_list(value: Any, path: str) -> list[float]:
    values = [float(item) for item in require_list(value, path)]
    if any(item <= 0 for item in values):
        raise ValueError(f"{path} values must be positive.")
    return values


def string_list(value: Any, path: str) -> list[str]:
    values = [str(item).strip().lower() for item in require_list(value, path)]
    if any(not item for item in values):
        raise ValueError(f"{path} contains an empty value.")
    return values


def build_runtime_args(cli_args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    date_range = require_mapping(config, "date_range")
    execution = require_mapping(config, "execution")
    strategy_defaults = require_mapping(config, "strategy_defaults")

    start_date = cli_args.start_date or date_range.get("start_date")
    end_date = cli_args.end_date or date_range.get("end_date")
    if not start_date or not end_date:
        raise ValueError("date_range.start_date and date_range.end_date are required.")

    experiment_mode = str(config.get("experiment_mode", "recomputed_events")).strip()
    if experiment_mode not in {"recomputed_events", "fixed_dates", "both"}:
        raise ValueError("experiment_mode must be one of: recomputed_events, fixed_dates, both.")

    return argparse.Namespace(
        config_file=str(cli_args.config_file),
        run_name=str(config.get("run_name", "smallcap_amount_shock_event_batch")),
        experiment_mode=experiment_mode,
        start_date=str(start_date),
        end_date=str(end_date),
        initial_cash=float(execution.get("initial_cash", 1_000_000)),
        max_positions=int(execution.get("max_positions", 10)),
        position_size_pct=float(execution.get("position_size_pct", 0.2)),
        hold_days=int(execution.get("hold_days", 20)),
        slippage_bps=float(execution.get("slippage_bps", 30.0)),
        signal_price_mode=str(execution.get("signal_price_mode", "qfq")),
        low_price_exit_threshold=execution.get("low_price_exit_threshold", 1.3),
        fixed_stop_loss_pct=execution.get("fixed_stop_loss_pct"),
        min_listing_trade_days=int(strategy_defaults.get("min_listing_trade_days", 120)),
        min_research_ret_10d=parse_optional_float(strategy_defaults.get("min_research_ret_10d", 0.12)),
        group_count=int(strategy_defaults.get("group_count", 5)),
        amount_fast_window=int(strategy_defaults.get("amount_fast_window", 5)),
        amount_slow_window=int(strategy_defaults.get("amount_slow_window", 20)),
        ret_window=int(strategy_defaults.get("ret_window", 10)),
        st_lookback_trade_days=str(strategy_defaults.get("st_lookback_trade_days", "100")),
        min_signal_close_price=strategy_defaults.get("min_signal_close_price", 1.5),
        output_dir=str(cli_args.output_dir),
        disable_cache=bool(cli_args.disable_cache),
        cache_version=str(cli_args.cache_version or config.get("cache_version", "amount_shock_event_batch_v1")),
        fixed_event_anchor_run_name=(
            str(cli_args.fixed_event_anchor_run_name or config.get("fixed_event_anchor_run_name") or "").strip() or None
        ),
    )


def validate_keep_groups(groups: Sequence[int], *, group_count: int, path: str) -> tuple[int, ...]:
    normalized = tuple(sorted(set(int(group) for group in groups)))
    invalid = [group for group in normalized if group < 1 or group > group_count]
    if invalid:
        raise ValueError(f"{path} must be within 1..{group_count}, got: {invalid}")
    return normalized


def build_experiments(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, object]]:
    grid = require_mapping(config, "parameter_grid")
    ret_top_pcts = float_list(grid.get("ret_top_pcts"), "parameter_grid.ret_top_pcts")
    if any(value > 1.0 for value in ret_top_pcts):
        raise ValueError("parameter_grid.ret_top_pcts values must be within (0, 1].")
    top_ks = int_list(grid.get("top_ks"), "parameter_grid.top_ks")
    experiment_configs = require_list(config.get("experiments"), "experiments")

    experiments: list[dict[str, object]] = []
    for experiment_config in experiment_configs:
        if not isinstance(experiment_config, dict):
            raise ValueError("each experiments item must be an object.")
        name = str(experiment_config.get("name", "")).strip()
        if not name:
            raise ValueError("experiment.name is required.")
        candidate_pool_sizes = int_list(experiment_config.get("candidate_pool_sizes"), f"experiments.{name}.candidate_pool_sizes")
        amount_keep_groups = validate_keep_groups(
            int_list(experiment_config.get("amount_keep_groups"), f"experiments.{name}.amount_keep_groups"),
            group_count=args.group_count,
            path=f"experiments.{name}.amount_keep_groups",
        )
        ret_keep_groups = validate_keep_groups(
            int_list(experiment_config.get("ret_keep_groups"), f"experiments.{name}.ret_keep_groups"),
            group_count=args.group_count,
            path=f"experiments.{name}.ret_keep_groups",
        )
        selection_sorts = string_list(experiment_config.get("selection_sorts"), f"experiments.{name}.selection_sorts")
        invalid_sorts = [sort for sort in selection_sorts if sort not in EVENT_SELECTION_SORTS]
        if invalid_sorts:
            allowed = ", ".join(sorted(EVENT_SELECTION_SORTS))
            raise ValueError(f"Invalid selection_sorts for {name}: {invalid_sorts}; allowed: {allowed}")

        description = str(experiment_config.get("description", "")).strip()
        for candidate_pool_size, ret_top_pct, top_k, selection_sort in itertools.product(
            candidate_pool_sizes,
            ret_top_pcts,
            top_ks,
            selection_sorts,
        ):
            experiments.append(
                {
                    "experiment_name": name,
                    "experiment_description": description,
                    "candidate_pool_size": int(candidate_pool_size),
                    "amount_keep_groups": amount_keep_groups,
                    "ret_keep_groups": ret_keep_groups,
                    "ret_top_pct": float(ret_top_pct),
                    "top_k": int(top_k),
                    "selection_sort": str(selection_sort),
                    "run_name": (
                        f"{name}__cp{candidate_pool_size}"
                        f"__retpct{float_label(ret_top_pct)}"
                        f"__topk{top_k}__{selection_sort}"
                    ),
                    "overlap_group": f"{name}|cp{candidate_pool_size}|retpct{float_label(ret_top_pct)}|topk{top_k}",
                }
            )

    run_names = [str(experiment["run_name"]) for experiment in experiments]
    if len(run_names) != len(set(run_names)):
        raise ValueError("expanded experiments contain duplicate run_name values.")
    if not experiments:
        raise ValueError("expanded experiment matrix is empty.")
    return experiments


def build_exit_policy(*, fixed_stop_loss_pct: float | None, low_price_exit_threshold: float | None):
    policies = []
    if fixed_stop_loss_pct is not None:
        policies.append(FixedStopLossExitPolicy(stop_loss_pct=fixed_stop_loss_pct))
    if low_price_exit_threshold is not None:
        policies.append(AbsoluteLowPriceExitPolicy(min_low_price=low_price_exit_threshold))
    if not policies:
        return None
    if len(policies) == 1:
        return policies[0]
    return CompositeExitPolicy(policies)


def base_strategy(
    args: argparse.Namespace,
    *,
    candidate_pool_size: int,
    amount_keep_groups: Sequence[int],
    ret_keep_groups: Sequence[int],
    top_k: int,
    ret_top_pct: float,
    selection_sort: str,
    st_lookback_trade_days: int | None,
    min_signal_close_price: float | None,
    fixed_event_dates: Sequence[datetime | str] | None = None,
    precomputed_candidate_indicator_frame: pd.DataFrame | None = None,
) -> SmallCapAmountShockEventStrategy:
    return SmallCapAmountShockEventStrategy(
        top_k=top_k,
        hold_days=args.hold_days,
        min_listing_trade_days=args.min_listing_trade_days,
        candidate_pool_size=candidate_pool_size,
        amount_fast_window=args.amount_fast_window,
        amount_slow_window=args.amount_slow_window,
        ret_window=args.ret_window,
        group_count=args.group_count,
        amount_keep_groups=amount_keep_groups,
        ret_keep_groups=ret_keep_groups,
        min_research_ret_10d=args.min_research_ret_10d,
        ret_top_pct=ret_top_pct,
        selection_sort=selection_sort,
        signal_price_mode=args.signal_price_mode,
        st_lookback_trade_days=st_lookback_trade_days,
        min_signal_close_price=min_signal_close_price,
        fixed_event_dates=fixed_event_dates,
        precomputed_candidate_indicator_frame=precomputed_candidate_indicator_frame,
    )


def build_candidate_indicator_base(
    args: argparse.Namespace,
    *,
    data_portal,
    trade_dates: Sequence[datetime],
    research_store: ResearchDailyHistoryStore,
    max_candidate_pool_size: int,
    st_lookback_trade_days: int | None,
    min_signal_close_price: float | None,
) -> pd.DataFrame:
    """只预计算与实验参数无关的候选和指标底表，事件筛选仍由各实验独立完成。"""

    frame_cache = getattr(data_portal, "frame_cache", None)
    signal_start = pd.Timestamp(trade_dates[0]).to_pydatetime()
    signal_end = pd.Timestamp(trade_dates[-1]).to_pydatetime()
    payload = {
        "feature_formula_version": "smallcap_amount_shock_event_candidate_indicator_base_v2",
        "start_date": signal_start,
        "end_date": signal_end,
        "max_candidate_pool_size": max_candidate_pool_size,
        "min_listing_trade_days": args.min_listing_trade_days,
        "amount_fast_window": args.amount_fast_window,
        "amount_slow_window": args.amount_slow_window,
        "ret_window": args.ret_window,
        "group_count": args.group_count,
        "signal_price_mode": args.signal_price_mode,
        "st_lookback_trade_days": st_lookback_trade_days,
        "min_signal_close_price": min_signal_close_price,
        "duckdb_revision": getattr(getattr(data_portal, "db_client", None), "cache_revision", "unknown"),
    }

    def builder() -> pd.DataFrame:
        strategy = base_strategy(
            args,
            candidate_pool_size=max_candidate_pool_size,
            amount_keep_groups=(1,),
            ret_keep_groups=(1,),
            top_k=1,
            ret_top_pct=1.0,
            selection_sort="amount_then_ret",
            st_lookback_trade_days=st_lookback_trade_days,
            min_signal_close_price=min_signal_close_price,
        )
        candidate_pool_frame, full_trade_calendar = strategy._prepare_cached_candidate_pool_frame(data_portal, trade_dates)
        if candidate_pool_frame.empty or full_trade_calendar.empty:
            return pd.DataFrame()
        warmup_start = strategy._warmup_trade_dates(full_trade_calendar, signal_start)
        indicator_frame = strategy._prepare_indicator_frame(
            data_portal,
            codes=sorted(candidate_pool_frame["code"].unique().tolist()),
            warmup_start=warmup_start,
            signal_end=signal_end,
            research_store=research_store,
        )
        if indicator_frame.empty:
            return pd.DataFrame()
        key_columns = ["code", "trade_date"]
        if candidate_pool_frame.duplicated(key_columns).any():
            raise ValueError("candidate_pool_frame contains duplicate (code, trade_date) rows.")
        if indicator_frame.duplicated(key_columns).any():
            raise ValueError("indicator_frame contains duplicate (code, trade_date) rows.")
        result = candidate_pool_frame.merge(
            indicator_frame,
            on=key_columns,
            how="inner",
            validate="one_to_one",
        )
        if result.empty:
            return result
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        return result.sort_values(["trade_date", "liqaMV", "code"], ascending=[True, True, True], kind="mergesort").reset_index(drop=True)

    if frame_cache is None:
        frame = builder()
    else:
        frame = frame_cache.load_or_build_frame("smallcap_amount_shock_event_candidate_indicator_base", payload, builder)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.sort_values(["trade_date", "liqaMV", "code"], ascending=[True, True, True], kind="mergesort").reset_index(drop=True)


def extract_event_dates(strategy: SmallCapAmountShockEventStrategy) -> list[pd.Timestamp]:
    frame = strategy.event_signal_frame
    if frame.empty or "trade_date" not in frame.columns:
        return []
    return sorted(pd.to_datetime(frame["trade_date"]).dt.normalize().drop_duplicates().tolist())


def closed_position_stats(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty or "realized_return" not in frame.columns:
        return {
            "closed_position_count": int(len(frame)),
            "avg_closed_return": None,
            "median_closed_return": None,
            "p25_closed_return": None,
            "win_rate_closed_return": None,
            "avg_holding_trade_days": None,
        }
    returns = pd.to_numeric(frame["realized_return"], errors="coerce")
    result = {
        "closed_position_count": int(len(frame)),
        "avg_closed_return": float(returns.mean()),
        "median_closed_return": float(returns.median()),
        "p25_closed_return": float(returns.quantile(0.25)),
        "win_rate_closed_return": float((returns > 0).mean()),
        "avg_holding_trade_days": None,
    }
    if "holding_trade_days" in frame.columns:
        result["avg_holding_trade_days"] = float(pd.to_numeric(frame["holding_trade_days"], errors="coerce").mean())
    return result


def signal_stats(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "trade_date" not in frame.columns:
        return {"event_signal_rows": 0, "event_signal_days": 0}
    return {
        "event_signal_rows": int(len(frame)),
        "event_signal_days": int(pd.to_datetime(frame["trade_date"]).dt.normalize().nunique()),
    }


def selected_topn(frame: pd.DataFrame, *, top_n: int) -> dict[str, tuple[str, ...]]:
    if frame.empty:
        return {}
    working = frame.copy()
    working["trade_date"] = pd.to_datetime(working["trade_date"]).dt.strftime("%Y-%m-%d")
    working = working.sort_values(["trade_date", "event_signal_rank", "code"], ascending=[True, True, True], kind="mergesort")
    return (
        working.groupby("trade_date", sort=True)
        .head(top_n)
        .groupby("trade_date")["code"]
        .apply(lambda values: tuple(str(value) for value in values))
        .to_dict()
    )


def build_overlap_rows(
    *,
    mode: str,
    overlap_group: str,
    experiment: dict[str, object],
    selections: dict[str, dict[str, tuple[str, ...]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sort_a, sort_b in itertools.combinations(sorted(selections), 2):
        left = selections[sort_a]
        right = selections[sort_b]
        dates = sorted(set(left) & set(right))
        exact = 0
        jaccards: list[float] = []
        for date in dates:
            if left[date] == right[date]:
                exact += 1
            left_set = set(left[date])
            right_set = set(right[date])
            union = left_set | right_set
            jaccards.append(len(left_set & right_set) / len(union) if union else 0.0)
        rows.append(
            {
                "experiment_mode": mode,
                "overlap_group": overlap_group,
                "experiment_name": str(experiment["experiment_name"]),
                "candidate_pool_size": int(experiment["candidate_pool_size"]),
                "ret_top_pct": float(experiment["ret_top_pct"]),
                "top_k": int(experiment["top_k"]),
                "top_n": int(experiment["top_k"]),
                "selection_sort_a": sort_a,
                "selection_sort_b": sort_b,
                "common_days": int(len(dates)),
                "exact_topn_days": int(exact),
                "avg_jaccard": float(sum(jaccards) / len(jaccards)) if jaccards else None,
            }
        )
    return rows


def run_single_experiment(
    args: argparse.Namespace,
    *,
    db_client: DuckDBConfig,
    data_portal,
    research_store: ResearchDailyHistoryStore,
    execution_model: SmallCapRotationDailyOpenExecutor,
    exit_policy,
    start_date: datetime,
    resolved_end_date: datetime,
    mode: str,
    run_output_dir: Path,
    experiment: dict[str, object],
    st_lookback_trade_days: int | None,
    min_signal_close_price: float | None,
    fixed_event_dates: Sequence[pd.Timestamp] | None,
    precomputed_candidate_indicator_frame: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, tuple[str, ...]]]:
    amount_keep_groups = tuple(experiment["amount_keep_groups"])
    ret_keep_groups = tuple(experiment["ret_keep_groups"])
    top_k = int(experiment["top_k"])
    ret_top_pct = float(experiment["ret_top_pct"])
    selection_sort = str(experiment["selection_sort"])
    strategy = base_strategy(
        args,
        candidate_pool_size=int(experiment["candidate_pool_size"]),
        amount_keep_groups=amount_keep_groups,
        ret_keep_groups=ret_keep_groups,
        top_k=top_k,
        ret_top_pct=ret_top_pct,
        selection_sort=selection_sort,
        st_lookback_trade_days=st_lookback_trade_days,
        min_signal_close_price=min_signal_close_price,
        fixed_event_dates=fixed_event_dates,
        precomputed_candidate_indicator_frame=precomputed_candidate_indicator_frame,
    )
    config = EngineConfig(
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
        position_size_pct=args.position_size_pct,
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

    result = engine.run(start_date, resolved_end_date, research_store=research_store)
    report = result.analyze()
    closed_positions = result.closed_positions_frame()

    run_output_dir.mkdir(parents=True, exist_ok=True)
    result.orders_frame().to_csv(run_output_dir / "orders.csv", index=False)
    result.trades_frame().to_csv(run_output_dir / "trades.csv", index=False)
    result.equity_frame().to_csv(run_output_dir / "equity.csv", index=False)
    closed_positions.to_csv(run_output_dir / "closed_positions.csv", index=False)
    strategy.event_signal_frame.to_csv(run_output_dir / "event_signal_table.csv", index=False, encoding="utf-8-sig")
    strategy.event_daily_window_features.to_csv(run_output_dir / "event_daily_window_features.csv", index=False, encoding="utf-8-sig")

    run_name = str(experiment["run_name"])
    report.export(
        run_output_dir / "analytics",
        metadata={
            "strategy_name": "smallcap_amount_shock_event",
            "script_name": Path(__file__).name,
            "run_name": run_name,
            "experiment_mode": mode,
            "date_range": f"{start_date:%Y-%m-%d} -> {resolved_end_date:%Y-%m-%d}",
            "parameters": {
                **vars(args),
                "candidate_pool_size": int(experiment["candidate_pool_size"]),
                "amount_keep_groups": list(amount_keep_groups),
                "ret_keep_groups": list(ret_keep_groups),
                "selection_sort": selection_sort,
                "ret_top_pct": ret_top_pct,
                "top_k": top_k,
                "st_lookback_trade_days": st_lookback_trade_days,
                "min_signal_close_price": min_signal_close_price,
                "fixed_event_date_count": 0 if fixed_event_dates is None else len(fixed_event_dates),
                "resolved_end_date": resolved_end_date.strftime("%Y-%m-%d"),
            },
        },
    )

    row = dict(report.summary)
    row.update(
        {
            "experiment_mode": mode,
            "run_name": run_name,
            "experiment_name": str(experiment["experiment_name"]),
            "experiment_description": str(experiment.get("experiment_description", "")),
            "selection_sort": selection_sort,
            "candidate_pool_size": int(experiment["candidate_pool_size"]),
            "amount_keep_groups": groups_label(amount_keep_groups),
            "ret_keep_groups": groups_label(ret_keep_groups),
            "ret_top_pct": ret_top_pct,
            "top_k": top_k,
            "max_positions": args.max_positions,
            "position_size_pct": args.position_size_pct,
            "hold_days": args.hold_days,
            "fixed_event_date_count": 0 if fixed_event_dates is None else len(fixed_event_dates),
            **signal_stats(strategy.event_signal_frame),
            **closed_position_stats(closed_positions),
        }
    )
    top_map = selected_topn(strategy.event_signal_frame, top_n=top_k)
    return row, top_map


def main() -> None:
    cli_args = parse_args()
    config_path = Path(cli_args.config_file)
    raw_config = load_json_config(config_path)
    args = build_runtime_args(cli_args, raw_config)
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    requested_end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    st_lookback_trade_days = parse_optional_int(args.st_lookback_trade_days)
    min_signal_close_price = parse_optional_float(args.min_signal_close_price)
    fixed_stop_loss_pct = parse_optional_float(args.fixed_stop_loss_pct)
    low_price_exit_threshold = parse_optional_float(args.low_price_exit_threshold)

    experiments = build_experiments(raw_config, args)
    max_candidate_pool_size = max(int(experiment["candidate_pool_size"]) for experiment in experiments)

    db_client = DuckDBConfig(read_only=True)
    frame_cache = None if args.disable_cache else FrameCache(DEFAULT_CACHE_DIR, version=args.cache_version)
    data_portal = DuckDBDataPortal(db_client) if frame_cache is None else CachedDuckDBDataPortal(db_client, frame_cache=frame_cache)
    trade_dates = data_portal.get_trade_calendar(start_date, requested_end_date)
    if not trade_dates:
        raise ValueError("No trade dates available in the requested date range.")
    resolved_end_date = trade_dates[-1]

    batch_output_dir = build_batch_output_dir(Path(args.output_dir), start_date, resolved_end_date, args.run_name)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    research_store = ResearchDailyHistoryStore(data_portal)
    execution_model = SmallCapRotationDailyOpenExecutor(slippage_bps=args.slippage_bps)
    exit_policy = build_exit_policy(
        fixed_stop_loss_pct=fixed_stop_loss_pct,
        low_price_exit_threshold=low_price_exit_threshold,
    )

    log_event(
        "info",
        "smallcap_amount_shock_event_batch_start",
        config_file=str(config_path),
        experiment_mode=args.experiment_mode,
        start_date=start_date,
        resolved_end_date=resolved_end_date,
        experiment_count=len(experiments),
        max_candidate_pool_size=max_candidate_pool_size,
        cache_enabled=frame_cache is not None,
        cache_dir=str(DEFAULT_CACHE_DIR),
        cache_version=args.cache_version,
    )

    precomputed_candidate_indicator = build_candidate_indicator_base(
        args,
        data_portal=data_portal,
        trade_dates=trade_dates,
        research_store=research_store,
        max_candidate_pool_size=max_candidate_pool_size,
        st_lookback_trade_days=st_lookback_trade_days,
        min_signal_close_price=min_signal_close_price,
    )
    if precomputed_candidate_indicator.empty:
        raise ValueError("Precomputed candidate indicator frame is empty.")

    modes = ["recomputed_events", "fixed_dates"] if args.experiment_mode == "both" else [args.experiment_mode]
    fixed_event_dates: list[pd.Timestamp] | None = None
    fixed_event_anchor_run_name: str | None = None
    if "fixed_dates" in modes:
        # 固定事件日必须来自明确命名的锚定实验，不能依赖实验列表顺序隐式选择。
        if not args.fixed_event_anchor_run_name:
            raise ValueError("fixed_dates/both mode requires fixed_event_anchor_run_name.")
        matching_anchors = [
            experiment
            for experiment in experiments
            if str(experiment["run_name"]) == args.fixed_event_anchor_run_name
        ]
        if len(matching_anchors) != 1:
            raise ValueError(
                "fixed_event_anchor_run_name must exactly match one expanded experiment run_name: "
                f"{args.fixed_event_anchor_run_name!r}"
            )
        anchor_experiment = matching_anchors[0]
        fixed_event_anchor_run_name = str(anchor_experiment["run_name"])
        anchor_strategy = base_strategy(
            args,
            candidate_pool_size=int(anchor_experiment["candidate_pool_size"]),
            amount_keep_groups=tuple(anchor_experiment["amount_keep_groups"]),
            ret_keep_groups=tuple(anchor_experiment["ret_keep_groups"]),
            top_k=int(anchor_experiment["top_k"]),
            ret_top_pct=float(anchor_experiment["ret_top_pct"]),
            selection_sort=str(anchor_experiment["selection_sort"]),
            st_lookback_trade_days=st_lookback_trade_days,
            min_signal_close_price=min_signal_close_price,
            precomputed_candidate_indicator_frame=precomputed_candidate_indicator,
        )
        anchor_strategy.prepare(data_portal, trade_dates, research_store=research_store)
        fixed_event_dates = extract_event_dates(anchor_strategy)
        fixed_dates_frame = pd.DataFrame({"trade_date": [date.strftime("%Y-%m-%d") for date in fixed_event_dates]})
        fixed_dates_frame.to_csv(batch_output_dir / "fixed_event_dates.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    # recomputed_events 会逐组重算事件日；fixed_dates 只固定事件日期，横截面排序仍按本组参数重算。
    for mode in modes:
        mode_output_dir = batch_output_dir / mode
        mode_output_dir.mkdir(parents=True, exist_ok=True)
        mode_selections: dict[str, dict[str, dict[str, object]]] = {}
        for experiment_index, experiment in enumerate(experiments, start=1):
            run_output_dir = mode_output_dir / f"{experiment_index:03d}_{experiment['run_name']}"
            row, top_map = run_single_experiment(
                args,
                db_client=db_client,
                data_portal=data_portal,
                research_store=research_store,
                execution_model=execution_model,
                exit_policy=exit_policy,
                start_date=start_date,
                resolved_end_date=resolved_end_date,
                mode=mode,
                run_output_dir=run_output_dir,
                experiment=experiment,
                st_lookback_trade_days=st_lookback_trade_days,
                min_signal_close_price=min_signal_close_price,
                fixed_event_dates=fixed_event_dates if mode == "fixed_dates" else None,
                precomputed_candidate_indicator_frame=precomputed_candidate_indicator,
            )
            summary_rows.append(row)
            overlap_group = str(experiment["overlap_group"])
            mode_selections.setdefault(overlap_group, {"experiment": experiment, "selections": {}})
            mode_selections[overlap_group]["selections"][str(experiment["selection_sort"])] = top_map
            log_event(
                "info",
                "smallcap_amount_shock_event_batch_run_finish",
                experiment_mode=mode,
                run_name=str(experiment["run_name"]),
                total_return=row.get("total_return"),
                max_drawdown=row.get("max_drawdown"),
                event_signal_days=row.get("event_signal_days"),
                closed_position_count=row.get("closed_position_count"),
            )
        for overlap_group, payload in mode_selections.items():
            overlap_rows.extend(
                build_overlap_rows(
                    mode=mode,
                    overlap_group=overlap_group,
                    experiment=payload["experiment"],
                    selections=payload["selections"],
                )
            )

    summary_frame = pd.DataFrame(summary_rows)
    if not summary_frame.empty:
        summary_frame.to_csv(batch_output_dir / "summary.csv", index=False, encoding="utf-8-sig")
        for mode in modes:
            mode_summary = summary_frame[summary_frame["experiment_mode"] == mode].copy()
            if not mode_summary.empty:
                mode_summary.to_csv(batch_output_dir / mode / "summary.csv", index=False, encoding="utf-8-sig")

    overlap_frame = pd.DataFrame(overlap_rows)
    if not overlap_frame.empty:
        overlap_frame.to_csv(batch_output_dir / "topn_overlap.csv", index=False, encoding="utf-8-sig")
        for mode in modes:
            mode_overlap = overlap_frame[overlap_frame["experiment_mode"] == mode].copy()
            if not mode_overlap.empty:
                mode_overlap.to_csv(batch_output_dir / mode / "topn_overlap.csv", index=False, encoding="utf-8-sig")

    (batch_output_dir / "batch_parameters.json").write_text(
        json.dumps(
            {
                "config_file": str(config_path),
                "raw_config": raw_config,
                "experiment_mode": args.experiment_mode,
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
                "parsed_parameters": {
                    "st_lookback_trade_days": st_lookback_trade_days,
                    "min_signal_close_price": min_signal_close_price,
                    "fixed_stop_loss_pct": fixed_stop_loss_pct,
                    "low_price_exit_threshold": low_price_exit_threshold,
                    "max_candidate_pool_size": max_candidate_pool_size,
                    "precomputed_candidate_indicator_rows": int(len(precomputed_candidate_indicator)),
                    "fixed_event_anchor_run_name": fixed_event_anchor_run_name,
                    "fixed_event_date_count": 0 if fixed_event_dates is None else len(fixed_event_dates),
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
    print(f"output_dir: {batch_output_dir}")


if __name__ == "__main__":
    main()
