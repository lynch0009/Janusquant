from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from backtest.risk import AbsoluteLowPriceExitPolicy, CompositeExitPolicy, FixedStopLossExitPolicy
from backtest.strategies.smallcap_amount_shock_event import EVENT_SELECTION_SORTS
from backtest.strategies.smallcap_amount_shock_event_regime_hold import SmallCapAmountShockEventRegimeHoldStrategy
from backtest.utils.log import log_event

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "smallcap_amount_shock_event_regime_hold"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "smallcap_amount_shock_event_regime_hold"

WEEKDAY_ALIASES = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "friday": 4,
    "fri": 4,
}


@dataclass
class RegimeHoldBacktestComponents:
    db_client: DuckDBConfig
    frame_cache: FrameCache | None
    data_portal: DuckDBDataPortal
    strategy: SmallCapAmountShockEventRegimeHoldStrategy
    engine: SignalDrivenBacktestEngine
    start_date: datetime
    requested_end_date: datetime
    resolved_end_date: datetime
    amount_keep_groups: tuple[int, ...]
    ret_keep_groups: tuple[int, ...]
    st_lookback_trade_days: int | None
    min_signal_close_price: float | None
    fixed_stop_loss_pct: float | None
    low_price_exit_threshold: float | None
    weekly_fill_weekday: int


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


def parse_weekday(raw_value: str) -> int:
    text = str(raw_value or "").strip().lower()
    if text.isdigit():
        value = int(text)
    else:
        value = WEEKDAY_ALIASES.get(text)
    if value is None or value < 0 or value > 4:
        allowed = ", ".join(sorted(WEEKDAY_ALIASES))
        raise ValueError(f"weekly_fill_weekday must be 0..4 or one of: {allowed}")
    return int(value)


def build_run_output_dir(base_dir: Path, start_date: datetime, end_date: datetime) -> Path:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{start_date:%Y%m%d}_{end_date:%Y%m%d}_{run_timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the event-triggered small-cap regime hold strategy.")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-05-21")
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--top-k", type=int, default=4, help="Event-day max new event candidates.")
    parser.add_argument("--max-positions", type=int, default=4)
    parser.add_argument("--position-size-pct", type=float, default=0.25)
    parser.add_argument("--hold-days", type=int, default=20, help="Trend-exit activation trade days, not forced holding days.")
    parser.add_argument("--scheduled-hold-days", type=int, default=10000, help="Far future scheduled exit; real exits use risk rules.")
    parser.add_argument("--candidate-pool-size", type=int, default=200)
    parser.add_argument("--min-listing-trade-days", type=int, default=120)
    parser.add_argument("--amount-keep-groups", default="1,2,3,4,5")
    parser.add_argument("--ret-keep-groups", default="1,2,3,4,5")
    parser.add_argument("--min-research-ret-10d", type=float, default=0.12)
    parser.add_argument("--ret-top-pct", type=float, default=0.40)
    parser.add_argument("--selection-sort", default="composite_zscore", choices=sorted(EVENT_SELECTION_SORTS))
    parser.add_argument("--group-count", type=int, default=5)
    parser.add_argument("--amount-fast-window", type=int, default=5)
    parser.add_argument("--amount-slow-window", type=int, default=20)
    parser.add_argument("--ret-window", type=int, default=10)
    parser.add_argument("--index-code", default="sz.399303")
    parser.add_argument("--index-fast-ma", type=int, default=5)
    parser.add_argument("--index-slow-ma", type=int, default=60)
    parser.add_argument("--stock-ma-exit-window", type=int, default=10)
    parser.add_argument("--no-new-high-exit-days", type=int, default=5)
    parser.add_argument("--weekly-fill-weekday", default="Friday")
    parser.add_argument("--disable-weekly-fill", action="store_true")
    parser.add_argument("--disable-trend-exit", action="store_true")
    parser.add_argument("--slippage-bps", type=float, default=30.0)
    parser.add_argument("--signal-price-mode", default="qfq", choices=["raw", "qfq", "hfq"])
    parser.add_argument("--st-lookback-trade-days", default="100")
    parser.add_argument("--min-signal-close-price", default="1.5")
    parser.add_argument("--fixed-stop-loss-pct", default=None)
    parser.add_argument("--low-price-exit-threshold", default="1.3")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--cache-version", default="amount_shock_event_regime_hold_v1")
    return parser.parse_args()


def build_exit_policy(
    strategy: SmallCapAmountShockEventRegimeHoldStrategy,
    *,
    fixed_stop_loss_pct: float | None,
    low_price_exit_threshold: float | None,
    trend_exit_enabled: bool,
):
    policies = []
    if fixed_stop_loss_pct is not None:
        policies.append(FixedStopLossExitPolicy(stop_loss_pct=fixed_stop_loss_pct))
    if low_price_exit_threshold is not None:
        policies.append(AbsoluteLowPriceExitPolicy(min_low_price=low_price_exit_threshold))
    if trend_exit_enabled:
        policies.append(strategy.exit_policy())
    if not policies:
        return None
    if len(policies) == 1:
        return policies[0]
    return CompositeExitPolicy(policies)


def build_regime_hold_backtest_components(
    args: argparse.Namespace,
    *,
    strategy_cls=SmallCapAmountShockEventRegimeHoldStrategy,
    requested_end_date: datetime | None = None,
    progress_logging: bool = True,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_frame_cache: bool | None = None,
    liquidate_at_end: bool = True,
    enable_weekly_fill: bool | None = None,
    weekly_fill_weekday: int | None = None,
) -> RegimeHoldBacktestComponents:
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    requested_end_date = requested_end_date or datetime.strptime(args.end_date, "%Y-%m-%d")
    amount_keep_groups = parse_int_tuple(args.amount_keep_groups)
    ret_keep_groups = parse_int_tuple(args.ret_keep_groups)
    st_lookback_trade_days = parse_optional_int(args.st_lookback_trade_days)
    min_signal_close_price = parse_optional_float(args.min_signal_close_price)
    fixed_stop_loss_pct = parse_optional_float(args.fixed_stop_loss_pct)
    low_price_exit_threshold = parse_optional_float(args.low_price_exit_threshold)
    if weekly_fill_weekday is not None:
        resolved_weekly_fill_weekday = int(weekly_fill_weekday)
    elif hasattr(args, "weekly_fill_weekday"):
        resolved_weekly_fill_weekday = parse_weekday(args.weekly_fill_weekday)
    else:
        resolved_weekly_fill_weekday = 4
    resolved_enable_weekly_fill = (
        not bool(getattr(args, "disable_weekly_fill", False))
        if enable_weekly_fill is None
        else bool(enable_weekly_fill)
    )

    db_client = DuckDBConfig(read_only=True)
    cache_enabled = not bool(getattr(args, "disable_cache", False)) if use_frame_cache is None else bool(use_frame_cache)
    frame_cache = FrameCache(cache_dir, version=args.cache_version) if cache_enabled else None
    data_portal = DuckDBDataPortal(db_client) if frame_cache is None else CachedDuckDBDataPortal(db_client, frame_cache=frame_cache)
    trade_dates = data_portal.get_trade_calendar(start_date, requested_end_date)
    if not trade_dates:
        raise ValueError("No trade dates available in the requested date range.")
    resolved_end_date = trade_dates[-1]

    strategy = strategy_cls(
        top_k=args.top_k,
        max_positions=args.max_positions,
        hold_days=args.hold_days,
        scheduled_hold_days=args.scheduled_hold_days,
        min_listing_trade_days=args.min_listing_trade_days,
        candidate_pool_size=args.candidate_pool_size,
        amount_fast_window=args.amount_fast_window,
        amount_slow_window=args.amount_slow_window,
        ret_window=args.ret_window,
        group_count=args.group_count,
        amount_keep_groups=amount_keep_groups,
        ret_keep_groups=ret_keep_groups,
        min_research_ret_10d=args.min_research_ret_10d,
        ret_top_pct=args.ret_top_pct,
        selection_sort=args.selection_sort,
        signal_price_mode=args.signal_price_mode,
        st_lookback_trade_days=st_lookback_trade_days,
        min_signal_close_price=min_signal_close_price,
        index_code=args.index_code,
        index_fast_ma=args.index_fast_ma,
        index_slow_ma=args.index_slow_ma,
        weekly_fill_weekday=resolved_weekly_fill_weekday,
        enable_weekly_fill=resolved_enable_weekly_fill,
        post_hold_trend_start_days=args.hold_days,
        stock_ma_exit_window=args.stock_ma_exit_window,
        no_new_high_exit_days=args.no_new_high_exit_days,
    )
    exit_policy = build_exit_policy(
        strategy,
        fixed_stop_loss_pct=fixed_stop_loss_pct,
        low_price_exit_threshold=low_price_exit_threshold,
        trend_exit_enabled=not args.disable_trend_exit,
    )
    config = EngineConfig(
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
        position_size_pct=args.position_size_pct,
        execute_on_next_trade_date=True,
        progress_logging=progress_logging,
        liquidate_at_end=liquidate_at_end,
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
    return RegimeHoldBacktestComponents(
        db_client=db_client,
        frame_cache=frame_cache,
        data_portal=data_portal,
        strategy=strategy,
        engine=engine,
        start_date=start_date,
        requested_end_date=requested_end_date,
        resolved_end_date=resolved_end_date,
        amount_keep_groups=amount_keep_groups,
        ret_keep_groups=ret_keep_groups,
        st_lookback_trade_days=st_lookback_trade_days,
        min_signal_close_price=min_signal_close_price,
        fixed_stop_loss_pct=fixed_stop_loss_pct,
        low_price_exit_threshold=low_price_exit_threshold,
        weekly_fill_weekday=resolved_weekly_fill_weekday,
    )


def main() -> None:
    args = parse_args()
    components = build_regime_hold_backtest_components(args)
    log_event(
        "info",
        "smallcap_amount_shock_event_regime_hold_run_start",
        start_date=components.start_date,
        resolved_end_date=components.resolved_end_date,
        candidate_pool_size=args.candidate_pool_size,
        top_k=args.top_k,
        max_positions=args.max_positions,
        position_size_pct=args.position_size_pct,
        trend_start_days=args.hold_days,
        scheduled_hold_days=args.scheduled_hold_days,
        amount_keep_groups=list(components.amount_keep_groups),
        ret_keep_groups=list(components.ret_keep_groups),
        min_research_ret_10d=args.min_research_ret_10d,
        ret_top_pct=args.ret_top_pct,
        selection_sort=args.selection_sort,
        index_code=args.index_code,
        index_fast_ma=args.index_fast_ma,
        index_slow_ma=args.index_slow_ma,
        stock_ma_exit_window=args.stock_ma_exit_window,
        no_new_high_exit_days=args.no_new_high_exit_days,
        weekly_fill_weekday=components.weekly_fill_weekday,
        weekly_fill_enabled=not args.disable_weekly_fill,
        trend_exit_enabled=not args.disable_trend_exit,
        fixed_stop_loss_pct=components.fixed_stop_loss_pct,
        low_price_exit_threshold=components.low_price_exit_threshold,
        slippage_bps=args.slippage_bps,
        cache_enabled=components.frame_cache is not None,
        cache_dir=str(DEFAULT_CACHE_DIR),
        cache_version=args.cache_version,
    )
    result = components.engine.run(
        components.start_date,
        components.resolved_end_date,
        research_store=ResearchDailyHistoryStore(components.data_portal),
    )
    report = result.analyze()
    cache_summary = components.frame_cache.summary() if components.frame_cache is not None else {}

    output_dir = build_run_output_dir(Path(args.output_dir), components.start_date, components.resolved_end_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.orders_frame().to_csv(output_dir / "orders.csv", index=False)
    result.trades_frame().to_csv(output_dir / "trades.csv", index=False)
    result.equity_frame().to_csv(output_dir / "equity.csv", index=False)
    result.closed_positions_frame().to_csv(output_dir / "closed_positions.csv", index=False)
    components.strategy.regime_state_frame.to_csv(output_dir / "regime_state.csv", index=False, encoding="utf-8-sig")
    components.strategy.event_signal_frame.to_csv(output_dir / "event_signal_table.csv", index=False, encoding="utf-8-sig")
    components.strategy.weekly_fill_signal_frame.to_csv(output_dir / "weekly_fill_signal_table.csv", index=False, encoding="utf-8-sig")
    components.strategy.event_daily_window_features.to_csv(output_dir / "event_daily_window_features.csv", index=False, encoding="utf-8-sig")
    report.export(
        output_dir / "analytics",
        metadata={
            "strategy_name": "smallcap_amount_shock_event_regime_hold",
            "script_name": Path(__file__).name,
            "date_range": f"{components.start_date:%Y-%m-%d} -> {components.resolved_end_date:%Y-%m-%d}",
            "cache_summary": cache_summary,
            "parameters": {
                **vars(args),
                "amount_keep_groups": list(components.amount_keep_groups),
                "ret_keep_groups": list(components.ret_keep_groups),
                "st_lookback_trade_days": components.st_lookback_trade_days,
                "min_signal_close_price": components.min_signal_close_price,
                "fixed_stop_loss_pct": components.fixed_stop_loss_pct,
                "low_price_exit_threshold": components.low_price_exit_threshold,
                "weekly_fill_weekday": components.weekly_fill_weekday,
                "resolved_end_date": components.resolved_end_date.strftime("%Y-%m-%d"),
            },
        },
    )

    log_event(
        "info",
        "smallcap_amount_shock_event_regime_hold_run_finish",
        output_dir=str(output_dir),
        final_equity=report.summary.get("final_equity"),
        total_return=report.summary.get("total_return"),
        trade_count=report.summary.get("trade_count"),
        event_signal_rows=len(components.strategy.event_signal_frame),
        weekly_fill_signal_rows=len(components.strategy.weekly_fill_signal_frame),
        cache_summary=cache_summary,
    )
    print(json.dumps(report.summary, ensure_ascii=False, indent=2, default=str))
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
