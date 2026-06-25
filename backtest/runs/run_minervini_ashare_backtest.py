# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.data import MongoDataPortal
from backtest.db import MongoDBConfig
from backtest.execution import EngineConfig, SignalDrivenBacktestEngine
from backtest.execution.smallcap_rotation_executor import SmallCapRotationDailyOpenExecutor
from backtest.portfolio import EqualSlotSizer
from backtest.risk import (
    CloseBelowMaExitPolicy,
    CompositeExitPolicy,
    FixedStopLossExitPolicy,
    PositionStopExitPolicy,
)
from backtest.strategies import MinerviniAshareStrategy


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "minervini_ashare_backtest"


def build_run_output_dir(base_dir: Path, start_date: datetime, end_date: datetime) -> Path:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}_{run_timestamp}"
    return base_dir / run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A-share Minervini-style backtest.")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-04-04")
    parser.add_argument("--benchmark-code", default=EngineConfig().benchmark_code)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--rebalance-every", type=int, default=10)
    parser.add_argument("--min-listing-trade-days", type=int, default=250)
    parser.add_argument("--min-liqa-mv", type=float, default=4e9)
    parser.add_argument("--max-liqa-mv", type=float, default=None)
    parser.add_argument("--min-revenue-yoy", type=float, default=0.05)
    parser.add_argument("--min-net-profit-yoy", type=float, default=0.10)
    parser.add_argument("--min-roe", type=float, default=0.08)
    parser.add_argument("--min-cfo-to-np", type=float, default=None)
    parser.add_argument("--min-revenue-acceleration", type=float, default=None)
    parser.add_argument("--min-net-profit-acceleration", type=float, default=None)
    parser.add_argument("--min-rps", type=float, default=85.0)
    parser.add_argument("--min-close-to-high-250", type=float, default=0.75)
    parser.add_argument("--min-above-low-250", type=float, default=1.25)
    parser.add_argument("--breakout-buffer-pct", type=float, default=0.0)
    parser.add_argument("--min-breakout-volume-ratio", type=float, default=1.5)
    parser.add_argument("--platform-window", type=int, default=20)
    parser.add_argument("--max-platform-depth", type=float, default=0.20)
    parser.add_argument("--vcp-short-window", type=int, default=5)
    parser.add_argument("--vcp-mid-window", type=int, default=10)
    parser.add_argument("--vcp-base-window", type=int, default=20)
    parser.add_argument("--vcp-long-window", type=int, default=40)
    parser.add_argument("--max-vcp-depth", type=float, default=0.30)
    parser.add_argument("--stop-atr-multiple", type=float, default=1.5)
    parser.add_argument("--max-initial-stop-pct", type=float, default=0.12)
    parser.add_argument("--risk-fraction", type=float, default=0.005)
    parser.add_argument("--add-on-risk-fraction", type=float, default=0.0025)
    parser.add_argument("--add-on-short-pivot-window", type=int, default=10)
    parser.add_argument("--min-add-on-volume-ratio", type=float, default=1.2)
    parser.add_argument("--add-on-trigger-r-multiples", default="0.5,1.0")
    parser.add_argument("--max-add-on-count", type=int, default=2)
    parser.add_argument("--price-mode", default="hfq", choices=["raw", "qfq", "hfq"])
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--fixed-stop-loss-pct", type=float, default=None)
    parser.add_argument("--ma-stop-window", type=int, default=50)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    requested_end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    db_client = MongoDBConfig()
    data_portal = MongoDataPortal(db_client)
    trade_dates = data_portal.get_trade_calendar(start_date, requested_end_date)
    if not trade_dates:
        raise ValueError("No trade dates available in the requested date range.")

    resolved_end_date = trade_dates[-1]
    # 命令行里把加仓触发倍数写成逗号分隔字符串，
    # 方便直接试验 "0.5R, 1.0R, ..." 这类 pyramiding 阶梯。
    add_on_trigger_r_multiples = tuple(
        float(item.strip())
        for item in str(args.add_on_trigger_r_multiples).split(",")
        if item.strip()
    )
    strategy = MinerviniAshareStrategy(
        benchmark_code=args.benchmark_code,
        top_k=args.top_k,
        hold_days=args.hold_days,
        rebalance_every_n_trade_days=args.rebalance_every,
        min_listing_trade_days=args.min_listing_trade_days,
        min_liqa_mv=args.min_liqa_mv,
        max_liqa_mv=args.max_liqa_mv,
        min_revenue_yoy=args.min_revenue_yoy,
        min_net_profit_yoy=args.min_net_profit_yoy,
        min_roe=args.min_roe,
        min_cfo_to_np=args.min_cfo_to_np,
        min_revenue_acceleration=args.min_revenue_acceleration,
        min_net_profit_acceleration=args.min_net_profit_acceleration,
        min_rps=args.min_rps,
        min_close_to_high_250=args.min_close_to_high_250,
        min_above_low_250=args.min_above_low_250,
        breakout_buffer_pct=args.breakout_buffer_pct,
        min_breakout_volume_ratio=args.min_breakout_volume_ratio,
        platform_window=args.platform_window,
        max_platform_depth=args.max_platform_depth,
        vcp_short_window=args.vcp_short_window,
        vcp_mid_window=args.vcp_mid_window,
        vcp_base_window=args.vcp_base_window,
        vcp_long_window=args.vcp_long_window,
        max_vcp_depth=args.max_vcp_depth,
        stop_atr_multiple=args.stop_atr_multiple,
        max_initial_stop_pct=args.max_initial_stop_pct,
        risk_fraction=args.risk_fraction,
        add_on_risk_fraction=args.add_on_risk_fraction,
        add_on_short_pivot_window=args.add_on_short_pivot_window,
        min_add_on_volume_ratio=args.min_add_on_volume_ratio,
        add_on_trigger_r_multiples=add_on_trigger_r_multiples,
        max_add_on_count=args.max_add_on_count,
        price_mode=args.price_mode,
    )

    execution_model = SmallCapRotationDailyOpenExecutor(slippage_bps=args.slippage_bps)
    exit_policies = []
    if args.fixed_stop_loss_pct is not None:
        exit_policies.append(FixedStopLossExitPolicy(stop_loss_pct=args.fixed_stop_loss_pct))
    else:
        # 默认优先使用 Minervini 策略自己生成的 stop，
        # 而不是再额外覆盖成一个固定百分比止损。
        exit_policies.append(PositionStopExitPolicy())
    if args.ma_stop_window is not None:
        exit_policies.append(CloseBelowMaExitPolicy(ma_window=args.ma_stop_window, price_mode=args.price_mode))

    exit_policy = None
    if len(exit_policies) == 1:
        exit_policy = exit_policies[0]
    elif len(exit_policies) > 1:
        exit_policy = CompositeExitPolicy(exit_policies)

    config = EngineConfig(
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
        position_size_pct=1 / max(args.max_positions, 1),
        benchmark_code=args.benchmark_code,
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
    )

    result = engine.run(start_date, resolved_end_date)
    report = result.analyze()

    output_dir = build_run_output_dir(Path(args.output_dir), start_date, resolved_end_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.orders_frame().to_csv(output_dir / "orders.csv", index=False)
    result.trades_frame().to_csv(output_dir / "trades.csv", index=False)
    result.equity_frame().to_csv(output_dir / "equity.csv", index=False)
    result.closed_positions_frame().to_csv(output_dir / "closed_positions.csv", index=False)
    report.export(
        output_dir / "analytics",
        metadata={
            "strategy_name": "minervini_ashare_backtest",
            "script_name": Path(__file__).name,
            "date_range": f"{start_date:%Y-%m-%d} -> {resolved_end_date:%Y-%m-%d}",
            "parameters": {
                **vars(args),
                "resolved_end_date": resolved_end_date.strftime("%Y-%m-%d"),
            },
        },
    )

    print(json.dumps(report.summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
