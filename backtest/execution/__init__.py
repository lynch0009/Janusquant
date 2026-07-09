"""执行层导出。

这一层包含回测引擎本体、执行参数以及不同成交模型的统一入口。
"""

from __future__ import annotations

from .config import EngineConfig

__all__ = [
    "BaseExecutionModel",
    "BaseMinuteExecutor",
    "DailyBarExecutor",
    "EngineConfig",
    "SignalDrivenBacktestEngine",
    "WindowFirstBarExecutor",
    "build_exit_trade_from_price",
    "calculate_entry_quantity",
    "calculate_required_cash",
]


def __getattr__(name: str):
    """按需导入重量级执行模块，避免包初始化时出现循环依赖。"""

    if name == "SignalDrivenBacktestEngine":
        from .engine import SignalDrivenBacktestEngine

        return SignalDrivenBacktestEngine
    if name in {
        "calculate_entry_quantity",
        "calculate_required_cash",
    }:
        from .trading import calculate_entry_quantity, calculate_required_cash

        exports = {
            "calculate_entry_quantity": calculate_entry_quantity,
            "calculate_required_cash": calculate_required_cash,
        }
        return exports[name]
    if name in {
        "BaseExecutionModel",
        "BaseMinuteExecutor",
        "DailyBarExecutor",
        "WindowFirstBarExecutor",
        "build_exit_trade_from_price",
    }:
        from .executors import (
            BaseExecutionModel,
            BaseMinuteExecutor,
            DailyBarExecutor,
            WindowFirstBarExecutor,
            build_exit_trade_from_price,
        )

        exports = {
            "BaseExecutionModel": BaseExecutionModel,
            "BaseMinuteExecutor": BaseMinuteExecutor,
            "DailyBarExecutor": DailyBarExecutor,
            "WindowFirstBarExecutor": WindowFirstBarExecutor,
            "build_exit_trade_from_price": build_exit_trade_from_price,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
