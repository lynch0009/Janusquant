"""回测执行参数定义。

这里集中描述资金、仓位、手续费和时间窗口等执行侧配置，避免这些参数
散落在引擎和执行器实现里。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True)
class EngineConfig:
    """描述一次回测执行所需的核心配置。"""

    # 初始账户资金，回测开始时的可用现金。
    initial_cash: float = 1_000_000.0
    # 组合允许同时持有的最大股票数量。
    max_positions: int = 20
    # 单个仓位默认占总资产的比例，常用于等权或固定比例分配。
    position_size_pct: float = 0.05
    # A 股最小交易单位，通常为 100 股一手。
    lot_size: int = 100
    # 买入手续费率，按成交额乘以该费率计算。
    buy_commission_rate: float = 0.0001
    # 卖出手续费率，按成交额乘以该费率计算。
    sell_commission_rate: float = 0.0001
    # 卖出印花税率，通常只在卖出侧收取。
    tax_rate: float = 0.0005
    # 买入撮合允许的最早时间。
    entry_start_time: time = time(9, 31)
    # 买入撮合允许的最晚时间。
    entry_end_time: time = time(10, 30)
    # 卖出撮合允许的最早时间。
    exit_start_time: time = time(14, 55)
    # 卖出撮合允许的最晚时间。
    exit_end_time: time = time(15, 0)
    # 风控检查允许触发的最早时间。
    risk_start_time: time = time(9, 31)
    # 风控检查允许触发的最晚时间。
    risk_end_time: time = time(15, 0)
    # 回测分析和报表使用的基准指数代码。
    benchmark_code: str = "sh.000905"
    # 绩效分析使用的无风险利率，通常用于夏普比率等指标。
    risk_free_rate: float = 0.014
    # 是否把当日信号延后到下一个交易日执行。
    execute_on_next_trade_date: bool = True
    # 是否输出主循环进度日志，便于长区间回测排查过程。
    progress_logging: bool = False
