"""回测结果对象定义。

这一层负责把账户中的原始列表整理成结构化结果，
并提供导出 DataFrame 与调用分析器的统一入口。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from backtest.execution.config import EngineConfig
from backtest.utils.frame_utils import records_to_frame

from .corporate_actions import CorporateActionRecord
from .orders import StockOrder
from .positions import StockPosition


@dataclass(frozen=True)
class TradeRecord:
    """描述一笔已经成交的买卖记录。"""

    code: str
    side: str
    signal_date: datetime
    trade_date: datetime
    trade_time: datetime
    price: float
    quantity: int
    notional: float
    commission: float
    tax: float
    reason: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EquityPoint:
    """描述某个交易日收盘后的组合权益快照。"""

    trade_date: datetime
    cash: float
    market_value: float
    total_equity: float
    position_count: int
    cash_receivable_value: float = 0.0
    stock_receivable_value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkPoint:
    """描述某个交易日的基准指数收盘快照。"""

    trade_date: datetime
    code: str
    close: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    """回测完成后的统一结果对象。"""

    trades: list[TradeRecord]
    orders: list[StockOrder]
    equity_curve: list[EquityPoint]
    final_positions: dict[str, StockPosition]
    closed_positions: list[StockPosition]
    corporate_actions: list[CorporateActionRecord] = field(default_factory=list)
    benchmark_code: str = EngineConfig().benchmark_code
    risk_free_rate: float = EngineConfig().risk_free_rate
    benchmark_curve: list[BenchmarkPoint] = field(default_factory=list)

    def trades_frame(self) -> pd.DataFrame:
        """导出成交流水。"""

        return records_to_frame(self.trades)

    def orders_frame(self) -> pd.DataFrame:
        """导出订单流水。"""

        return records_to_frame(self.orders)

    def equity_frame(self) -> pd.DataFrame:
        """导出权益曲线。"""

        return records_to_frame(self.equity_curve)

    def closed_positions_frame(self) -> pd.DataFrame:
        """导出已平仓持仓明细。"""

        return records_to_frame(self.closed_positions)

    def benchmark_frame(self) -> pd.DataFrame:
        """导出基准指数曲线。"""

        return records_to_frame(self.benchmark_curve)

    def corporate_actions_frame(self) -> pd.DataFrame:
        """导出公司行为处理流水。"""

        return records_to_frame(self.corporate_actions)

    def analyze(self, *, annual_trading_days: int = 252, risk_free_rate: float | None = None):
        """调用分析器生成统计结果。"""

        from backtest.analytics import BacktestAnalyzer

        return BacktestAnalyzer(
            annual_trading_days=annual_trading_days,
            risk_free_rate=self.risk_free_rate if risk_free_rate is None else risk_free_rate,
        ).analyze(self)

    def summary(self) -> dict[str, float | int | None]:
        """返回分析摘要。"""

        return self.analyze().summary
