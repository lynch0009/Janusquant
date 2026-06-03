"""持仓对象定义。

一笔持仓从开仓到平仓的完整状态都保存在这里，
也是收益统计和公司行为成本拆分的基础。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StockPosition:
    """描述一笔持仓从开仓到平仓的完整状态。"""

    position_id: str
    code: str
    quantity: int
    entry_trade_date: datetime
    entry_time: datetime
    entry_price: float
    target_exit_trade_date: datetime
    signal_date: datetime
    open_order_id: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    entry_transaction_cost: float = 0.0
    entry_trade_index: int | None = None
    share_cost_basis: float | None = None
    adjusted_avg_price: float | None = None
    cum_cash_dividend: float = 0.0
    cum_bonus_quantity: int = 0
    last_price: float | None = None
    status: str = "OPEN"
    exit_trade_date: datetime | None = None
    exit_time: datetime | None = None
    exit_price: float | None = None
    close_order_id: str | None = None
    exit_trade_index: int | None = None
    holding_trade_days: int | None = None
    exit_reason: str | None = None
    realized_pnl: float | None = None
    realized_return: float | None = None
    initial_stop_loss: float | None = None
    current_stop_loss: float | None = None
    take_profit_price: float | None = None
    highest_price_since_entry: float | None = None
    lowest_price_since_entry: float | None = None
    risk_budget: float | None = None

    def __post_init__(self) -> None:
        """补齐依赖其他字段推导出的成本口径。"""

        if self.share_cost_basis is None:
            self.share_cost_basis = self.entry_price * self.quantity
        self.refresh_adjusted_avg_price()

    def refresh_adjusted_avg_price(self) -> None:
        """根据当前总成本和股数刷新摊薄后的持仓成本价。"""

        if self.quantity > 0:
            self.adjusted_avg_price = self.share_cost_basis / self.quantity
        else:
            self.adjusted_avg_price = None

    @property
    def market_value(self) -> float:
        """按最近已知价格估算当前持仓市值。"""

        if self.last_price is None:
            return self.quantity * self.entry_price
        return self.quantity * self.last_price

    def close(
        self,
        *,
        trade_date: datetime,
        trade_time: datetime,
        trade_price: float,
        close_order_id: str,
        total_sell_cost: float,
        exit_trade_index: int | None = None,
        exit_reason: str | None = None,
    ) -> None:
        """完成平仓结算，并统一计算收益。

        realized_pnl 和 realized_return 都同时考虑：
        1. 当前剩余持仓对应的成本
        2. 开仓交易成本
        3. 卖出交易成本
        """

        self.status = "CLOSED"
        self.exit_trade_date = trade_date
        self.exit_time = trade_time
        self.exit_price = trade_price
        self.close_order_id = close_order_id
        self.exit_trade_index = exit_trade_index
        if self.entry_trade_index is not None and exit_trade_index is not None:
            self.holding_trade_days = max(exit_trade_index - self.entry_trade_index, 0)
        self.exit_reason = exit_reason

        gross_pnl = (trade_price * self.quantity) - self.share_cost_basis
        total_transaction_cost = self.entry_transaction_cost + total_sell_cost
        self.realized_pnl = gross_pnl - total_transaction_cost
        invested = self.share_cost_basis + self.entry_transaction_cost
        self.realized_return = self.realized_pnl / invested if invested > 0 else None

    def to_dict(self) -> dict[str, Any]:
        """导出为便于落表和 DataFrame 分析的字典。"""

        return asdict(self)


PositionState = StockPosition
