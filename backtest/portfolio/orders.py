"""订单对象定义。

订单是“策略信号”与“实际成交”之间的桥梁，用于保留调度日期、请求参数、
成交状态和跳过原因，方便回测复盘。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
import uuid


@dataclass
class StockOrder:
    """描述一笔回测订单从创建到完成的完整状态。"""

    order_id: str
    code: str
    side: str
    signal_date: datetime
    scheduled_trade_date: datetime
    execution_model: str
    reason: str
    requested_budget: float | None = None
    requested_quantity: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "NEW"
    filled_quantity: int = 0
    filled_price: float | None = None
    executed_at: datetime | None = None
    commission: float = 0.0
    tax: float = 0.0
    trigger_price: float | None = None
    risk_rule_name: str | None = None
    skip_reason: str | None = None

    @classmethod
    def create(
        cls,
        *,
        code: str,
        side: str,
        signal_date: datetime,
        scheduled_trade_date: datetime,
        execution_model: str,
        reason: str,
        requested_budget: float | None = None,
        requested_quantity: int | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
        trigger_price: float | None = None,
        risk_rule_name: str | None = None,
    ) -> "StockOrder":
        """创建订单并自动生成唯一 order_id。"""

        return cls(
            order_id=uuid.uuid4().hex,
            code=code,
            side=side,
            signal_date=signal_date,
            scheduled_trade_date=scheduled_trade_date,
            execution_model=execution_model,
            reason=reason,
            requested_budget=requested_budget,
            requested_quantity=requested_quantity,
            score=score,
            metadata=metadata or {},
            trigger_price=trigger_price,
            risk_rule_name=risk_rule_name,
        )

    def mark_filled(self, trade: "TradeRecord") -> None:
        """用实际成交结果回填订单状态。"""

        self.status = "FILLED"
        self.filled_quantity = trade.quantity
        self.filled_price = trade.price
        self.executed_at = trade.trade_time
        self.commission = trade.commission
        self.tax = trade.tax
        self.skip_reason = None

    def mark_skipped(self, reason: str) -> None:
        """把订单标记为跳过，并记录原因。"""

        self.status = "SKIPPED"
        self.skip_reason = reason

    def to_dict(self) -> dict[str, Any]:
        """导出为字典，便于汇总和落盘。"""

        return asdict(self)
