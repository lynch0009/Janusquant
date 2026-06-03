"""公司行为相关对象定义。

把公司行为拆成三层对象：
1. 标准化事件：来自数据层，描述“发生了什么”
2. 应收对象：描述“账户已经享有，但还没到账/上市”的资产
3. 流水对象：描述“回测当天实际做了什么处理”
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class CorporateActionEvent:
    """标准化后的公司行为事件。"""

    event_type: str
    code: str
    operate_date: datetime
    settle_date: datetime
    cash_dividend_per_share: float = 0.0
    stock_dividend_ratio: float = 0.0
    stock_dividend_share_ratio: float = 0.0
    reserve_to_stock_ratio: float = 0.0
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CashDividendReceivable:
    """现金分红应收。"""

    code: str
    operate_date: datetime
    settle_date: datetime
    eligible_quantity: int
    cash_dividend_per_share: float
    amount: float
    origin_position_id: str
    signal_date: datetime
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StockDividendReceivable:
    """送转股应收。"""

    code: str
    operate_date: datetime
    settle_date: datetime
    eligible_quantity: int
    stock_dividend_ratio: float
    bonus_quantity: int
    allocated_cost_basis: float
    origin_position_id: str
    signal_date: datetime
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorporateActionRecord:
    """公司行为处理流水。"""

    trade_date: datetime
    code: str
    event_type: str
    stage: str
    operate_date: datetime
    settle_date: datetime
    eligible_quantity: int
    cash_amount: float = 0.0
    bonus_quantity: int = 0
    allocated_cost_basis: float = 0.0
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
