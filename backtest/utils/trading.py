"""交易数量和现金测算工具。"""

from __future__ import annotations

from backtest.execution.config import EngineConfig


def calculate_entry_quantity(budget: float, price: float, config: EngineConfig) -> int:
    """根据预算、价格和最小手数限制计算可买数量。"""

    if budget <= 0 or price <= 0:
        return 0
    gross_budget = budget / (1 + config.buy_commission_rate)
    raw_shares = int(gross_budget // price)
    return (raw_shares // config.lot_size) * config.lot_size


def calculate_required_cash(quantity: int, price: float, config: EngineConfig) -> float:
    """计算买入指定股数所需的总现金。"""

    if quantity <= 0 or price <= 0:
        return 0.0
    notional = price * quantity
    commission = notional * config.buy_commission_rate
    return notional + commission
