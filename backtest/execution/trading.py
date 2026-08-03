"""Execution-side quantity and cash helpers."""

from __future__ import annotations

from backtest.execution.config import EngineConfig


def calculate_entry_quantity(budget: float, price: float, config: EngineConfig) -> int:
    """Calculate buy quantity under budget, commission, and lot-size constraints."""

    if budget <= 0 or price <= 0:
        return 0
    gross_budget = budget / (1 + config.buy_commission_rate)
    raw_shares = int(gross_budget // price)
    return (raw_shares // config.lot_size) * config.lot_size


def calculate_required_cash(quantity: int, price: float, config: EngineConfig) -> float:
    """Calculate total cash required for a buy order."""

    if quantity <= 0 or price <= 0:
        return 0.0
    notional = price * quantity
    commission = notional * config.buy_commission_rate
    return notional + commission


__all__ = [
    "calculate_entry_quantity",
    "calculate_required_cash",
]
