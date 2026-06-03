from __future__ import annotations

from datetime import datetime

import pandas as pd

from backtest.execution.config import EngineConfig
from backtest.execution.executors import (
    DailyBarExecutor,
    QuantityEntryExecutionResult,
    build_exit_trade_from_price,
)
from backtest.portfolio import PositionState, TradeRecord
from backtest.strategies.models import DailyCandidate
from backtest.utils.metadata import merge_metadata
from backtest.utils.price_limits import decide_daily_buy_fill, decide_daily_sell_fill
from backtest.utils.trading import calculate_entry_quantity, calculate_required_cash


class SmallCapRotationDailyOpenExecutor(DailyBarExecutor):
    """小市值轮动专用日线开盘执行器。

    在普通日线开盘成交逻辑上，额外加入：
    1. 涨停买不到
    2. 跌停卖不出
    3. 盘中开板/开跌停时按涨跌停价成交
    """

    def __init__(self, *, slippage_bps: float = 0.0):
        super().__init__(price_field="open", slippage_bps=slippage_bps)

    def _resolve_buy_fill(
        self,
        candidate: DailyCandidate,
        daily_bars: pd.DataFrame,
    ):
        row = self._pick_row(daily_bars)
        if row is None:
            return None, None

        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        preclose = float(row["preclose"]) if "preclose" in row and pd.notna(row["preclose"]) else None
        is_st = bool(row["isST"]) if "isST" in row and pd.notna(row["isST"]) else False

        fallback_price = self._apply_slippage(open_price, side="BUY")
        fill_decision = decide_daily_buy_fill(
            candidate.code,
            preclose=preclose,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            fallback_price=fallback_price,
            is_st=is_st,
        )
        return row, fill_decision

    def execute_entry(
        self,
        candidate: DailyCandidate,
        trade_date: datetime,
        daily_bars: pd.DataFrame,
        budget: float,
        config: EngineConfig,
        target_exit_trade_date: datetime,
    ) -> tuple[PositionState | None, TradeRecord | None, float]:
        row, fill_decision = self._resolve_buy_fill(candidate, daily_bars)
        if row is None or fill_decision is None:
            return None, None, 0.0

        price = fill_decision.execution_price
        if not fill_decision.fillable or price is None or price <= 0:
            return None, None, 0.0

        quantity = calculate_entry_quantity(budget, price, config)
        if quantity <= 0:
            return None, None, 0.0

        result = self._build_entry_result(
            candidate=candidate,
            trade_date=trade_date,
            price=price,
            quantity=quantity,
            config=config,
            target_exit_trade_date=target_exit_trade_date,
            extra_metadata={"limit_state": fill_decision.limit_state},
        )
        return result.position, result.trade, result.cash_delta

    def execute_entry_by_quantity(
        self,
        candidate: DailyCandidate,
        trade_date: datetime,
        daily_bars: pd.DataFrame,
        requested_quantity: int,
        available_cash: float,
        config: EngineConfig,
        target_exit_trade_date: datetime,
    ) -> QuantityEntryExecutionResult:
        if requested_quantity <= 0:
            return QuantityEntryExecutionResult(None, None, 0.0, "invalid_requested_quantity")

        row, fill_decision = self._resolve_buy_fill(candidate, daily_bars)
        if row is None or fill_decision is None:
            return QuantityEntryExecutionResult(None, None, 0.0, "no_execution_data")

        price = fill_decision.execution_price
        if not fill_decision.fillable or price is None or price <= 0:
            return QuantityEntryExecutionResult(None, None, 0.0, fill_decision.reject_reason or "no_fill")

        required_cash = calculate_required_cash(requested_quantity, price, config)
        if required_cash > available_cash + 1e-9:
            return QuantityEntryExecutionResult(None, None, 0.0, "no_cash")

        return self._build_entry_result(
            candidate=candidate,
            trade_date=trade_date,
            price=price,
            quantity=requested_quantity,
            config=config,
            target_exit_trade_date=target_exit_trade_date,
            extra_metadata={"limit_state": fill_decision.limit_state},
        )

    def execute_exit(
        self,
        position: PositionState,
        trade_date: datetime,
        daily_bars: pd.DataFrame,
        config: EngineConfig,
        *,
        reason: str,
    ) -> tuple[TradeRecord | None, float]:
        row = self._pick_row(daily_bars)
        if row is None:
            return None, 0.0

        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        preclose = float(row["preclose"]) if "preclose" in row and pd.notna(row["preclose"]) else None
        is_st = bool(row["isST"]) if "isST" in row and pd.notna(row["isST"]) else False

        fallback_price = self._apply_slippage(open_price, side="SELL")
        fill_decision = decide_daily_sell_fill(
            position.code,
            preclose=preclose,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            fallback_price=fallback_price,
            is_st=is_st,
        )
        price = fill_decision.execution_price
        if not fill_decision.fillable or price is None or price <= 0:
            return None, 0.0

        return build_exit_trade_from_price(
            position,
            trade_date,
            self._trade_time(trade_date),
            price,
            config,
            reason=f"daily_open_{reason}",
            metadata=merge_metadata(position.metadata, {"limit_state": fill_decision.limit_state}),
        )
