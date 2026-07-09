"""成交模型实现。

执行器负责把候选信号变成可落账的成交记录。不同执行器只关心：
在给定行情窗口里，用什么价格、什么时间成交。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, time

import pandas as pd

from backtest.execution.config import EngineConfig
from backtest.portfolio import PositionState, TradeRecord
from backtest.strategies import DailyCandidate
from backtest.utils.datetime_utils import combine_trade_date, to_pydatetime
from backtest.utils.frame_utils import first_sorted_row
from backtest.utils.metadata import copy_metadata, merge_metadata
from backtest.execution.trading import calculate_entry_quantity, calculate_required_cash


def build_exit_trade_from_price(
    position: PositionState,
    trade_date: datetime,
    trade_time: datetime,
    price: float,
    config: EngineConfig,
    *,
    reason: str,
    metadata: dict | None = None,
) -> tuple[TradeRecord | None, float]:
    """根据给定触发价格，直接构造一笔卖出成交记录。"""

    if price <= 0:
        return None, 0.0

    notional = price * position.quantity
    commission = notional * config.sell_commission_rate
    tax = notional * config.tax_rate
    cash_delta = notional - commission - tax

    trade = TradeRecord(
        code=position.code,
        side="SELL",
        signal_date=position.signal_date,
        trade_date=trade_date,
        trade_time=trade_time,
        price=price,
        quantity=position.quantity,
        notional=notional,
        commission=commission,
        tax=tax,
        reason=reason,
        score=position.score,
        metadata=metadata or copy_metadata(position.metadata),
    )
    return trade, cash_delta


@dataclass(frozen=True)
class QuantityEntryExecutionResult:
    """描述按请求股数执行买入后的结果。"""

    position: PositionState | None
    trade: TradeRecord | None
    cash_delta: float
    reject_reason: str | None = None


class BaseExecutionModel(ABC):
    """执行模型抽象基类。"""

    data_frequency = "minute"

    @abstractmethod
    def execute_entry(
        self,
        candidate: DailyCandidate,
        trade_date: datetime,
        minute_bars: pd.DataFrame,
        budget: float,
        config: EngineConfig,
        target_exit_trade_date: datetime,
    ) -> tuple[PositionState | None, TradeRecord | None, float]:
        """执行买入并返回持仓、成交记录和现金变化量。"""

        raise NotImplementedError

    @abstractmethod
    def execute_entry_by_quantity(
        self,
        candidate: DailyCandidate,
        trade_date: datetime,
        minute_bars: pd.DataFrame,
        requested_quantity: int,
        available_cash: float,
        config: EngineConfig,
        target_exit_trade_date: datetime,
    ) -> QuantityEntryExecutionResult:
        """按请求股数执行买入，并返回成交结果与失败原因。"""

        raise NotImplementedError

    @abstractmethod
    def execute_exit(
        self,
        position: PositionState,
        trade_date: datetime,
        minute_bars: pd.DataFrame,
        config: EngineConfig,
        *,
        reason: str,
    ) -> tuple[TradeRecord | None, float]:
        """执行卖出并返回成交记录和现金变化量。"""

        raise NotImplementedError


class BaseMinuteExecutor(BaseExecutionModel):
    """分钟线执行器基类，仅用于标识数据频率。"""

    data_frequency = "minute"


class WindowFirstBarExecutor(BaseMinuteExecutor):
    """
    使用窗口内第一根可用分钟线成交的执行器。

    优点是简单、稳定、容易复现；缺点是无法刻画更复杂的撮合细节。
    """

    def __init__(self, *, price_field: str = "close", slippage_bps: float = 0.0):
        """指定分钟线成交价字段以及双边滑点。"""

        self.price_field = price_field
        self.slippage_bps = slippage_bps

    def _pick_bar(self, minute_bars: pd.DataFrame) -> pd.Series | None:
        """挑选执行窗口里的第一根有效分钟线。"""

        return first_sorted_row(minute_bars)

    def _apply_slippage(self, price: float, *, side: str) -> float:
        """按买卖方向应用基点滑点。"""

        if self.slippage_bps == 0:
            return price
        multiplier = 1 + self.slippage_bps / 10_000 if side == "BUY" else 1 - self.slippage_bps / 10_000
        return price * multiplier

    def _build_entry_result(
        self,
        *,
        candidate: DailyCandidate,
        trade_date: datetime,
        trade_time: datetime,
        price: float,
        quantity: int,
        config: EngineConfig,
        target_exit_trade_date: datetime,
        extra_metadata: dict | None = None,
    ) -> QuantityEntryExecutionResult:
        """根据成交参数构造买入持仓与成交记录。"""

        metadata = merge_metadata(candidate.metadata, extra_metadata)
        notional = price * quantity
        commission = notional * config.buy_commission_rate
        cash_delta = -(notional + commission)
        trade = TradeRecord(
            code=candidate.code,
            side="BUY",
            signal_date=candidate.signal_date,
            trade_date=trade_date,
            trade_time=trade_time,
            price=price,
            quantity=quantity,
            notional=notional,
            commission=commission,
            tax=0.0,
            reason=str(candidate.metadata.get("entry_reason", "scheduled_entry")),
            score=candidate.score,
            metadata=metadata,
        )
        position = PositionState(
            position_id="",
            code=candidate.code,
            quantity=quantity,
            entry_trade_date=trade_date,
            entry_time=trade_time,
            entry_price=price,
            target_exit_trade_date=target_exit_trade_date,
            signal_date=candidate.signal_date,
            open_order_id="",
            score=candidate.score,
            metadata=metadata,
            last_price=price,
        )
        return QuantityEntryExecutionResult(
            position=position,
            trade=trade,
            cash_delta=cash_delta,
            reject_reason=None,
        )

    def execute_entry(
        self,
        candidate: DailyCandidate,
        trade_date: datetime,
        minute_bars: pd.DataFrame,
        budget: float,
        config: EngineConfig,
        target_exit_trade_date: datetime,
    ) -> tuple[PositionState | None, TradeRecord | None, float]:
        """按分钟窗口第一根 bar 执行买入。"""

        bar = self._pick_bar(minute_bars)
        if bar is None:
            return None, None, 0.0

        price = self._apply_slippage(float(bar[self.price_field]), side="BUY")
        if price <= 0:
            return None, None, 0.0

        quantity = calculate_entry_quantity(budget, price, config)
        if quantity <= 0:
            return None, None, 0.0

        result = self._build_entry_result(
            candidate=candidate,
            trade_date=trade_date,
            trade_time=to_pydatetime(bar["dt"]),
            price=price,
            quantity=quantity,
            config=config,
            target_exit_trade_date=target_exit_trade_date,
        )
        return result.position, result.trade, result.cash_delta

    def execute_entry_by_quantity(
        self,
        candidate: DailyCandidate,
        trade_date: datetime,
        minute_bars: pd.DataFrame,
        requested_quantity: int,
        available_cash: float,
        config: EngineConfig,
        target_exit_trade_date: datetime,
    ) -> QuantityEntryExecutionResult:
        """按请求股数执行分钟级买入。"""

        if requested_quantity <= 0:
            return QuantityEntryExecutionResult(None, None, 0.0, "invalid_requested_quantity")

        bar = self._pick_bar(minute_bars)
        if bar is None:
            return QuantityEntryExecutionResult(None, None, 0.0, "no_execution_data")

        price = self._apply_slippage(float(bar[self.price_field]), side="BUY")
        if price <= 0:
            return QuantityEntryExecutionResult(None, None, 0.0, "invalid_execution_price")

        required_cash = calculate_required_cash(requested_quantity, price, config)
        if required_cash > available_cash + 1e-9:
            return QuantityEntryExecutionResult(None, None, 0.0, "no_cash")

        return self._build_entry_result(
            candidate=candidate,
            trade_date=trade_date,
            trade_time=to_pydatetime(bar["dt"]),
            price=price,
            quantity=requested_quantity,
            config=config,
            target_exit_trade_date=target_exit_trade_date,
        )

    def execute_exit(
        self,
        position: PositionState,
        trade_date: datetime,
        minute_bars: pd.DataFrame,
        config: EngineConfig,
        *,
        reason: str,
    ) -> tuple[TradeRecord | None, float]:
        """按分钟窗口第一根 bar 执行卖出。"""

        bar = self._pick_bar(minute_bars)
        if bar is None:
            return None, 0.0

        price = self._apply_slippage(float(bar[self.price_field]), side="SELL")
        if price <= 0:
            return None, 0.0

        return build_exit_trade_from_price(
            position,
            trade_date,
            to_pydatetime(bar["dt"]),
            price,
            config,
            reason=reason,
            metadata=copy_metadata(position.metadata),
        )


class DailyBarExecutor(BaseExecutionModel):
    """按日线开盘价或收盘价成交的执行器。"""

    data_frequency = "daily"

    def __init__(self, *, price_field: str = "open", slippage_bps: float = 0.0):
        """指定使用开盘价还是收盘价成交。"""

        if price_field not in {"open", "close"}:
            raise ValueError("price_field must be 'open' or 'close'")
        self.price_field = price_field
        self.slippage_bps = slippage_bps

    def _pick_row(self, daily_frame: pd.DataFrame) -> pd.Series | None:
        """从日线快照里选出用于成交的记录。"""

        return first_sorted_row(daily_frame)

    def _apply_slippage(self, price: float, *, side: str) -> float:
        """按买卖方向应用基点滑点。"""

        if self.slippage_bps == 0:
            return price
        multiplier = 1 + self.slippage_bps / 10_000 if side == "BUY" else 1 - self.slippage_bps / 10_000
        return price * multiplier

    def _trade_time(self, trade_date: datetime) -> datetime:
        """把日线成交映射到一个可读的成交时间点。"""

        chosen_time = time(9, 30) if self.price_field == "open" else time(15, 0)
        return combine_trade_date(trade_date, chosen_time)

    def _build_entry_result(
        self,
        *,
        candidate: DailyCandidate,
        trade_date: datetime,
        price: float,
        quantity: int,
        config: EngineConfig,
        target_exit_trade_date: datetime,
        extra_metadata: dict | None = None,
    ) -> QuantityEntryExecutionResult:
        """根据成交参数构造日线买入结果。"""

        trade_time = self._trade_time(trade_date)
        metadata = merge_metadata(candidate.metadata, extra_metadata)
        notional = price * quantity
        commission = notional * config.buy_commission_rate
        cash_delta = -(notional + commission)
        trade = TradeRecord(
            code=candidate.code,
            side="BUY",
            signal_date=candidate.signal_date,
            trade_date=trade_date,
            trade_time=trade_time,
            price=price,
            quantity=quantity,
            notional=notional,
            commission=commission,
            tax=0.0,
            reason=str(candidate.metadata.get("entry_reason", f"daily_{self.price_field}_entry")),
            score=candidate.score,
            metadata=metadata,
        )
        position = PositionState(
            position_id="",
            code=candidate.code,
            quantity=quantity,
            entry_trade_date=trade_date,
            entry_time=trade_time,
            entry_price=price,
            target_exit_trade_date=target_exit_trade_date,
            signal_date=candidate.signal_date,
            open_order_id="",
            score=candidate.score,
            metadata=metadata,
            last_price=price,
        )
        return QuantityEntryExecutionResult(
            position=position,
            trade=trade,
            cash_delta=cash_delta,
            reject_reason=None,
        )

    def execute_entry(
        self,
        candidate: DailyCandidate,
        trade_date: datetime,
        minute_bars: pd.DataFrame,
        budget: float,
        config: EngineConfig,
        target_exit_trade_date: datetime,
    ) -> tuple[PositionState | None, TradeRecord | None, float]:
        """按日线价格执行买入。"""

        row = self._pick_row(minute_bars)
        if row is None:
            return None, None, 0.0

        price = self._apply_slippage(float(row[self.price_field]), side="BUY")
        if price <= 0:
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
        )
        return result.position, result.trade, result.cash_delta

    def execute_entry_by_quantity(
        self,
        candidate: DailyCandidate,
        trade_date: datetime,
        minute_bars: pd.DataFrame,
        requested_quantity: int,
        available_cash: float,
        config: EngineConfig,
        target_exit_trade_date: datetime,
    ) -> QuantityEntryExecutionResult:
        """按请求股数执行日线买入。"""

        if requested_quantity <= 0:
            return QuantityEntryExecutionResult(None, None, 0.0, "invalid_requested_quantity")

        row = self._pick_row(minute_bars)
        if row is None:
            return QuantityEntryExecutionResult(None, None, 0.0, "no_execution_data")

        price = self._apply_slippage(float(row[self.price_field]), side="BUY")
        if price <= 0:
            return QuantityEntryExecutionResult(None, None, 0.0, "invalid_execution_price")

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
        )

    def execute_exit(
        self,
        position: PositionState,
        trade_date: datetime,
        minute_bars: pd.DataFrame,
        config: EngineConfig,
        *,
        reason: str,
    ) -> tuple[TradeRecord | None, float]:
        """按日线价格执行卖出。"""

        row = self._pick_row(minute_bars)
        if row is None:
            return None, 0.0

        price = self._apply_slippage(float(row[self.price_field]), side="SELL")
        if price <= 0:
            return None, 0.0

        return build_exit_trade_from_price(
            position,
            trade_date,
            self._trade_time(trade_date),
            price,
            config,
            reason=f"daily_{self.price_field}_{reason}",
            metadata=copy_metadata(position.metadata),
        )
