"""退出规则实现。

这一层统一描述：
1. 退出规则运行在哪个阶段
2. 规则依赖哪些历史数据
3. 规则命中后返回什么退出决策
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

import pandas as pd

from backtest.portfolio.positions import StockPosition
from backtest.utils.datetime_utils import trade_time_for_frequency
from backtest.utils.frame_utils import first_sorted_row, sort_frame


EXIT_STAGE_INTRADAY = "intraday"
EXIT_STAGE_CLOSE_CONFIRMED = "close_confirmed"


@dataclass(frozen=True)
class ExitDecision:
    """描述一次已经满足条件的退出决策。"""

    reason: str
    trade_price: float | None = None
    trade_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExitDataRequirement:
    """描述退出规则需要的数据。"""

    stage: str = EXIT_STAGE_INTRADAY
    history_frequency: str | None = None
    history_lookback: int = 0
    history_fields: tuple[str, ...] = ()
    price_mode: str = "raw"


def _risk_state(position: StockPosition) -> dict[str, Any]:
    """从持仓 metadata 中取出风控运行时状态，并在缺失时自动初始化。"""

    return position.metadata.setdefault("_risk_state", {})


def _first_trade_point(
    market_frame: pd.DataFrame,
    trade_date: datetime,
    *,
    data_frequency: str,
) -> tuple[float, datetime] | None:
    """读取给定行情窗口里的第一个可成交价格点。"""

    if market_frame is None or market_frame.empty:
        return None

    sort_field = "dt" if "dt" in market_frame.columns else market_frame.columns[0]
    row = first_sorted_row(market_frame, sort_field=sort_field)
    if row is None:
        return None
    trade_time = trade_time_for_frequency(row.get("dt"), trade_date, data_frequency=data_frequency)
    return float(row["open"]), trade_time


class BaseExitPolicy(ABC):
    """退出规则抽象基类。"""

    def initialize_position(self, position: StockPosition) -> None:
        """在开仓时初始化风控状态，默认无需处理。"""

        return None

    def data_requirement(self) -> ExitDataRequirement:
        """返回规则所需的阶段与数据要求。默认是日内执行且不依赖历史窗口。"""

        return ExitDataRequirement()

    def policies_for_stage(self, stage: str) -> list["BaseExitPolicy"]:
        """按阶段过滤规则。单规则默认只有自己。"""

        requirement = self.data_requirement()
        return [self] if requirement.stage == stage else []

    @abstractmethod
    def evaluate(
        self,
        position: StockPosition,
        trade_date: datetime,
        market_frame: pd.DataFrame | None,
        *,
        data_frequency: str,
        history_frame: pd.DataFrame | None = None,
    ) -> ExitDecision | None:
        """判断当前持仓在给定行情下是否应该退出。"""

        raise NotImplementedError


class CompositeExitPolicy(BaseExitPolicy):
    """把多个退出规则按顺序串联执行。"""

    def __init__(self, policies: list[BaseExitPolicy]):
        self.policies = [policy for policy in policies if policy is not None]

    def initialize_position(self, position: StockPosition) -> None:
        for policy in self.policies:
            policy.initialize_position(position)

    def policies_for_stage(self, stage: str) -> list[BaseExitPolicy]:
        selected: list[BaseExitPolicy] = []
        for policy in self.policies:
            selected.extend(policy.policies_for_stage(stage))
        return selected

    def evaluate(
        self,
        position: StockPosition,
        trade_date: datetime,
        market_frame: pd.DataFrame | None,
        *,
        data_frequency: str,
        history_frame: pd.DataFrame | None = None,
    ) -> ExitDecision | None:
        """兼容旧调用方式：顺序执行，命中一个就返回。"""

        for policy in self.policies:
            decision = policy.evaluate(
                position,
                trade_date,
                market_frame,
                data_frequency=data_frequency,
                history_frame=history_frame,
            )
            if decision is not None:
                return decision
        return None


class FixedPriceExitPolicy(BaseExitPolicy):
    """固定百分比止损/止盈/跟踪止损规则。"""

    def __init__(
        self,
        *,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        trailing_stop_pct: float | None = None,
        intrabar_sequence: str = "stop_first",
    ):
        if intrabar_sequence not in {"stop_first", "take_first"}:
            raise ValueError("intrabar_sequence must be 'stop_first' or 'take_first'")
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.intrabar_sequence = intrabar_sequence

    def initialize_position(self, position: StockPosition) -> None:
        position.highest_price_since_entry = position.entry_price
        position.lowest_price_since_entry = position.entry_price
        if self.stop_loss_pct is not None:
            position.initial_stop_loss = position.entry_price * (1 - self.stop_loss_pct)
            position.current_stop_loss = position.initial_stop_loss
        if self.take_profit_pct is not None:
            position.take_profit_price = position.entry_price * (1 + self.take_profit_pct)

    def evaluate(
        self,
        position: StockPosition,
        trade_date: datetime,
        market_frame: pd.DataFrame | None,
        *,
        data_frequency: str,
        history_frame: pd.DataFrame | None = None,
    ) -> ExitDecision | None:
        if market_frame is None or market_frame.empty:
            return None

        frame = sort_frame(market_frame)
        if position.highest_price_since_entry is None:
            position.highest_price_since_entry = position.entry_price
        if position.lowest_price_since_entry is None:
            position.lowest_price_since_entry = position.entry_price

        for row in frame.itertuples(index=False):
            trade_time = trade_time_for_frequency(getattr(row, "dt", None), trade_date, data_frequency=data_frequency)
            decision = self._evaluate_bar(
                position,
                trade_time=trade_time,
                open_price=float(row.open),
                high_price=float(row.high),
                low_price=float(row.low),
            )
            if decision is not None:
                return decision
            self._update_extremes(position, high_price=float(row.high), low_price=float(row.low))
        return None

    def _evaluate_bar(
        self,
        position: StockPosition,
        *,
        trade_time: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
    ) -> ExitDecision | None:
        stop_price = position.current_stop_loss
        take_profit_price = position.take_profit_price

        if stop_price is not None and open_price <= stop_price:
            return self._build_exit_decision(position, "stop", open_price, trade_time)
        if take_profit_price is not None and open_price >= take_profit_price:
            return self._build_exit_decision(position, "take", open_price, trade_time)

        hit_stop = stop_price is not None and low_price <= stop_price
        hit_take = take_profit_price is not None and high_price >= take_profit_price

        if hit_stop and hit_take:
            if self.intrabar_sequence == "take_first":
                return self._build_exit_decision(position, "take", take_profit_price, trade_time)
            return self._build_exit_decision(position, "stop", stop_price, trade_time)
        if hit_stop:
            return self._build_exit_decision(position, "stop", stop_price, trade_time)
        if hit_take:
            return self._build_exit_decision(position, "take", take_profit_price, trade_time)
        return None

    def _build_exit_decision(
        self,
        position: StockPosition,
        side: str,
        trade_price: float,
        trade_time: datetime,
    ) -> ExitDecision:
        if side == "take":
            return ExitDecision(
                reason="take_profit",
                trade_price=trade_price,
                trade_time=trade_time,
                metadata={"take_profit_price": position.take_profit_price},
            )

        reason = "stop_loss"
        if (
            self.trailing_stop_pct is not None
            and position.current_stop_loss is not None
            and position.initial_stop_loss is not None
            and position.current_stop_loss > position.initial_stop_loss
        ):
            reason = "trailing_stop"
        return ExitDecision(
            reason=reason,
            trade_price=trade_price,
            trade_time=trade_time,
            metadata={"stop_price": position.current_stop_loss},
        )

    def _update_extremes(self, position: StockPosition, *, high_price: float, low_price: float) -> None:
        if position.highest_price_since_entry is None:
            position.highest_price_since_entry = high_price
        else:
            position.highest_price_since_entry = max(position.highest_price_since_entry, high_price)

        if position.lowest_price_since_entry is None:
            position.lowest_price_since_entry = low_price
        else:
            position.lowest_price_since_entry = min(position.lowest_price_since_entry, low_price)

        if self.trailing_stop_pct is not None and position.highest_price_since_entry is not None:
            trailing_stop = position.highest_price_since_entry * (1 - self.trailing_stop_pct)
            if position.current_stop_loss is None:
                position.current_stop_loss = trailing_stop
            else:
                position.current_stop_loss = max(position.current_stop_loss, trailing_stop)


class FixedStopLossExitPolicy(FixedPriceExitPolicy):
    """固定止损规则，只保留固定亏损退出。"""

    def __init__(self, *, stop_loss_pct: float):
        if stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive")
        super().__init__(stop_loss_pct=stop_loss_pct)


class PositionStopExitPolicy(FixedPriceExitPolicy):
    """直接使用持仓对象上已经写好的 stop 值。"""

    def __init__(self, *, trailing_stop_pct: float | None = None, intrabar_sequence: str = "stop_first"):
        super().__init__(
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=trailing_stop_pct,
            intrabar_sequence=intrabar_sequence,
        )

    def initialize_position(self, position: StockPosition) -> None:
        # Minervini 策略会在信号阶段自己算好止损价。
        # 这个 policy 只负责确保这些字段已经挂到持仓对象上，
        # 让通用的 FixedPriceExitPolicy 能直接消费。
        if position.highest_price_since_entry is None:
            position.highest_price_since_entry = position.entry_price
        if position.lowest_price_since_entry is None:
            position.lowest_price_since_entry = position.entry_price
        if position.current_stop_loss is None and position.initial_stop_loss is not None:
            position.current_stop_loss = position.initial_stop_loss


class AbsoluteLowPriceExitPolicy(BaseExitPolicy):
    """日线低价跌破固定价格阈值时退出。"""

    def __init__(self, *, min_low_price: float):
        if min_low_price <= 0:
            raise ValueError("min_low_price must be positive")
        self.min_low_price = float(min_low_price)

    def evaluate(
        self,
        position: StockPosition,
        trade_date: datetime,
        market_frame: pd.DataFrame | None,
        *,
        data_frequency: str,
        history_frame: pd.DataFrame | None = None,
    ) -> ExitDecision | None:
        if market_frame is None or market_frame.empty:
            return None

        frame = sort_frame(market_frame)
        for row in frame.itertuples(index=False):
            low_price = float(row.low)
            if low_price > self.min_low_price:
                continue
            trade_time = trade_time_for_frequency(getattr(row, "dt", None), trade_date, data_frequency=data_frequency)
            open_price = float(row.open)
            high_price = float(row.high)
            close_price = float(row.close)
            trigger_price = open_price if open_price <= self.min_low_price else self.min_low_price
            return ExitDecision(
                reason="low_price_exit",
                trade_price=trigger_price,
                trade_time=trade_time,
                metadata={
                    "threshold_price": self.min_low_price,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                },
            )
        return None


class AtrExitPolicy(BaseExitPolicy):
    """基于 ATR 的止损/止盈规则。"""

    def __init__(
        self,
        *,
        stop_atr_multiple: float = 2.0,
        take_profit_atr_multiple: float | None = None,
        atr_field: str = "atr_14",
        intrabar_sequence: str = "stop_first",
    ):
        if intrabar_sequence not in {"stop_first", "take_first"}:
            raise ValueError("intrabar_sequence must be 'stop_first' or 'take_first'")
        self.stop_atr_multiple = stop_atr_multiple
        self.take_profit_atr_multiple = take_profit_atr_multiple
        self.atr_field = atr_field
        self.intrabar_sequence = intrabar_sequence

    def initialize_position(self, position: StockPosition) -> None:
        atr_value = position.metadata.get(self.atr_field)
        if atr_value is None or atr_value <= 0:
            return
        state = _risk_state(position)
        state["atr_value"] = float(atr_value)
        state["atr_stop_price"] = position.entry_price - float(atr_value) * self.stop_atr_multiple
        if self.take_profit_atr_multiple is not None:
            state["atr_take_profit_price"] = position.entry_price + float(atr_value) * self.take_profit_atr_multiple

    def evaluate(
        self,
        position: StockPosition,
        trade_date: datetime,
        market_frame: pd.DataFrame | None,
        *,
        data_frequency: str,
        history_frame: pd.DataFrame | None = None,
    ) -> ExitDecision | None:
        state = _risk_state(position)
        stop_price = state.get("atr_stop_price")
        take_profit_price = state.get("atr_take_profit_price")
        if stop_price is None and take_profit_price is None:
            return None
        if market_frame is None or market_frame.empty:
            return None

        frame = sort_frame(market_frame)
        for row in frame.itertuples(index=False):
            trade_time = trade_time_for_frequency(getattr(row, "dt", None), trade_date, data_frequency=data_frequency)
            open_price = float(row.open)
            high_price = float(row.high)
            low_price = float(row.low)

            if stop_price is not None and open_price <= stop_price:
                return ExitDecision("atr_stop", open_price, trade_time, {"atr_stop_price": stop_price})
            if take_profit_price is not None and open_price >= take_profit_price:
                return ExitDecision("atr_take_profit", open_price, trade_time, {"atr_take_profit_price": take_profit_price})

            hit_stop = stop_price is not None and low_price <= stop_price
            hit_take = take_profit_price is not None and high_price >= take_profit_price
            if hit_stop and hit_take:
                if self.intrabar_sequence == "take_first":
                    return ExitDecision("atr_take_profit", take_profit_price, trade_time, {"atr_take_profit_price": take_profit_price})
                return ExitDecision("atr_stop", stop_price, trade_time, {"atr_stop_price": stop_price})
            if hit_stop:
                return ExitDecision("atr_stop", stop_price, trade_time, {"atr_stop_price": stop_price})
            if hit_take:
                return ExitDecision("atr_take_profit", take_profit_price, trade_time, {"atr_take_profit_price": take_profit_price})
        return None


class BreakEvenExitPolicy(BaseExitPolicy):
    """达到一定浮盈后，把止损上移到保本价附近。"""

    def __init__(
        self,
        *,
        trigger_profit_pct: float = 0.05,
        buffer_pct: float = 0.0,
    ):
        self.trigger_profit_pct = trigger_profit_pct
        self.buffer_pct = buffer_pct

    def initialize_position(self, position: StockPosition) -> None:
        state = _risk_state(position)
        state["breakeven_armed"] = False
        state["breakeven_stop_price"] = position.entry_price * (1 + self.buffer_pct)

    def evaluate(
        self,
        position: StockPosition,
        trade_date: datetime,
        market_frame: pd.DataFrame | None,
        *,
        data_frequency: str,
        history_frame: pd.DataFrame | None = None,
    ) -> ExitDecision | None:
        if market_frame is None or market_frame.empty:
            return None

        state = _risk_state(position)
        frame = sort_frame(market_frame)
        trigger_price = position.entry_price * (1 + self.trigger_profit_pct)
        stop_price = state.get("breakeven_stop_price", position.entry_price)

        for row in frame.itertuples(index=False):
            trade_time = trade_time_for_frequency(getattr(row, "dt", None), trade_date, data_frequency=data_frequency)
            open_price = float(row.open)
            high_price = float(row.high)
            low_price = float(row.low)

            if not state.get("breakeven_armed", False) and max(open_price, high_price) >= trigger_price:
                state["breakeven_armed"] = True

            if not state.get("breakeven_armed", False):
                continue

            if open_price <= stop_price:
                return ExitDecision("breakeven_stop", open_price, trade_time, {"breakeven_stop_price": stop_price})
            if low_price <= stop_price:
                return ExitDecision("breakeven_stop", stop_price, trade_time, {"breakeven_stop_price": stop_price})

        return None


class TimeExitPolicy(BaseExitPolicy):
    """达到最大持有交易日后强制退出。"""

    def __init__(self, *, max_trade_days: int):
        if max_trade_days <= 0:
            raise ValueError("max_trade_days must be positive")
        self.max_trade_days = max_trade_days

    def initialize_position(self, position: StockPosition) -> None:
        state = _risk_state(position)
        state["trade_days_held"] = 0
        state["last_time_stop_eval"] = None

    def evaluate(
        self,
        position: StockPosition,
        trade_date: datetime,
        market_frame: pd.DataFrame | None,
        *,
        data_frequency: str,
        history_frame: pd.DataFrame | None = None,
    ) -> ExitDecision | None:
        state = _risk_state(position)
        current_date = pd.Timestamp(trade_date).normalize()
        last_eval = state.get("last_time_stop_eval")
        if last_eval is None or pd.Timestamp(last_eval).normalize() != current_date:
            state["trade_days_held"] = int(state.get("trade_days_held", 0)) + 1
            state["last_time_stop_eval"] = trade_date

        if int(state.get("trade_days_held", 0)) < self.max_trade_days:
            return None

        first_point = _first_trade_point(market_frame, trade_date, data_frequency=data_frequency)
        if first_point is None:
            return None
        trade_price, trade_time = first_point
        return ExitDecision(
            reason="time_stop",
            trade_price=trade_price,
            trade_time=trade_time,
            metadata={"trade_days_held": int(state.get("trade_days_held", 0))},
        )


class CloseBelowMaExitPolicy(BaseExitPolicy):
    """收盘价跌破均线后，下一交易日执行卖出。"""

    def __init__(
        self,
        *,
        ma_window: int = 5,
        price_mode: str = "qfq",
        price_field: str = "close",
    ):
        if ma_window <= 1:
            raise ValueError("ma_window must be greater than 1")
        self.ma_window = ma_window
        self.price_mode = price_mode
        self.price_field = price_field

    def data_requirement(self) -> ExitDataRequirement:
        return ExitDataRequirement(
            stage=EXIT_STAGE_CLOSE_CONFIRMED,
            history_frequency="daily",
            history_lookback=self.ma_window,
            history_fields=(self.price_field,),
            price_mode=self.price_mode,
        )

    def evaluate(
        self,
        position: StockPosition,
        trade_date: datetime,
        market_frame: pd.DataFrame | None,
        *,
        data_frequency: str,
        history_frame: pd.DataFrame | None = None,
    ) -> ExitDecision | None:
        if history_frame is None or history_frame.empty:
            return None
        if self.price_field not in history_frame.columns:
            return None

        frame = history_frame.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame[frame["trade_date"] <= pd.Timestamp(trade_date)].sort_values("trade_date").reset_index(drop=True)
        if len(frame) < self.ma_window:
            return None

        latest_row = frame.iloc[-1]
        close_price = float(latest_row[self.price_field])
        ma_value = float(frame[self.price_field].tail(self.ma_window).mean())
        if close_price >= ma_value:
            return None

        decision_time = datetime.combine(trade_date.date(), time(15, 0))
        return ExitDecision(
            reason=f"close_below_ma{self.ma_window}",
            trade_price=close_price,
            trade_time=decision_time,
            metadata={
                "ma_window": self.ma_window,
                "close_price": close_price,
                "ma_value": ma_value,
                "price_mode": self.price_mode,
            },
        )
