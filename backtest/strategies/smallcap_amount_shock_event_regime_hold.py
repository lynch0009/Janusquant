from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Sequence

import numpy as np
import pandas as pd

from backtest.portfolio.positions import StockPosition
from backtest.risk.exits import (
    BaseExitPolicy,
    ExitDataRequirement,
    ExitDecision,
    EXIT_STAGE_CLOSE_CONFIRMED,
)
from backtest.strategies.models import DailyCandidate, IndexSlotRebalanceIntent
from backtest.strategies.smallcap_amount_shock_event import (
    EVENT_SELECTION_SORTS,
    SmallCapAmountShockEventStrategy,
)
from backtest.utils.datetime_utils import to_pydatetime


INDEX_SLOT_VIRTUAL_UNIT = 0.01


@dataclass(frozen=True)
class RegimeHoldExitConfig:
    post_hold_trend_start_days: int = 20
    stock_ma_exit_window: int = 10
    no_new_high_exit_days: int = 5
    price_mode: str = "qfq"


class RegimeHoldTrendExitPolicy(BaseExitPolicy):
    """事件 regime 持仓策略的收盘确认退出。

    退出优先级：
    1. 小票指数风险关闭，所有持仓退出。
    2. 持仓满 post_hold_trend_start_days 后，收盘跌破 MA。
    3. 持仓满 post_hold_trend_start_days 后，连续 N 日未创新高。
    """

    def __init__(
        self,
        *,
        regime_frame: pd.DataFrame | None = None,
        strategy: "SmallCapAmountShockEventRegimeHoldStrategy | None" = None,
        config: RegimeHoldExitConfig | None = None,
    ) -> None:
        self.config = config or RegimeHoldExitConfig()
        self.strategy = strategy
        self.regime_risk_off_dates = self._extract_risk_off_dates(regime_frame if regime_frame is not None else pd.DataFrame())

    @staticmethod
    def _extract_risk_off_dates(regime_frame: pd.DataFrame) -> frozenset[pd.Timestamp]:
        if regime_frame.empty or "trade_date" not in regime_frame.columns:
            return frozenset()
        frame = regime_frame.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        risk_col = "regime_force_exit" if "regime_force_exit" in frame.columns else "index_risk_off"
        if risk_col not in frame.columns:
            return frozenset()
        mask = frame[risk_col].fillna(False).astype(bool)
        return frozenset(frame.loc[mask, "trade_date"].dropna().tolist())

    def data_requirement(self) -> ExitDataRequirement:
        lookback = max(
            self.config.stock_ma_exit_window,
            self.config.post_hold_trend_start_days + self.config.no_new_high_exit_days + 5,
        )
        return ExitDataRequirement(
            stage=EXIT_STAGE_CLOSE_CONFIRMED,
            history_frequency="daily",
            history_lookback=lookback,
            history_fields=("close", "high"),
            price_mode=self.config.price_mode,
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
        current_date = pd.Timestamp(trade_date).normalize()
        risk_off_dates = self.regime_risk_off_dates
        if self.strategy is not None:
            risk_off_dates = self.strategy.regime_force_exit_dates
        if current_date in risk_off_dates:
            return ExitDecision(
                reason="index_ma5_cross_below_ma60_regime_off",
                trade_time=datetime.combine(trade_date.date(), time(15, 0)),
                metadata={"regime_exit": True},
            )
        if self.strategy is not None and position.code == self.strategy.index_code:
            return None

        if history_frame is None or history_frame.empty:
            return None
        required = {"trade_date", "close", "high"}
        if not required.issubset(history_frame.columns):
            return None

        frame = history_frame.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        entry_date = pd.Timestamp(position.entry_trade_date).normalize()
        frame = frame[(frame["trade_date"] >= entry_date) & (frame["trade_date"] <= current_date)].copy()
        frame = frame.sort_values("trade_date").dropna(subset=["close", "high"]).reset_index(drop=True)
        if frame.empty:
            return None

        holding_trade_days = len(frame)
        if holding_trade_days < self.config.post_hold_trend_start_days:
            return None

        latest = frame.iloc[-1]
        close_price = float(latest["close"])
        if len(frame) >= self.config.stock_ma_exit_window:
            ma_value = float(frame["close"].tail(self.config.stock_ma_exit_window).mean())
            if close_price < ma_value:
                return ExitDecision(
                    reason=f"post20_close_below_ma{self.config.stock_ma_exit_window}",
                    trade_time=datetime.combine(trade_date.date(), time(15, 0)),
                    metadata={
                        "holding_trade_days": int(holding_trade_days),
                        "ma_window": int(self.config.stock_ma_exit_window),
                        "close_price": close_price,
                        "ma_value": ma_value,
                    },
                )

        post_start_index = self.config.post_hold_trend_start_days - 1
        post_start = frame.iloc[post_start_index:].copy()
        if len(post_start) < self.config.no_new_high_exit_days:
            return None
        highs = pd.to_numeric(frame["high"], errors="coerce")
        # 只有严格突破此前最高价才算创新高，相同高点不能重置退出计时。
        prior_cum_high = highs.cummax().shift(1)
        new_high = highs > prior_cum_high
        post_new_high = new_high.iloc[post_start_index:]
        if not bool(post_new_high.tail(self.config.no_new_high_exit_days).any()):
            return ExitDecision(
                reason=f"post20_no_new_high_{self.config.no_new_high_exit_days}d",
                trade_time=datetime.combine(trade_date.date(), time(15, 0)),
                metadata={
                    "holding_trade_days": int(holding_trade_days),
                    "no_new_high_days": int(self.config.no_new_high_exit_days),
                    "highest_high_since_entry": float(highs.max()),
                    "latest_high": float(latest["high"]),
                    "close_price": close_price,
                },
            )
        return None


class SmallCapAmountShockEventRegimeHoldStrategy(SmallCapAmountShockEventStrategy):
    """事件触发的小票趋势持仓实验策略 V1。"""

    def __init__(
        self,
        *,
        max_positions: int = 10,
        scheduled_hold_days: int = 10000,
        index_code: str = "sz.399303",
        index_fast_ma: int = 5,
        index_slow_ma: int = 60,
        weekly_fill_weekday: int = 4,
        enable_weekly_fill: bool = True,
        post_hold_trend_start_days: int = 20,
        stock_ma_exit_window: int = 10,
        no_new_high_exit_days: int = 5,
        **kwargs,
    ) -> None:
        if kwargs.get("selection_sort", "composite_zscore") not in EVENT_SELECTION_SORTS:
            raise ValueError("selection_sort is not supported.")
        kwargs.setdefault("selection_sort", "composite_zscore")
        super().__init__(**kwargs)
        self.max_positions = int(max_positions)
        self.scheduled_hold_days = int(scheduled_hold_days)
        self.index_code = str(index_code)
        self.index_fast_ma = int(index_fast_ma)
        self.index_slow_ma = int(index_slow_ma)
        self.weekly_fill_weekday = int(weekly_fill_weekday)
        self.enable_weekly_fill = bool(enable_weekly_fill)
        self.post_hold_trend_start_days = int(post_hold_trend_start_days)
        self.stock_ma_exit_window = int(stock_ma_exit_window)
        self.no_new_high_exit_days = int(no_new_high_exit_days)
        self.regime_state_frame = pd.DataFrame()
        self.weekly_fill_signal_frame = pd.DataFrame()
        self._regime_table: dict[pd.Timestamp, dict] = {}
        self._event_signal_table: dict[datetime, pd.DataFrame] = {}
        self._weekly_fill_dates_set: set[pd.Timestamp] = set()
        self._entry_signal_date_by_execution_date: dict[pd.Timestamp, pd.Timestamp] = {}
        self._event_trigger_dates: set[pd.Timestamp] = set()
        self.regime_force_exit_dates: frozenset[pd.Timestamp] = frozenset()
        self._validate_regime_hold_parameters()

    def _validate_regime_hold_parameters(self) -> None:
        if self.max_positions < 1:
            raise ValueError("max_positions must be >= 1")
        if self.scheduled_hold_days < self.post_hold_trend_start_days:
            raise ValueError("scheduled_hold_days must be >= post_hold_trend_start_days")
        if self.index_fast_ma < 1 or self.index_slow_ma <= self.index_fast_ma:
            raise ValueError("index_slow_ma must be greater than index_fast_ma")
        if self.weekly_fill_weekday < 0 or self.weekly_fill_weekday > 4:
            raise ValueError("weekly_fill_weekday must be within 0..4")
        if self.post_hold_trend_start_days < 1:
            raise ValueError("post_hold_trend_start_days must be >= 1")
        if self.stock_ma_exit_window < 2:
            raise ValueError("stock_ma_exit_window must be >= 2")
        if self.no_new_high_exit_days < 1:
            raise ValueError("no_new_high_exit_days must be >= 1")

    def exit_policy(self) -> RegimeHoldTrendExitPolicy:
        return RegimeHoldTrendExitPolicy(
            strategy=self,
            config=RegimeHoldExitConfig(
                post_hold_trend_start_days=self.post_hold_trend_start_days,
                stock_ma_exit_window=self.stock_ma_exit_window,
                no_new_high_exit_days=self.no_new_high_exit_days,
                price_mode=self.signal_price_mode,
            ),
        )

    def _build_index_regime_features(self, data_portal, trade_dates: Sequence[datetime], *, research_store=None) -> pd.DataFrame:
        if not trade_dates:
            return pd.DataFrame()
        signal_start = pd.Timestamp(trade_dates[0]).to_pydatetime()
        signal_end = pd.Timestamp(trade_dates[-1]).to_pydatetime()
        warmup_start = signal_start - pd.Timedelta(days=max(self.index_slow_ma * 3, self.index_slow_ma + 30))
        fields = ["code", "trade_date", "close"]
        if research_store is not None:
            index_history = research_store.load_daily_history(
                warmup_start,
                signal_end + pd.Timedelta(days=1),
                codes=[self.index_code],
                fields=fields,
                include_stopped=False,
                price_mode="raw",
                batch_size=1000,
            )
        else:
            index_history = data_portal.get_daily_history(
                warmup_start,
                signal_end + pd.Timedelta(days=1),
                codes=[self.index_code],
                fields=fields,
                include_stopped=False,
                price_mode="raw",
                batch_size=1000,
            )
        if index_history.empty:
            raise ValueError(f"No index daily history found for {self.index_code}.")
        index_history = index_history.copy()
        index_history["trade_date"] = pd.to_datetime(index_history["trade_date"]).dt.normalize()
        index_history["index_close"] = pd.to_numeric(index_history["close"], errors="coerce")
        index_history = index_history.dropna(subset=["index_close"]).sort_values("trade_date").reset_index(drop=True)
        if index_history.empty:
            raise ValueError(f"No valid index close history found for {self.index_code}.")
        index_history["index_ma_fast"] = index_history["index_close"].rolling(
            self.index_fast_ma,
            min_periods=self.index_fast_ma,
        ).mean()
        index_history["index_ma_slow"] = index_history["index_close"].rolling(
            self.index_slow_ma,
            min_periods=self.index_slow_ma,
        ).mean()
        index_history["index_risk_off"] = index_history["index_ma_fast"] < index_history["index_ma_slow"]
        risk_off_values = index_history["index_risk_off"].to_numpy(dtype=bool)
        prev_risk_off = np.concatenate(([False], risk_off_values[:-1]))
        index_history["index_risk_cross_down"] = risk_off_values & ~prev_risk_off
        trade_date_frame = pd.DataFrame({"trade_date": pd.to_datetime(trade_dates).normalize()})
        result = trade_date_frame.merge(
            index_history[
                [
                    "trade_date",
                    "index_close",
                    "index_ma_fast",
                    "index_ma_slow",
                    "index_risk_off",
                    "index_risk_cross_down",
                ]
            ],
            on="trade_date",
            how="left",
        )
        if result["index_close"].isna().all():
            raise ValueError(f"Index {self.index_code} has no overlap with requested trade dates.")
        required_columns = ["index_close", "index_ma_fast", "index_ma_slow", "index_risk_off", "index_risk_cross_down"]
        missing_mask = result[required_columns].isna().any(axis=1)
        if missing_mask.any():
            missing_dates = result.loc[missing_mask, "trade_date"].dt.strftime("%Y-%m-%d").tolist()
            raise ValueError(
                f"Index regime data or moving-average warmup is incomplete for {self.index_code}: "
                + ", ".join(missing_dates[:20])
            )
        result["index_risk_off"] = result["index_risk_off"].astype(bool)
        result["index_risk_cross_down"] = result["index_risk_cross_down"].astype(bool)
        return result

    @staticmethod
    def _weekly_fill_dates(trade_dates: Sequence[datetime], weekday: int) -> set[pd.Timestamp]:
        # 只使用明确的星期锚点，不能把截断区间的最后一天误当成完整周结束日。
        dates = pd.DatetimeIndex(pd.to_datetime(trade_dates)).normalize()
        return set(dates[dates.weekday == weekday])

    def _build_regime_state_frame(self, daily_windows: pd.DataFrame, index_frame: pd.DataFrame) -> pd.DataFrame:
        windows = daily_windows.copy()
        windows["trade_date"] = pd.to_datetime(windows["trade_date"]).dt.normalize()
        windows["event_trigger"] = pd.to_numeric(windows["matched_rule_count"], errors="coerce").fillna(0) > 0
        regime = windows.merge(index_frame, on="trade_date", how="left")
        rows = []
        active = False
        regime_open_date: pd.Timestamp | None = None
        for row in regime.sort_values("trade_date").itertuples(index=False):
            trade_date = pd.Timestamp(row.trade_date).normalize()
            event_trigger = bool(getattr(row, "event_trigger"))
            index_risk_off = bool(getattr(row, "index_risk_off", False))
            index_risk_cross_down = bool(getattr(row, "index_risk_cross_down", False))
            transition = ""
            regime_force_exit = False
            # 事件信号负责开仓和延续，不受当前指数均线状态限制；
            # 指数只在持仓状态中出现新的下穿、且同日没有新事件时触发清仓。
            if active and index_risk_cross_down and not event_trigger:
                active = False
                regime_open_date = None
                regime_force_exit = True
                transition = "close_index_risk_cross_down"
            if (not active) and event_trigger:
                active = True
                regime_open_date = trade_date
                transition = "open_event_trigger"
            elif active and event_trigger:
                transition = "event_refresh"

            rows.append(
                {
                    "trade_date": trade_date,
                    "regime_state": "active" if active else "inactive",
                    "regime_active": bool(active),
                    "regime_open_date": regime_open_date,
                    "event_trigger": event_trigger,
                    "regime_force_exit": regime_force_exit,
                    "transition": transition,
                    "index_code": self.index_code,
                    "index_close": getattr(row, "index_close", np.nan),
                    "index_ma5": getattr(row, "index_ma_fast", np.nan),
                    "index_ma60": getattr(row, "index_ma_slow", np.nan),
                    "index_risk_off": index_risk_off,
                    "index_risk_cross_down": index_risk_cross_down,
                    "matched_rules": getattr(row, "matched_rules", ""),
                    "matched_rule_count": getattr(row, "matched_rule_count", 0),
                    "signal_count": getattr(row, "signal_count", 0),
                    "pool_ret_20d_equal": getattr(row, "pool_ret_20d_equal", np.nan),
                    "pool_up_ratio_1d": getattr(row, "pool_up_ratio_1d", np.nan),
                }
            )
        result = pd.DataFrame(rows)
        if not result.empty:
            result["regime_open_date"] = pd.to_datetime(result["regime_open_date"], errors="coerce")
        return result

    def _regime_meta(self, signal_date: datetime) -> dict:
        key = pd.Timestamp(signal_date).normalize()
        return dict(self._regime_table.get(key, {}))

    def _target_exit_trade_date(self) -> datetime:
        return datetime.max.replace(hour=0, minute=0, second=0, microsecond=0)

    def _add_common_entry_metadata(self, row, *, entry_source: str, entry_reason: str, signal_date: datetime) -> dict:
        regime = self._regime_meta(signal_date)
        metadata = {
            "entry_source": entry_source,
            "entry_reason": entry_reason,
            "regime_state": regime.get("regime_state"),
            "regime_open_date": regime.get("regime_open_date"),
            "index_ma5": regime.get("index_ma5"),
            "index_ma60": regime.get("index_ma60"),
            "index_risk_off": regime.get("index_risk_off"),
            "index_risk_cross_down": regime.get("index_risk_cross_down"),
            "target_exit_trade_date": self._target_exit_trade_date(),
            "post_hold_trend_start_days": self.post_hold_trend_start_days,
            "stock_ma_exit_window": self.stock_ma_exit_window,
            "no_new_high_exit_days": self.no_new_high_exit_days,
        }
        for key in (
            self.cap_field,
            "close",
            "raw_close",
            "amount_expand",
            "ret_10d",
            "research_ret_10d",
            "selection_sort_score",
            self.amount_fast_col,
            self.amount_slow_col,
        ):
            value = getattr(row, key, None)
            if pd.notna(value):
                metadata[key] = float(value)
        return metadata

    def _build_event_candidates(self, signal_date: datetime, frame: pd.DataFrame, held_codes: set[str], available_slots: int) -> list[DailyCandidate]:
        candidates: list[DailyCandidate] = []
        for row in frame.itertuples(index=False):
            code = str(row.code)
            if code in held_codes:
                continue
            metadata = self._add_common_entry_metadata(
                row,
                entry_source="event_signal",
                entry_reason="amount_shock_event_regime_event_entry",
                signal_date=signal_date,
            )
            for key in (
                "matched_rules",
                "selection_sort",
            ):
                value = getattr(row, key, None)
                if value is not None:
                    metadata[key] = str(value)
            for key in (
                "matched_rule_count",
                "event_signal_rank",
                "amount_group",
                "ret_group",
                "signal_count",
                "ret_top_rank",
                "ret_top_keep_count",
            ):
                value = getattr(row, key, None)
                if pd.notna(value):
                    metadata[key] = int(value)
            candidates.append(
                DailyCandidate(
                    signal_date=signal_date,
                    code=code,
                    score=float(getattr(row, "selection_sort_score", 0.0)),
                    hold_days=self.scheduled_hold_days,
                    metadata=metadata,
                )
            )
            if len(candidates) >= available_slots:
                break
        return candidates

    def _portfolio_equity_from_snapshot(self, portfolio_snapshot: dict) -> float:
        cash = float(portfolio_snapshot.get("cash", 0.0) or 0.0)
        positions = portfolio_snapshot.get("positions", {}) or {}
        market_value = sum(float(getattr(position, "market_value", 0.0) or 0.0) for position in positions.values())
        return max(cash + market_value, 0.0)

    def _event_stock_count_from_snapshot(self, portfolio_snapshot: dict) -> int:
        positions = portfolio_snapshot.get("positions", {}) or {}
        return sum(1 for code in positions if str(code) != self.index_code)

    def _scheduled_event_entry_count(self, scheduled_candidates: Sequence[DailyCandidate], portfolio_snapshot: dict) -> int:
        positions = portfolio_snapshot.get("positions", {}) or {}
        existing_codes = set(positions.keys())
        seen_codes: set[str] = set()
        capacity = max(self.max_positions - self._event_stock_count_from_snapshot(portfolio_snapshot), 0)
        for candidate in scheduled_candidates:
            code = str(candidate.code)
            if code == self.index_code or code in existing_codes or code in seen_codes:
                continue
            seen_codes.add(code)
            if len(seen_codes) >= capacity:
                break
        return len(seen_codes)

    def _record_weekly_index_slot_intent(
        self,
        *,
        trade_date: datetime,
        target_market_value: float,
        target_index_slots: int,
        event_stock_count: int,
        scheduled_event_entry_count: int,
        reason: str,
        portfolio_snapshot: dict,
    ) -> None:
        row = {
            "trade_date": pd.Timestamp(trade_date).normalize(),
            "code": self.index_code,
            "entry_source": "weekly_index_slot_fill",
            "entry_reason": "amount_shock_event_regime_index_slot_fill",
            "selection_sort": "index_slot_fill",
            "selection_sort_score": float(target_index_slots),
            "fill_rank": 1,
            "index_virtual_unit": INDEX_SLOT_VIRTUAL_UNIT,
            "target_index_slots": int(target_index_slots),
            "event_stock_count": int(event_stock_count),
            "scheduled_event_entry_count": int(scheduled_event_entry_count),
            "target_market_value": float(target_market_value),
            "portfolio_cash": float(portfolio_snapshot.get("cash", 0.0) or 0.0),
            "reason": reason,
        }
        self.weekly_fill_signal_frame = pd.concat(
            [self.weekly_fill_signal_frame, pd.DataFrame([row])],
            ignore_index=True,
        )

    def prepare_before_entries(
        self,
        trade_date: datetime,
        portfolio_snapshot: dict,
        scheduled_candidates: Sequence[DailyCandidate],
    ) -> list[IndexSlotRebalanceIntent]:
        execution_date = pd.Timestamp(trade_date).normalize()
        signal_date = self._entry_signal_date_by_execution_date.get(execution_date)
        if signal_date is None:
            return []
        regime = self._regime_table.get(signal_date)
        if not regime or not bool(regime.get("regime_active", False)) or bool(regime.get("regime_force_exit", False)):
            return []

        scheduled_event_count = self._scheduled_event_entry_count(scheduled_candidates, portfolio_snapshot)
        is_weekly_fill_date = signal_date in self._weekly_fill_dates_set
        should_release_for_events = scheduled_event_count > 0
        should_fill_weekly = self.enable_weekly_fill and is_weekly_fill_date and not should_release_for_events
        if not should_release_for_events and not should_fill_weekly:
            return []

        current_event_stock_count = self._event_stock_count_from_snapshot(portfolio_snapshot)
        target_event_stock_count = min(current_event_stock_count + scheduled_event_count, self.max_positions)
        target_index_slots = max(self.max_positions - target_event_stock_count, 0)
        positions = portfolio_snapshot.get("positions", {}) or {}
        has_index_position = self.index_code in positions
        if should_release_for_events and not has_index_position:
            return []
        if target_index_slots <= 0 and not has_index_position:
            return []

        portfolio_equity = self._portfolio_equity_from_snapshot(portfolio_snapshot)
        target_market_value = portfolio_equity * (target_index_slots / self.max_positions) if self.max_positions > 0 else 0.0
        reason = "index_slot_release_for_event_entry" if should_release_for_events else "index_slot_weekly_fill"
        self._record_weekly_index_slot_intent(
            trade_date=trade_date,
            target_market_value=target_market_value,
            target_index_slots=target_index_slots,
            event_stock_count=current_event_stock_count,
            scheduled_event_entry_count=scheduled_event_count,
            reason=reason,
            portfolio_snapshot=portfolio_snapshot,
        )
        return [
            IndexSlotRebalanceIntent(
                signal_date=signal_date.to_pydatetime(),
                code=self.index_code,
                target_market_value=target_market_value,
                reason=reason,
                metadata={
                    "entry_source": "weekly_index_slot_fill",
                    "entry_reason": "amount_shock_event_regime_index_slot_fill",
                    "index_virtual_unit": INDEX_SLOT_VIRTUAL_UNIT,
                    "target_index_slots": int(target_index_slots),
                    "event_stock_count": int(current_event_stock_count),
                    "scheduled_event_entry_count": int(scheduled_event_count),
                    "weekly_fill_enabled": bool(self.enable_weekly_fill),
                    "index_slot_signal_date": signal_date.to_pydatetime(),
                },
            )
        ]

    def prepare(self, data_portal, trade_dates: Sequence[datetime], *, research_store=None) -> None:
        self.reset_signal_table()
        self.event_signal_frame = pd.DataFrame()
        self.event_daily_window_features = pd.DataFrame()
        self.regime_state_frame = pd.DataFrame()
        self.weekly_fill_signal_frame = pd.DataFrame()
        self._event_signal_table = {}
        self._regime_table = {}
        self._weekly_fill_dates_set = set()
        self._entry_signal_date_by_execution_date = {}
        self._event_trigger_dates = set()
        self.regime_force_exit_dates = frozenset()
        if not trade_dates:
            return

        signal_start = pd.Timestamp(trade_dates[0]).to_pydatetime()
        signal_end = pd.Timestamp(trade_dates[-1]).to_pydatetime()

        if self.precomputed_candidate_indicator_frame is not None:
            candidate_indicator = self._slice_precomputed_candidate_indicator(trade_dates)
        else:
            candidate_pool_frame, full_trade_calendar = self._prepare_cached_candidate_pool_frame(data_portal, trade_dates)
            if candidate_pool_frame.empty or full_trade_calendar.empty:
                return
            warmup_start = self._warmup_trade_dates(full_trade_calendar, signal_start)
            indicator_frame = self._prepare_indicator_frame(
                data_portal,
                codes=sorted(candidate_pool_frame["code"].unique().tolist()),
                warmup_start=warmup_start,
                signal_end=signal_end,
                research_store=research_store,
            )
            if indicator_frame.empty:
                return
            candidate_indicator = candidate_pool_frame.merge(indicator_frame, on=["code", "trade_date"], how="inner")
        if candidate_indicator.empty:
            return

        candidate_indicator = candidate_indicator.copy()
        candidate_indicator["trade_date"] = pd.to_datetime(candidate_indicator["trade_date"]).dt.normalize()
        candidate_indicator = self._ensure_raw_close_column(candidate_indicator)
        pool_windows = self._build_pool_window_features(candidate_indicator)

        signal_frame = candidate_indicator.copy()
        signal_frame = signal_frame[signal_frame["isST"].fillna(False) == False].copy()
        if self.st_lookback_trade_days:
            signal_frame = signal_frame[signal_frame["is_st_lookback_flag"].fillna(False) == False].copy()
        if self.min_signal_close_price is not None:
            signal_frame = signal_frame[pd.to_numeric(signal_frame["raw_close"], errors="coerce") > self.min_signal_close_price].copy()
        required_factor_columns = ["amount_expand", "ret_10d", "research_ret_10d"]
        signal_frame = signal_frame.dropna(subset=required_factor_columns).copy()
        if not signal_frame.empty:
            signal_frame = self._assign_groups(signal_frame)
            signal_frame = self._apply_signal_filters(signal_frame)

        signal_counts = (
            signal_frame.groupby("trade_date").size().rename("signal_count").reset_index()
            if not signal_frame.empty
            else pd.DataFrame(columns=["trade_date", "signal_count"])
        )
        daily_windows = pool_windows.merge(signal_counts, on="trade_date", how="left")
        daily_windows["signal_count"] = daily_windows["signal_count"].fillna(0).astype(int)
        daily_windows = self._annotate_window_rules(daily_windows)
        index_frame = self._build_index_regime_features(data_portal, trade_dates, research_store=research_store)
        self.regime_state_frame = self._build_regime_state_frame(daily_windows, index_frame)
        self.regime_force_exit_dates = RegimeHoldTrendExitPolicy._extract_risk_off_dates(self.regime_state_frame)
        self._regime_table = {
            pd.Timestamp(row.trade_date).normalize(): row._asdict()
            for row in self.regime_state_frame.itertuples(index=False)
        }
        self.event_daily_window_features = daily_windows.copy()

        event_dates = set(
            pd.to_datetime(
                daily_windows.loc[pd.to_numeric(daily_windows["matched_rule_count"], errors="coerce").fillna(0) > 0, "trade_date"]
            ).dt.normalize()
        )
        self._event_trigger_dates = set(event_dates)
        if not signal_frame.empty:
            rule_columns = ["trade_date", "matched_rules", "matched_rule_count"]
            rule_columns.extend([column for column in daily_windows.columns if column.startswith("rule_")])
            window_feature_columns = [
                "pool_constituent_count",
                "pool_ret_5d_equal",
                "pool_ret_20d_equal",
                "pool_up_ratio_1d",
                "signal_count",
            ]
            rule_columns.extend([column for column in window_feature_columns if column in daily_windows.columns])
            rule_columns = list(dict.fromkeys(rule_columns))
            event_frame = signal_frame.merge(daily_windows[rule_columns], on="trade_date", how="left")
            event_frame = event_frame[pd.to_numeric(event_frame["matched_rule_count"], errors="coerce").fillna(0) > 0].copy()
            if not event_frame.empty:
                event_frame = self._apply_ret_top_pct_filter(event_frame)
                event_frame = self._sort_event_signals(event_frame).copy()
                event_frame["event_signal_rank"] = event_frame.groupby("trade_date").cumcount() + 1
                event_frame["ret_top_pct"] = self.ret_top_pct
                event_frame["selection_sort"] = self.event_selection_sort
                event_frame["entry_source"] = "event_signal"
                self.event_signal_frame = event_frame.reset_index(drop=True)
                self._event_signal_table = self.group_signal_table(self.event_signal_frame, limit=None)

        self._weekly_fill_dates_set = self._weekly_fill_dates(trade_dates, self.weekly_fill_weekday)
        normalized_trade_dates = pd.DatetimeIndex(pd.to_datetime(trade_dates)).normalize()
        self._entry_signal_date_by_execution_date = {
            execution_date: signal_date
            for signal_date, execution_date in zip(normalized_trade_dates[:-1], normalized_trade_dates[1:])
        }

        combined = []
        if not self.event_signal_frame.empty:
            combined.append(self.event_signal_frame)
        self.set_signal_table(self.group_signal_table(pd.concat(combined, ignore_index=True), limit=None) if combined else {})

    def generate_candidates(
        self,
        signal_date: datetime,
        feature_slice: pd.DataFrame,
        portfolio_snapshot: dict,
    ) -> list[DailyCandidate]:
        regime = self._regime_meta(signal_date)
        if not regime or not bool(regime.get("regime_active", False)) or bool(regime.get("regime_force_exit", False)):
            return []

        held_codes = set(portfolio_snapshot.get("held_codes", []))
        event_held_codes = {code for code in held_codes if str(code) != self.index_code}
        available_slots = max(self.max_positions - len(event_held_codes), 0)
        if available_slots <= 0:
            return []

        event_frame = self._event_signal_table.get(to_pydatetime(signal_date))
        if event_frame is not None and not event_frame.empty:
            return self._build_event_candidates(signal_date, event_frame, held_codes, min(available_slots, self.top_k))
        return []
