"""信号驱动回测引擎。

负责把策略、执行模型、账户、公司行为和风控规则串起来，
按交易日推进整个回测流程。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any
import uuid

import pandas as pd

from backtest.data import DuckDBDataPortal, ResearchDailyHistoryStore
from backtest.execution.config import EngineConfig
from backtest.execution.executors import (
    BaseExecutionModel,
    WindowFirstBarExecutor,
    build_exit_trade_from_price,
)
from backtest.portfolio import (
    BacktestResult,
    BenchmarkPoint,
    FixedFractionSizer,
    PortfolioLedger,
    StockOrder,
)
from backtest.portfolio.sizing import BasePositionSizer
from backtest.risk import (
    BaseExitPolicy,
    EXIT_STAGE_CLOSE_CONFIRMED,
    EXIT_STAGE_INTRADAY,
)
from backtest.strategies import BaseSelectionStrategy, DailyCandidate, IndexSlotRebalanceIntent
from backtest.utils.datetime_utils import to_pydatetime
from backtest.utils.frame_utils import first_sorted_row
from backtest.utils.log import log_event
from backtest.utils.metadata import copy_metadata, merge_metadata
from backtest.utils.price_limits import decide_daily_sell_fill
from .engine_accounting import EngineAccountingMixin
from .engine_flow import EngineFlowMixin
from .engine_target_portfolio import EngineTargetPortfolioMixin
from .engine_types import PendingRiskExit


class SignalDrivenBacktestEngine(
    EngineTargetPortfolioMixin,
    EngineAccountingMixin,
    EngineFlowMixin,
):
    """按交易日驱动组合状态变化的主引擎。"""

    def __init__(
        self,
        db_client,
        strategy: BaseSelectionStrategy,
        *,
        execution_model: BaseExecutionModel | None = None,
        config: EngineConfig | None = None,
        calendar_code: str = "sh.000001",
        position_sizer: BasePositionSizer | None = None,
        exit_policy: BaseExitPolicy | None = None,
        data_portal: DuckDBDataPortal | None = None,
    ):
        self.strategy = strategy
        self.execution_model = execution_model or WindowFirstBarExecutor()
        self.config = config or EngineConfig()
        self.data_portal = data_portal or DuckDBDataPortal(db_client, calendar_code=calendar_code)
        self.benchmark_code = self.config.benchmark_code
        self.position_sizer = position_sizer or FixedFractionSizer()
        self.exit_policy = exit_policy
        self._exit_policy_cache: dict[str, list[BaseExitPolicy]] = {}
        self._uses_target_portfolio_rebalance = self.strategy.uses_target_portfolio_rebalance()

    def _build_benchmark_curve(self, trade_dates: list[datetime]) -> list[BenchmarkPoint]:
        """读取默认基准指数，并对齐到本次回测交易日历。"""

        if not trade_dates:
            return []

        benchmark_history = self.data_portal.get_daily_history(
            trade_dates[0],
            trade_dates[-1] + timedelta(days=1),
            codes=[self.benchmark_code],
            fields=["code", "trade_date", "close"],
            include_stopped=True,
            price_mode="raw",
        )
        if benchmark_history.empty:
            return []

        frame = benchmark_history.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["trade_date", "close"]).sort_values("trade_date")
        if frame.empty:
            return []

        aligned = (
            frame[["trade_date", "code", "close"]]
            .drop_duplicates(subset="trade_date", keep="last")
            .set_index("trade_date")
            .reindex(pd.to_datetime(trade_dates))
        )
        aligned.index.name = "trade_date"
        aligned["code"] = aligned["code"].fillna(self.benchmark_code)
        aligned["close"] = aligned["close"].ffill()
        aligned = aligned.dropna(subset=["close"]).reset_index()
        if aligned.empty:
            return []

        return [
            BenchmarkPoint(
                trade_date=to_pydatetime(row.trade_date),
                code=str(row.code),
                close=float(row.close),
            )
            for row in aligned.itertuples(index=False)
        ]

    def _should_log_progress(self) -> bool:
        return bool(getattr(self.config, "progress_logging", False))

    def _log_event(self, event: str, **fields) -> None:
        if self._should_log_progress():
            log_event("info", event, **fields)

    def _exit_policies_for_stage(self, stage: str) -> list[BaseExitPolicy]:
        """取出某个阶段需要执行的退出规则。"""

        if self.exit_policy is None:
            return []
        if stage not in self._exit_policy_cache:
            self._exit_policy_cache[stage] = self.exit_policy.policies_for_stage(stage)
        return self._exit_policy_cache[stage]

    def _resolve_exit_trade_date(
        self,
        trade_dates: list[datetime],
        entry_index: int,
        hold_days: int,
    ) -> datetime:
        """根据持有交易日数推导目标退出日。"""

        target_index = min(entry_index + hold_days, len(trade_dates) - 1)
        return trade_dates[target_index]

    def _load_execution_frames(
        self,
        codes: list[str],
        trade_date: datetime,
        *,
        for_exit: bool,
    ) -> dict:
        """按执行模型需要的频率加载入场/出场行情切片。"""

        if not codes:
            return {}

        if self.execution_model.data_frequency == "minute":
            start_time = self.config.exit_start_time if for_exit else self.config.entry_start_time
            end_time = self.config.exit_end_time if for_exit else self.config.entry_end_time
            return self.data_portal.get_minute_bars_batch(
                codes,
                trade_date,
                start_time=start_time,
                end_time=end_time,
            )

        if self.execution_model.data_frequency == "daily":
            return self.data_portal.get_daily_bar_snapshot(codes, trade_date)

        raise ValueError(f"unsupported execution data frequency: {self.execution_model.data_frequency}")

    def _load_intraday_risk_frames(self, codes: list[str], trade_date: datetime) -> dict:
        """为日内风控规则加载行情窗口。"""

        if not codes:
            return {}

        if self.execution_model.data_frequency == "minute":
            return self.data_portal.get_minute_bars_batch(
                codes,
                trade_date,
                start_time=self.config.risk_start_time,
                end_time=self.config.risk_end_time,
            )
        if self.execution_model.data_frequency == "daily":
            return self.data_portal.get_daily_bar_snapshot(codes, trade_date)
        raise ValueError(f"unsupported execution data frequency: {self.execution_model.data_frequency}")

    def _schedule_candidates(
        self,
        pending_entries: dict[datetime, list[DailyCandidate]],
        trade_dates: list[datetime],
        current_index: int,
        candidates: list[DailyCandidate],
    ) -> None:
        """把信号挂到真正执行的交易日上。"""

        if not candidates:
            return

        if self.config.execute_on_next_trade_date:
            if current_index + 1 >= len(trade_dates):
                return
            execution_date = trade_dates[current_index + 1]
        else:
            execution_date = trade_dates[current_index]

        pending_entries[execution_date].extend(candidates)

    def _generate_and_schedule_candidates(
        self,
        ledger: PortfolioLedger,
        pending_entries: dict[datetime, list[DailyCandidate]],
        trade_dates: list[datetime],
        trade_index: int,
        trade_date: datetime,
    ) -> None:
        """读取当日特征，生成候选，并调度到执行日。"""

        feature_slice = self.data_portal.get_feature_slice(
            trade_date,
            fields=list(self.strategy.required_feature_fields()),
        )
        candidates = self.strategy.generate_candidates(trade_date, feature_slice, ledger.snapshot())
        self._log_event(
            "generate_candidates",
            trade_date=trade_date,
            candidate_count=len(candidates),
            candidate_codes=[candidate.code for candidate in candidates],
        )
        self._schedule_candidates(pending_entries, trade_dates, trade_index, candidates)

    def _build_close_confirmed_history_specs(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, dict[str, Any]]:
        """根据收盘确认类规则汇总所需历史数据规格。"""

        specs: dict[str, dict[str, Any]] = {}
        for policy in self._exit_policies_for_stage(EXIT_STAGE_CLOSE_CONFIRMED):
            requirement = policy.data_requirement()
            if requirement.history_frequency != "daily":
                continue
            spec = specs.setdefault(
                requirement.price_mode,
                {
                    "lookback": 0,
                    "fields": set(),
                    "start_date": start_date,
                    "end_date": end_date + timedelta(days=1),
                },
            )
            spec["lookback"] = max(spec["lookback"], requirement.history_lookback)
            spec["fields"].update(requirement.history_fields)

        for price_mode, spec in specs.items():
            lookback = int(spec["lookback"])
            extra_days = max(lookback * 3, lookback + 20)
            spec["start_date"] = start_date - timedelta(days=extra_days)
            spec["fields"] = tuple(sorted(spec["fields"]))
            self._log_event(
                "risk_history_spec_ready",
                stage=EXIT_STAGE_CLOSE_CONFIRMED,
                price_mode=price_mode,
                start_date=spec["start_date"],
                end_date=spec["end_date"],
                lookback=lookback,
                fields=list(spec["fields"]),
            )
        return specs

    def _load_close_confirmed_history_frame(
        self,
        code: str,
        trade_date: datetime,
        policy: BaseExitPolicy,
        history_specs: dict[str, dict[str, Any]],
        research_store: ResearchDailyHistoryStore,
    ) -> pd.DataFrame | None:
        """按代码和价格口径懒加载收盘确认类规则所需历史数据。"""

        requirement = policy.data_requirement()
        if requirement.history_frequency != "daily" or requirement.history_lookback <= 0:
            return None

        spec = history_specs.get(requirement.price_mode)
        if spec is None:
            return None

        frame = research_store.load_daily_history(
            spec["start_date"],
            spec["end_date"],
            codes=[code],
            fields=list(spec["fields"]),
            include_stopped=False,
            price_mode=requirement.price_mode,
            batch_size=1000,
        )
        if frame.empty:
            return None
        return frame[frame["trade_date"] <= pd.Timestamp(trade_date)].copy()

    def _has_pending_risk_exit(
        self,
        pending_risk_exits: dict[datetime, list[PendingRiskExit]],
        code: str,
    ) -> bool:
        """判断某只股票是否已经有待执行的风险退出。"""

        for exits in pending_risk_exits.values():
            for pending_exit in exits:
                if pending_exit.code == code:
                    return True
        return False

    def _schedule_pending_risk_exit(
        self,
        pending_risk_exits: dict[datetime, list[PendingRiskExit]],
        trade_dates: list[datetime],
        trade_index: int,
        position,
        decision,
        policy: BaseExitPolicy,
    ) -> None:
        """把收盘确认类退出挂到下一交易日执行。"""

        if trade_index + 1 >= len(trade_dates):
            return
        next_trade_date = trade_dates[trade_index + 1]
        pending_exit = PendingRiskExit(
            code=position.code,
            signal_date=position.signal_date,
            scheduled_trade_date=next_trade_date,
            reason=decision.reason,
            score=position.score,
            metadata=merge_metadata(position.metadata, decision.metadata),
            risk_rule_name=policy.__class__.__name__,
        )
        pending_risk_exits[next_trade_date].append(pending_exit)
        self._log_event(
            "pending_risk_exit_scheduled",
            signal_date=trade_dates[trade_index],
            scheduled_trade_date=next_trade_date,
            code=position.code,
            reason=decision.reason,
            risk_rule_name=policy.__class__.__name__,
        )

    def _process_pending_risk_exits(
        self,
        ledger: PortfolioLedger,
        pending_risk_exits: dict[datetime, list[PendingRiskExit]],
        trade_dates: list[datetime],
        trade_index: int,
        trade_date: datetime,
    ) -> None:
        """执行前一日收盘确认、今日需要卖出的风险退出。"""

        todays_pending = pending_risk_exits.pop(trade_date, [])
        if not todays_pending:
            return

        execution_frames = self._load_execution_frames(
            [item.code for item in todays_pending],
            trade_date,
            for_exit=True,
        )
        for pending_exit in todays_pending:
            position = ledger.positions.get(pending_exit.code)
            if position is None:
                continue

            frame = execution_frames.get(pending_exit.code)
            order_metadata = merge_metadata(position.metadata, pending_exit.metadata)
            virtual_unit = self._index_slot_virtual_unit(order_metadata, position=position)
            exit_order = StockOrder.create(
                code=pending_exit.code,
                side="SELL",
                signal_date=position.signal_date,
                scheduled_trade_date=trade_date,
                execution_model=self.execution_model.data_frequency,
                reason=pending_exit.reason,
                requested_quantity=position.quantity,
                score=position.score,
                metadata=order_metadata,
                risk_rule_name=pending_exit.risk_rule_name,
            )

            if frame is None or frame.empty:
                exit_order.mark_skipped("no_execution_data")
                ledger.register_order(exit_order)
                if trade_index + 1 < len(trade_dates):
                    retry_exit = replace(
                        pending_exit,
                        scheduled_trade_date=trade_dates[trade_index + 1],
                    )
                    pending_risk_exits[retry_exit.scheduled_trade_date].append(retry_exit)
                self._log_event(
                    "pending_risk_exit_delayed",
                    trade_date=trade_date,
                    code=pending_exit.code,
                    reason=pending_exit.reason,
                    skip_reason="no_execution_data",
                )
                continue

            frame = self._scale_index_slot_execution_frame(frame, virtual_unit, order_metadata)
            exit_order.metadata = copy_metadata(order_metadata)
            trade, cash_delta = self.execution_model.execute_exit(
                position,
                trade_date,
                frame,
                self._index_slot_virtual_config(virtual_unit),
                reason=pending_exit.reason,
            )
            if trade is None:
                exit_order.mark_skipped("no_fill")
                ledger.register_order(exit_order)
                if trade_index + 1 < len(trade_dates):
                    retry_exit = replace(
                        pending_exit,
                        scheduled_trade_date=trade_dates[trade_index + 1],
                    )
                    pending_risk_exits[retry_exit.scheduled_trade_date].append(retry_exit)
                self._log_event(
                    "pending_risk_exit_delayed",
                    trade_date=trade_date,
                    code=pending_exit.code,
                    reason=pending_exit.reason,
                    skip_reason="no_fill",
                )
                continue

            exit_order.mark_filled(trade)
            ledger.register_order(exit_order)
            ledger.close_position(
                pending_exit.code,
                trade,
                cash_delta,
                exit_order.order_id,
                exit_trade_index=trade_index,
                exit_reason=pending_exit.reason,
            )
            self._log_event(
                "pending_risk_exit_filled",
                trade_date=trade_date,
                code=pending_exit.code,
                reason=pending_exit.reason,
                risk_rule_name=pending_exit.risk_rule_name,
                price=trade.price,
                quantity=trade.quantity,
                cash=ledger.cash,
            )

    def _process_intraday_risk_exits(
        self,
        ledger: PortfolioLedger,
        trade_index: int,
        trade_date: datetime,
    ) -> None:
        """执行当天即可触发并成交的风险退出。"""

        intraday_policies = self._exit_policies_for_stage(EXIT_STAGE_INTRADAY)
        if not intraday_policies or not ledger.positions:
            return

        risk_frames = self._load_intraday_risk_frames(list(ledger.positions.keys()), trade_date)
        for code in list(ledger.positions.keys()):
            position = ledger.positions.get(code)
            if position is None:
                continue

            frame = risk_frames.get(code)
            if frame is None or frame.empty:
                continue

            selected_policy = None
            decision = None
            for policy in intraday_policies:
                decision = policy.evaluate(
                    position,
                    trade_date,
                    frame,
                    data_frequency=self.execution_model.data_frequency,
                    history_frame=None,
                )
                if decision is not None:
                    selected_policy = policy
                    break

            if selected_policy is None or decision is None:
                continue

            exit_order = StockOrder.create(
                code=code,
                side="SELL",
                signal_date=position.signal_date,
                scheduled_trade_date=trade_date,
                execution_model=self.execution_model.data_frequency,
                reason=decision.reason,
                requested_quantity=position.quantity,
                score=position.score,
                metadata=merge_metadata(position.metadata, decision.metadata),
                trigger_price=decision.trade_price,
                risk_rule_name=selected_policy.__class__.__name__,
            )
            if self.execution_model.data_frequency == "daily":
                row = first_sorted_row(frame)
                if row is None:
                    exit_order.mark_skipped("no_execution_data")
                    ledger.register_order(exit_order)
                    continue
                preclose = float(row["preclose"]) if "preclose" in row and pd.notna(row["preclose"]) else None
                is_st = bool(row["isST"]) if "isST" in row and pd.notna(row["isST"]) else False
                sell_fill = decide_daily_sell_fill(
                    position.code,
                    preclose=preclose,
                    open_price=float(row["open"]),
                    high_price=float(row["high"]),
                    low_price=float(row["low"]),
                    close_price=float(row["close"]),
                    fallback_price=float(decision.trade_price),
                    is_st=is_st,
                )
                if not sell_fill.fillable or sell_fill.execution_price is None:
                    exit_order.mark_skipped(sell_fill.reject_reason or "no_fill")
                    ledger.register_order(exit_order)
                    self._log_event(
                        "intraday_risk_exit_skipped",
                        trade_date=trade_date,
                        code=code,
                        reason=decision.reason,
                        risk_rule_name=selected_policy.__class__.__name__,
                        skip_reason=sell_fill.reject_reason or "no_fill",
                    )
                    continue
                decision = replace(decision, trade_price=sell_fill.execution_price)
            trade, cash_delta = build_exit_trade_from_price(
                position,
                trade_date,
                decision.trade_time,
                decision.trade_price,
                self.config,
                reason=decision.reason,
                metadata=merge_metadata(position.metadata, decision.metadata),
            )
            if trade is None:
                exit_order.mark_skipped("invalid_trigger_price")
                ledger.register_order(exit_order)
                self._log_event(
                    "intraday_risk_exit_skipped",
                    trade_date=trade_date,
                    code=code,
                    reason=decision.reason,
                    risk_rule_name=selected_policy.__class__.__name__,
                    skip_reason="invalid_trigger_price",
                )
                continue

            exit_order.mark_filled(trade)
            ledger.register_order(exit_order)
            ledger.close_position(
                code,
                trade,
                cash_delta,
                exit_order.order_id,
                exit_trade_index=trade_index,
                exit_reason=decision.reason,
            )
            self._log_event(
                "intraday_risk_exit_filled",
                trade_date=trade_date,
                code=code,
                reason=decision.reason,
                risk_rule_name=selected_policy.__class__.__name__,
                price=trade.price,
                quantity=trade.quantity,
                cash=ledger.cash,
            )

    @staticmethod
    def _candidate_entry_type(candidate: DailyCandidate) -> str:
        return str(candidate.metadata.get("entry_type", "initial") or "initial")

    def _process_index_slot_rebalance_intents(
        self,
        ledger: PortfolioLedger,
        trade_index: int,
        trade_date: datetime,
        intents: list[IndexSlotRebalanceIntent],
    ) -> None:
        """在普通信号模式下按指数槽位目标做窄口径调仓。"""

        if not intents:
            return

        execution_frames = self._load_execution_frames(
            sorted({intent.code for intent in intents}),
            trade_date,
            for_exit=False,
        )
        for intent in intents:
            frame = execution_frames.get(intent.code)
            metadata = copy_metadata(intent.metadata)
            metadata["index_target_market_value"] = float(intent.target_market_value)
            target_market_value = max(float(intent.target_market_value), 0.0)
            position = ledger.positions.get(intent.code)
            virtual_unit = self._index_slot_virtual_unit(metadata, position=position)
            metadata["index_virtual_unit"] = virtual_unit
            current_market_value = float(position.market_value) if position is not None else 0.0

            if position is not None and current_market_value > target_market_value:
                self._execute_index_slot_rebalance_sell(
                    ledger,
                    trade_index,
                    trade_date,
                    intent,
                    frame,
                    position,
                    current_market_value=current_market_value,
                    target_market_value=target_market_value,
                    metadata=metadata,
                )
                position = ledger.positions.get(intent.code)
                current_market_value = float(position.market_value) if position is not None else 0.0

            buy_budget = target_market_value - current_market_value
            if intent.reason == "index_slot_release_for_event_entry":
                continue
            if buy_budget <= 0:
                continue

            buy_metadata = merge_metadata(
                metadata,
                {
                    "entry_source": "weekly_index_slot_fill",
                    "entry_reason": "amount_shock_event_regime_index_slot_fill",
                    "forced_budget": buy_budget,
                    "target_exit_trade_date": datetime.max.replace(hour=0, minute=0, second=0, microsecond=0),
                },
            )
            self._execute_index_slot_rebalance_buy(
                ledger,
                trade_index,
                trade_date,
                intent,
                frame,
                buy_budget=buy_budget,
                metadata=buy_metadata,
                virtual_unit=virtual_unit,
            )

    def _execute_index_slot_rebalance_buy(
        self,
        ledger: PortfolioLedger,
        trade_index: int,
        trade_date: datetime,
        intent: IndexSlotRebalanceIntent,
        frame: pd.DataFrame | None,
        *,
        buy_budget: float,
        metadata: dict,
        virtual_unit: float,
    ) -> None:
        existing_position = ledger.positions.get(intent.code)
        buy_metadata = copy_metadata(metadata)
        buy_metadata["forced_budget"] = float(buy_budget)
        buy_metadata["target_exit_trade_date"] = datetime.max.replace(hour=0, minute=0, second=0, microsecond=0)
        if existing_position is not None:
            buy_metadata["entry_type"] = "add_on"
            buy_metadata["add_on_stage"] = "weekly_index_slot_fill"

        budget = max(min(float(buy_budget), ledger.cash), 0.0)
        entry_order = StockOrder.create(
            code=intent.code,
            side="BUY",
            signal_date=intent.signal_date,
            scheduled_trade_date=trade_date,
            execution_model=self.execution_model.data_frequency,
            reason=str(buy_metadata.get("entry_reason", "amount_shock_event_regime_index_slot_fill")),
            requested_budget=budget,
            score=0.0,
            metadata=buy_metadata,
        )
        if frame is None or frame.empty:
            entry_order.mark_skipped("no_execution_data")
            ledger.register_order(entry_order)
            self._log_event(
                "index_slot_rebalance_entry_skipped",
                trade_date=trade_date,
                code=intent.code,
                skip_reason="no_execution_data",
            )
            return
        if budget <= 0:
            entry_order.mark_skipped("no_cash")
            ledger.register_order(entry_order)
            self._log_event(
                "index_slot_rebalance_entry_skipped",
                trade_date=trade_date,
                code=intent.code,
                skip_reason="no_cash",
            )
            return

        scaled_frame = self._scale_index_slot_execution_frame(frame, virtual_unit, buy_metadata)
        target_exit_trade_date = buy_metadata["target_exit_trade_date"]
        candidate = DailyCandidate(
            signal_date=intent.signal_date,
            code=intent.code,
            score=0.0,
            hold_days=1,
            metadata=buy_metadata,
        )
        position, trade, cash_delta = self.execution_model.execute_entry(
            candidate,
            trade_date,
            scaled_frame,
            budget,
            self._index_slot_virtual_config(virtual_unit),
            target_exit_trade_date,
        )
        if position is not None and trade is not None:
            entry_order.metadata = copy_metadata(buy_metadata)
            entry_order.mark_filled(trade)
            ledger.register_order(entry_order)
            if existing_position is not None:
                existing_position.last_price = trade.price
                ledger.add_to_position(
                    intent.code,
                    trade,
                    cash_delta,
                    entry_order.order_id,
                    score=0.0,
                    metadata=copy_metadata(buy_metadata),
                    target_exit_trade_date=target_exit_trade_date,
                )
                self._log_event(
                    "index_slot_rebalance_add_on_filled",
                    trade_date=trade_date,
                    code=intent.code,
                    price=trade.price,
                    quantity=trade.quantity,
                    budget=budget,
                    cash=ledger.cash,
                )
                return

            position.position_id = uuid.uuid4().hex
            position.open_order_id = entry_order.order_id
            position.entry_transaction_cost = trade.commission + trade.tax
            position.entry_trade_index = trade_index
            self._apply_candidate_risk_to_position(position, candidate)
            if self.exit_policy is not None:
                self.exit_policy.initialize_position(position)
            ledger.open_position(position, trade, cash_delta)
            self._log_event(
                "index_slot_rebalance_entry_filled",
                trade_date=trade_date,
                code=intent.code,
                price=trade.price,
                quantity=trade.quantity,
                budget=budget,
                cash=ledger.cash,
            )
            return

        entry_order.mark_skipped("no_fill")
        ledger.register_order(entry_order)
        self._log_event(
            "index_slot_rebalance_entry_skipped",
            trade_date=trade_date,
            code=intent.code,
            skip_reason="no_fill",
            budget=budget,
        )

    def _execute_index_slot_rebalance_sell(
        self,
        ledger: PortfolioLedger,
        trade_index: int,
        trade_date: datetime,
        intent: IndexSlotRebalanceIntent,
        frame: pd.DataFrame | None,
        position,
        *,
        current_market_value: float,
        target_market_value: float,
        metadata: dict,
    ) -> None:
        virtual_unit = self._index_slot_virtual_unit(metadata, position=position)
        lot_size = self._index_slot_virtual_lot_size(virtual_unit)
        metadata["index_virtual_unit"] = virtual_unit
        if frame is None or frame.empty:
            fallback_price = self._index_slot_rebalance_fallback_price(position)
            if fallback_price is None:
                return
            sell_quantity = self._index_slot_rebalance_sell_quantity(
                position,
                current_market_value=current_market_value,
                target_market_value=target_market_value,
                reference_price=fallback_price,
                lot_size=lot_size,
            )
            if sell_quantity < lot_size:
                return
            order = StockOrder.create(
                code=intent.code,
                side="SELL",
                signal_date=position.signal_date,
                scheduled_trade_date=trade_date,
                execution_model=self.execution_model.data_frequency,
                reason=intent.reason,
                requested_quantity=sell_quantity,
                score=position.score,
                metadata=metadata,
            )
            order.mark_skipped("no_execution_data")
            ledger.register_order(order)
            return

        scaled_frame = self._scale_index_slot_execution_frame(frame, virtual_unit, metadata)
        reference_price = self._execution_reference_price(scaled_frame)
        if reference_price is None or reference_price <= 0:
            return

        sell_quantity = self._index_slot_rebalance_sell_quantity(
            position,
            current_market_value=current_market_value,
            target_market_value=target_market_value,
            reference_price=reference_price,
            lot_size=lot_size,
        )
        if sell_quantity < lot_size:
            return

        order = StockOrder.create(
            code=intent.code,
            side="SELL",
            signal_date=position.signal_date,
            scheduled_trade_date=trade_date,
            execution_model=self.execution_model.data_frequency,
            reason=intent.reason,
            requested_quantity=sell_quantity,
            score=position.score,
            metadata=metadata,
        )
        reduce_position = replace(position, quantity=sell_quantity)
        trade, cash_delta = self.execution_model.execute_exit(
            reduce_position,
            trade_date,
            scaled_frame,
            self._index_slot_virtual_config(virtual_unit),
            reason=intent.reason,
        )
        if trade is None:
            order.mark_skipped("no_fill")
            ledger.register_order(order)
            return

        order.mark_filled(trade)
        ledger.register_order(order)
        ledger.reduce_position(
            intent.code,
            trade,
            cash_delta,
            order.order_id,
            reduce_quantity=sell_quantity,
            exit_trade_index=trade_index,
            exit_reason=intent.reason,
        )

    def _index_slot_rebalance_sell_quantity(
        self,
        position,
        *,
        current_market_value: float,
        target_market_value: float,
        reference_price: float,
        lot_size: int | None = None,
    ) -> int:
        if reference_price <= 0:
            return 0
        resolved_lot_size = max(int(lot_size or self.config.lot_size), 1)
        position_quantity = int(position.quantity)
        if target_market_value <= 0:
            return position_quantity
        sell_value = max(current_market_value - target_market_value, 0.0)
        raw_quantity = int(sell_value // reference_price)
        sell_quantity = (raw_quantity // resolved_lot_size) * resolved_lot_size
        return min(sell_quantity, position_quantity)

    @staticmethod
    def _index_slot_virtual_unit(metadata: dict | None = None, *, position=None) -> float:
        for source in (metadata, getattr(position, "metadata", None)):
            if not source:
                continue
            value = source.get("index_virtual_unit")
            if value is None:
                continue
            unit = pd.to_numeric(value, errors="coerce")
            if pd.notna(unit) and float(unit) > 0:
                return float(unit)
        return 1.0

    def _index_slot_virtual_lot_size(self, virtual_unit: float) -> int:
        return 1 if virtual_unit != 1.0 else self.config.lot_size

    def _index_slot_virtual_config(self, virtual_unit: float) -> EngineConfig:
        if virtual_unit == 1.0 and self.config.lot_size == self._index_slot_virtual_lot_size(virtual_unit):
            return self.config
        return replace(self.config, lot_size=self._index_slot_virtual_lot_size(virtual_unit))

    def _scale_index_slot_execution_frame(
        self,
        frame: pd.DataFrame,
        virtual_unit: float,
        metadata: dict | None = None,
    ) -> pd.DataFrame:
        if virtual_unit == 1.0:
            return frame

        scaled = frame.copy()
        price_columns = ("open", "high", "low", "close", "preclose")
        for column in price_columns:
            if column in scaled.columns:
                scaled[column] = pd.to_numeric(scaled[column], errors="coerce") * virtual_unit

        row = first_sorted_row(frame)
        if metadata is not None and row is not None:
            raw_price = None
            for column in ("open", "close"):
                if column in row and pd.notna(row[column]):
                    raw_price = float(row[column])
                    break
            if raw_price is not None and raw_price > 0:
                metadata["raw_index_price"] = raw_price
                metadata["scaled_index_price"] = raw_price * virtual_unit
        return scaled

    @staticmethod
    def _index_slot_rebalance_fallback_price(position) -> float | None:
        for value in (getattr(position, "last_price", None), getattr(position, "entry_price", None)):
            price = pd.to_numeric(value, errors="coerce")
            if pd.notna(price) and float(price) > 0:
                return float(price)
        return None

    def _estimate_risk_budget(
        self,
        ledger: PortfolioLedger,
        candidate: DailyCandidate,
        *,
        trade_date: datetime,
        slot_budget: float,
        existing_position=None,
    ) -> float:
        # 把策略层给出的“每股风险”元数据换算成真正下单用的现金预算。
        metadata = copy_metadata(candidate.metadata)
        forced_budget = metadata.get("forced_budget")
        if forced_budget is not None:
            return max(float(forced_budget), 0.0)

        risk_per_share = metadata.get("risk_per_share")
        entry_reference_price = metadata.get("entry_reference_price")
        risk_fraction = metadata.get("risk_fraction")
        if risk_per_share is None or entry_reference_price is None or risk_fraction is None:
            return max(slot_budget, 0.0)

        risk_per_share = float(risk_per_share)
        entry_reference_price = float(entry_reference_price)
        risk_fraction = float(risk_fraction)
        if risk_per_share <= 0 or entry_reference_price <= 0 or risk_fraction <= 0:
            return max(slot_budget, 0.0)

        total_equity = self._portfolio_equity(ledger, trade_date)
        allowed_risk = total_equity * risk_fraction
        raw_quantity = int(allowed_risk // risk_per_share)
        quantity = (raw_quantity // self.config.lot_size) * self.config.lot_size
        if quantity <= 0:
            return 0.0

        estimated_budget = quantity * entry_reference_price * (1 + self.config.buy_commission_rate)
        code_budget_cap = total_equity * self.config.position_size_pct
        if existing_position is not None:
            code_budget_cap = max(code_budget_cap - existing_position.market_value, 0.0)
        estimated_budget = min(estimated_budget, code_budget_cap)
        return min(ledger.cash, estimated_budget, slot_budget if slot_budget > 0 else estimated_budget)

    @staticmethod
    def _apply_candidate_risk_to_position(position, candidate: DailyCandidate) -> None:
        # 把策略生成的 stop / risk 元数据写入持仓对象，
        # 这样退出规则就只需要读取 position，而不用反向理解策略细节。
        metadata = copy_metadata(candidate.metadata)
        position.metadata.update(metadata)
        if metadata.get("initial_stop_loss") is not None:
            position.initial_stop_loss = float(metadata["initial_stop_loss"])
        if metadata.get("current_stop_loss") is not None:
            position.current_stop_loss = float(metadata["current_stop_loss"])
        if metadata.get("risk_per_share") is not None:
            position.risk_budget = float(metadata["risk_per_share"]) * float(position.quantity)
        if position.highest_price_since_entry is None:
            position.highest_price_since_entry = position.entry_price
        if position.lowest_price_since_entry is None:
            position.lowest_price_since_entry = position.entry_price

    def _execute_entry_candidates(
        self,
        ledger: PortfolioLedger,
        trade_dates: list[datetime],
        trade_index: int,
        trade_date: datetime,
        candidates: list[DailyCandidate],
        *,
        max_total_budget: float | None = None,
        log_prefix: str = "scheduled",
    ) -> None:
        """执行一组买入候选，可选限制当日总买入资金。"""

        if not candidates:
            return

        remaining_slots = max(self.config.max_positions - len(ledger.positions), 0)
        processed_candidates: list[DailyCandidate] = []
        initial_candidates: list[DailyCandidate] = []
        add_on_candidates: list[DailyCandidate] = []
        seen_initial_codes: set[str] = set()
        seen_add_on_codes: set[str] = set()
        for candidate in candidates:
            entry_type = self._candidate_entry_type(candidate)
            if entry_type == "add_on":
                if candidate.code in seen_add_on_codes or candidate.code not in ledger.positions:
                    continue
                seen_add_on_codes.add(candidate.code)
                add_on_candidates.append(candidate)
                continue
            if candidate.code in seen_initial_codes or candidate.code in ledger.positions:
                continue
            seen_initial_codes.add(candidate.code)
            initial_candidates.append(candidate)

        if remaining_slots <= 0 and not add_on_candidates:
            self._log_event(
                f"{log_prefix}_entries_skipped",
                trade_date=trade_date,
                skip_reason="no_remaining_slots",
                candidate_codes=[candidate.code for candidate in initial_candidates],
            )
            return

        initial_candidates = initial_candidates[:remaining_slots]
        processed_candidates.extend(add_on_candidates)
        processed_candidates.extend(initial_candidates)
        if not processed_candidates:
            return

        execution_frames = self._load_execution_frames(
            sorted({candidate.code for candidate in processed_candidates}),
            trade_date,
            for_exit=False,
        )
        remaining_buy_budget = None
        if max_total_budget is not None:
            remaining_buy_budget = max(min(float(max_total_budget), ledger.cash), 0.0)

        initial_offset = 0
        for candidate in processed_candidates:
            frame = execution_frames.get(candidate.code)
            existing_position = ledger.positions.get(candidate.code)
            entry_type = self._candidate_entry_type(candidate)
            if entry_type == "add_on":
                cash_for_budget = ledger.cash if remaining_buy_budget is None else remaining_buy_budget
                slot_budget = cash_for_budget
            else:
                positions_left = max(remaining_slots - initial_offset, 1)
                cash_for_budget = ledger.cash if remaining_buy_budget is None else remaining_buy_budget
                slot_budget = self.position_sizer.allocate(
                    cash=cash_for_budget,
                    remaining_slots=positions_left,
                    max_position_pct=self.config.position_size_pct,
                )
                initial_offset += 1
            budget = self._estimate_risk_budget(
                ledger,
                candidate,
                trade_date=trade_date,
                slot_budget=slot_budget,
                existing_position=existing_position,
            )
            if remaining_buy_budget is not None:
                budget = min(budget, remaining_buy_budget)
            entry_order = StockOrder.create(
                code=candidate.code,
                side="BUY",
                signal_date=candidate.signal_date,
                scheduled_trade_date=trade_date,
                execution_model=self.execution_model.data_frequency,
                reason=str(candidate.metadata.get("entry_reason", f"{log_prefix}_entry")),
                requested_budget=budget,
                score=candidate.score,
                metadata=copy_metadata(candidate.metadata),
            )
            if frame is None or frame.empty:
                entry_order.mark_skipped("no_execution_data")
                ledger.register_order(entry_order)
                self._log_event(
                    f"{log_prefix}_entry_skipped",
                    trade_date=trade_date,
                    code=candidate.code,
                    skip_reason="no_execution_data",
                )
                continue
            if budget <= 0:
                entry_order.mark_skipped("no_cash")
                ledger.register_order(entry_order)
                self._log_event(
                    f"{log_prefix}_entry_skipped",
                    trade_date=trade_date,
                    code=candidate.code,
                    skip_reason="no_cash",
                )
                continue

            metadata_exit_trade_date = candidate.metadata.get("target_exit_trade_date")
            if isinstance(metadata_exit_trade_date, datetime):
                target_exit_trade_date = metadata_exit_trade_date
            else:
                target_exit_trade_date = self._resolve_exit_trade_date(
                    trade_dates,
                    trade_index,
                    candidate.hold_days,
                )

            position, trade, cash_delta = self.execution_model.execute_entry(
                candidate,
                trade_date,
                frame,
                budget,
                self.config,
                target_exit_trade_date,
            )
            if position is not None and trade is not None:
                entry_order.mark_filled(trade)
                ledger.register_order(entry_order)
                if remaining_buy_budget is not None:
                    remaining_buy_budget = max(remaining_buy_budget + cash_delta, 0.0)
                if entry_type == "add_on" and existing_position is not None:
                    existing_position.last_price = trade.price
                    ledger.add_to_position(
                        candidate.code,
                        trade,
                        cash_delta,
                        entry_order.order_id,
                        score=candidate.score,
                        metadata=copy_metadata(candidate.metadata),
                        target_exit_trade_date=target_exit_trade_date,
                    )
                    self._log_event(
                        f"{log_prefix}_add_on_filled",
                        trade_date=trade_date,
                        code=candidate.code,
                        price=trade.price,
                        quantity=trade.quantity,
                        budget=budget,
                        target_exit_trade_date=target_exit_trade_date,
                        cash=ledger.cash,
                    )
                else:
                    position.position_id = uuid.uuid4().hex
                    position.open_order_id = entry_order.order_id
                    position.entry_transaction_cost = trade.commission + trade.tax
                    position.entry_trade_index = trade_index
                    self._apply_candidate_risk_to_position(position, candidate)
                    if self.exit_policy is not None:
                        self.exit_policy.initialize_position(position)
                    ledger.open_position(position, trade, cash_delta)
                    self._log_event(
                        f"{log_prefix}_entry_filled",
                        trade_date=trade_date,
                        code=candidate.code,
                        price=trade.price,
                        quantity=trade.quantity,
                        budget=budget,
                        target_exit_trade_date=target_exit_trade_date,
                        cash=ledger.cash,
                    )
            else:
                if remaining_buy_budget is not None:
                    remaining_buy_budget = max(remaining_buy_budget, 0.0)
                entry_order.mark_skipped("no_fill")
                ledger.register_order(entry_order)
                self._log_event(
                    f"{log_prefix}_entry_skipped",
                    trade_date=trade_date,
                    code=candidate.code,
                    skip_reason="no_fill",
                    budget=budget,
                )

    def run(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        research_store: ResearchDailyHistoryStore | None = None,
    ) -> BacktestResult:
        """执行完整回测。"""

        trade_dates = self.data_portal.get_trade_calendar(start_date, end_date)
        if not trade_dates:
            return BacktestResult(
                trades=[],
                orders=[],
                equity_curve=[],
                final_positions={},
                closed_positions=[],
                corporate_actions=[],
                benchmark_code=self.benchmark_code,
                risk_free_rate=self.config.risk_free_rate,
                benchmark_curve=[],
            )

        ledger = PortfolioLedger.create(self.config.initial_cash)
        pending_entries: dict[datetime, list[DailyCandidate]] = defaultdict(list)
        pending_risk_exits: dict[datetime, list[PendingRiskExit]] = defaultdict(list)
        close_confirmed_history_specs = self._build_close_confirmed_history_specs(start_date, end_date)
        if research_store is None:
            research_store = ResearchDailyHistoryStore(self.data_portal)

        self.strategy.prepare(self.data_portal, trade_dates, research_store=research_store)
        corporate_action_events_by_date = self._preload_corporate_action_events(start_date, end_date)
        self._log_event(
            "backtest_start",
            start_date=start_date,
            end_date=end_date,
            trade_date_count=len(trade_dates),
            initial_cash=self.config.initial_cash,
            max_positions=self.config.max_positions,
            corporate_action_event_count=sum(len(events) for events in corporate_action_events_by_date.values()),
            intraday_risk_policy_count=len(self._exit_policies_for_stage(EXIT_STAGE_INTRADAY)),
            close_confirmed_risk_policy_count=len(self._exit_policies_for_stage(EXIT_STAGE_CLOSE_CONFIRMED)),
        )

        for idx, trade_date in enumerate(trade_dates):
            self._log_event(
                "trade_date_start",
                trade_date=trade_date,
                trade_index=idx,
                total_trade_dates=len(trade_dates),
                cash=ledger.cash,
                position_count=len(ledger.positions),
                held_codes=list(ledger.positions.keys()),
                pending_entry_count=len(pending_entries.get(trade_date, [])),
                pending_risk_exit_count=len(pending_risk_exits.get(trade_date, [])),
            )

            self._process_corporate_actions(
                ledger,
                trade_date,
                corporate_action_events_by_date.get(trade_date, []),
            )
            self._process_pending_risk_exits(
                ledger,
                pending_risk_exits,
                trade_dates,
                idx,
                trade_date,
            )
            self._process_intraday_risk_exits(ledger, idx, trade_date)

            if not self.config.execute_on_next_trade_date:
                self._generate_and_schedule_candidates(ledger, pending_entries, trade_dates, idx, trade_date)
            self._process_execution_day(
                ledger,
                trade_dates,
                idx,
                trade_date,
                pending_entries,
            )
            self._mark_portfolio_to_market(ledger, trade_date)
            self._schedule_close_confirmed_risk_exits(
                ledger,
                pending_risk_exits,
                trade_dates,
                idx,
                trade_date,
                close_confirmed_history_specs,
                research_store,
            )
            if self.config.execute_on_next_trade_date:
                self._generate_and_schedule_candidates(ledger, pending_entries, trade_dates, idx, trade_date)

            self._log_event(
                "trade_date_end",
                trade_date=trade_date,
                cash=ledger.cash,
                position_count=len(ledger.positions),
                held_codes=list(ledger.positions.keys()),
            )

        return BacktestResult(
            trades=ledger.trades,
            orders=ledger.orders,
            equity_curve=ledger.equity_curve,
            final_positions=ledger.positions,
            closed_positions=ledger.closed_positions,
            corporate_actions=ledger.corporate_action_records,
            benchmark_code=self.benchmark_code,
            risk_free_rate=self.config.risk_free_rate,
            benchmark_curve=self._build_benchmark_curve(trade_dates),
        )
