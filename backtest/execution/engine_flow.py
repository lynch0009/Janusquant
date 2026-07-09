"""执行引擎里的日执行流程与普通调度辅助逻辑。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from backtest.data import ResearchDailyHistoryStore
from backtest.portfolio import PortfolioLedger, StockOrder
from backtest.risk import EXIT_STAGE_CLOSE_CONFIRMED
from backtest.strategies import DailyCandidate
from backtest.utils.metadata import copy_metadata

from .engine_types import PendingRiskExit


class EngineFlowMixin:
    def _process_scheduled_exits(
        self,
        ledger: PortfolioLedger,
        trade_dates: list[datetime],
        trade_index: int,
        trade_date: datetime,
    ) -> None:
        """处理达到计划退出日的持仓。"""

        exit_codes = [
            code
            for code, position in ledger.positions.items()
            if position.target_exit_trade_date <= trade_date
        ]
        if not exit_codes:
            return

        execution_frames = self._load_execution_frames(exit_codes, trade_date, for_exit=True)
        for code in exit_codes:
            position = ledger.positions.get(code)
            if position is None:
                continue

            frame = execution_frames.get(code)
            exit_order = StockOrder.create(
                code=code,
                side="SELL",
                signal_date=position.signal_date,
                scheduled_trade_date=trade_date,
                execution_model=self.execution_model.data_frequency,
                reason="scheduled_exit",
                requested_quantity=position.quantity,
                score=position.score,
                metadata=copy_metadata(position.metadata),
            )
            if frame is None or frame.empty:
                exit_order.mark_skipped("no_execution_data")
                ledger.register_order(exit_order)
                self._log_event(
                    "scheduled_exit_skipped",
                    trade_date=trade_date,
                    code=code,
                    skip_reason="no_execution_data",
                )
                continue

            trade, cash_delta = self.execution_model.execute_exit(
                position,
                trade_date,
                frame,
                self.config,
                reason="scheduled_exit",
            )
            if trade is not None:
                exit_order.mark_filled(trade)
                ledger.register_order(exit_order)
                ledger.close_position(
                    code,
                    trade,
                    cash_delta,
                    exit_order.order_id,
                    exit_trade_index=trade_index,
                    exit_reason="scheduled_exit",
                )
                self._log_event(
                    "scheduled_exit_filled",
                    trade_date=trade_date,
                    code=code,
                    price=trade.price,
                    quantity=trade.quantity,
                    cash=ledger.cash,
                )
            else:
                exit_order.mark_skipped("no_fill")
                ledger.register_order(exit_order)
                self._log_event(
                    "scheduled_exit_skipped",
                    trade_date=trade_date,
                    code=code,
                    skip_reason="no_fill",
                )

    def _process_scheduled_entries(
        self,
        ledger: PortfolioLedger,
        trade_dates: list[datetime],
        trade_index: int,
        trade_date: datetime,
        pending_entries: dict[datetime, list[DailyCandidate]],
        *,
        max_total_budget: float | None = None,
    ) -> None:
        """把调度到今天的候选转成真实买单并更新持仓。"""

        todays_candidates = pending_entries.pop(trade_date, [])
        if hasattr(self.strategy, "prepare_before_entries"):
            intents = self.strategy.prepare_before_entries(
                trade_date,
                ledger.snapshot(),
                todays_candidates,
            )
            if intents:
                self._process_index_slot_rebalance_intents(
                    ledger,
                    trade_index,
                    trade_date,
                    list(intents),
                )
        if not todays_candidates:
            return

        self._log_event(
            "scheduled_entries_ready",
            trade_date=trade_date,
            candidate_count=len(todays_candidates),
            candidate_codes=[candidate.code for candidate in todays_candidates],
            max_total_budget=max_total_budget,
        )
        self._execute_entry_candidates(
            ledger,
            trade_dates,
            trade_index,
            trade_date,
            todays_candidates,
            max_total_budget=max_total_budget,
            log_prefix="scheduled",
        )

    def _process_execution_day(
        self,
        ledger: PortfolioLedger,
        trade_dates: list[datetime],
        trade_index: int,
        trade_date: datetime,
        pending_entries: dict[datetime, list[DailyCandidate]],
    ) -> None:
        """执行当天应落账的调仓/买卖流程。"""

        if self._uses_target_portfolio_rebalance:
            # target-portfolio 模式由目标组合 + target_exposure 驱动，
            # 不沿用普通路径里的 scheduled hold_days forced exit 语义。
            self._process_target_portfolio_day(
                ledger,
                trade_dates,
                trade_index,
                trade_date,
                pending_entries,
            )
            return

        self._process_scheduled_exits(ledger, trade_dates, trade_index, trade_date)
        self._process_scheduled_entries(
            ledger,
            trade_dates,
            trade_index,
            trade_date,
            pending_entries,
        )

    def _schedule_close_confirmed_risk_exits(
        self,
        ledger: PortfolioLedger,
        pending_risk_exits: dict[datetime, list[PendingRiskExit]],
        trade_dates: list[datetime],
        trade_index: int,
        trade_date: datetime,
        history_specs: dict[str, dict[str, Any]],
        research_store: ResearchDailyHistoryStore,
    ) -> None:
        """在收盘后评估次日执行的风控规则，例如跌破均线。"""

        close_confirmed_policies = self._exit_policies_for_stage(EXIT_STAGE_CLOSE_CONFIRMED)
        if not close_confirmed_policies or trade_index + 1 >= len(trade_dates):
            return

        for code in list(ledger.positions.keys()):
            if self._has_pending_risk_exit(pending_risk_exits, code):
                continue

            position = ledger.positions.get(code)
            if position is None:
                continue

            for policy in close_confirmed_policies:
                history_frame = self._load_close_confirmed_history_frame(
                    code,
                    trade_date,
                    policy,
                    history_specs,
                    research_store,
                )
                decision = policy.evaluate(
                    position,
                    trade_date,
                    market_frame=None,
                    data_frequency="daily",
                    history_frame=history_frame,
                )
                if decision is None:
                    continue

                self._schedule_pending_risk_exit(
                    pending_risk_exits,
                    trade_dates,
                    trade_index,
                    position,
                    decision,
                    policy,
                )
                break
