"""执行引擎里的账务、公司行为与权益估值辅助逻辑。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from backtest.portfolio import CorporateActionEvent, PortfolioLedger
from backtest.utils.datetime_utils import to_pydatetime


class EngineAccountingMixin:
    def _preload_corporate_action_events(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[datetime, list[CorporateActionEvent]]:
        """一次性预加载公司行为事件，并按除权除息日分桶。"""

        events_frame = self.data_portal.get_corporate_action_events(start_date, end_date)
        if events_frame.empty:
            self._log_event(
                "corporate_action_preload",
                start_date=start_date,
                end_date=end_date,
                event_count=0,
                operate_date_count=0,
            )
            return {}

        events_by_date: dict[datetime, list[CorporateActionEvent]] = defaultdict(list)
        for row in events_frame.itertuples(index=False):
            operate_date = to_pydatetime(row.operate_date)
            settle_date = to_pydatetime(row.settle_date)
            events_by_date[operate_date].append(
                CorporateActionEvent(
                    event_type=row.event_type,
                    code=row.code,
                    operate_date=operate_date,
                    settle_date=settle_date,
                    cash_dividend_per_share=float(row.cash_dividend_per_share or 0.0),
                    stock_dividend_ratio=float(row.stock_dividend_ratio or 0.0),
                    stock_dividend_share_ratio=float(row.stock_dividend_share_ratio or 0.0),
                    reserve_to_stock_ratio=float(row.reserve_to_stock_ratio or 0.0),
                    raw_text=row.raw_text or "",
                    metadata={},
                )
            )

        self._log_event(
            "corporate_action_preload",
            start_date=start_date,
            end_date=end_date,
            event_count=len(events_frame),
            operate_date_count=len(events_by_date),
        )
        return dict(events_by_date)

    def _process_corporate_actions(
        self,
        ledger: PortfolioLedger,
        trade_date: datetime,
        daily_events: list[CorporateActionEvent],
    ) -> None:
        """先生成应收，再结算今天到期的现金和送转股份。"""

        if daily_events:
            event_codes = sorted({event.code for event in daily_events})
            eligible_quantity_map = ledger.capture_eligible_quantity_snapshot(event_codes)
            ordered_events = sorted(
                daily_events,
                key=lambda event: (
                    event.code,
                    0 if event.event_type == "cash_dividend" else 1,
                    event.settle_date,
                ),
            )
            for event in ordered_events:
                eligible_quantity = eligible_quantity_map.get(event.code, 0)
                position = ledger.positions.get(event.code)
                if position is None or eligible_quantity <= 0:
                    continue

                if event.event_type == "cash_dividend":
                    receivable = ledger.create_cash_dividend_receivable(
                        position,
                        event,
                        eligible_quantity,
                        trade_date,
                    )
                    if receivable is not None:
                        self._log_event(
                            "cash_dividend_receivable_created",
                            trade_date=trade_date,
                            code=event.code,
                            operate_date=event.operate_date,
                            settle_date=event.settle_date,
                            eligible_quantity=eligible_quantity,
                            cash_dividend_per_share=event.cash_dividend_per_share,
                            cash_amount=receivable.amount,
                        )
                elif event.event_type == "stock_dividend":
                    receivable = ledger.create_stock_dividend_receivable(
                        position,
                        event,
                        eligible_quantity,
                        trade_date,
                    )
                    if receivable is not None:
                        self._log_event(
                            "stock_dividend_receivable_created",
                            trade_date=trade_date,
                            code=event.code,
                            operate_date=event.operate_date,
                            settle_date=event.settle_date,
                            eligible_quantity=eligible_quantity,
                            stock_dividend_ratio=event.stock_dividend_ratio,
                            bonus_quantity=receivable.bonus_quantity,
                            allocated_cost_basis=receivable.allocated_cost_basis,
                        )

        due_cash_receivables = [
            receivable
            for receivable in ledger.cash_dividend_receivables
            if receivable.settle_date <= trade_date
        ]
        due_stock_receivables = [
            receivable
            for receivable in ledger.stock_dividend_receivables
            if receivable.settle_date <= trade_date
        ]
        ledger.settle_cash_dividends(trade_date)
        ledger.settle_stock_dividends(trade_date)

        for receivable in due_cash_receivables:
            self._log_event(
                "cash_dividend_settled",
                trade_date=trade_date,
                code=receivable.code,
                operate_date=receivable.operate_date,
                settle_date=receivable.settle_date,
                eligible_quantity=receivable.eligible_quantity,
                cash_amount=receivable.amount,
            )
        for receivable in due_stock_receivables:
            self._log_event(
                "stock_dividend_settled",
                trade_date=trade_date,
                code=receivable.code,
                operate_date=receivable.operate_date,
                settle_date=receivable.settle_date,
                eligible_quantity=receivable.eligible_quantity,
                bonus_quantity=receivable.bonus_quantity,
                allocated_cost_basis=receivable.allocated_cost_basis,
            )

    def _build_portfolio_valuation_close_map(
        self,
        ledger: PortfolioLedger,
        trade_date: datetime | None,
    ) -> dict[str, float]:
        """构造执行中途可复用的估值价格映射。"""

        codes = ledger.valuation_codes()
        close_map: dict[str, float] = {}
        if trade_date is not None and codes:
            close_map = {
                str(code): float(price)
                for code, price in self.data_portal.get_daily_close_map(codes, trade_date).items()
                if price is not None
            }

        for code, position in ledger.positions.items():
            if code in close_map:
                continue
            if position.last_price is not None and position.last_price > 0:
                close_map[code] = float(position.last_price)
            elif position.entry_price > 0:
                close_map[code] = float(position.entry_price)
        return close_map

    def _portfolio_equity(
        self,
        ledger: PortfolioLedger,
        trade_date: datetime | None = None,
    ) -> float:
        """估算当前总权益，统一纳入现金/持仓/应收资产。"""

        market_value = sum(position.market_value for position in ledger.positions.values())
        close_map = self._build_portfolio_valuation_close_map(ledger, trade_date)
        cash_receivable_value, stock_receivable_value = ledger.receivable_market_value(close_map)
        return ledger.cash + market_value + cash_receivable_value + stock_receivable_value

    def _mark_portfolio_to_market(self, ledger: PortfolioLedger, trade_date: datetime) -> None:
        """按收盘价更新持仓和应收资产的总权益。"""

        close_map = self.data_portal.get_daily_close_map(ledger.valuation_codes(), trade_date)
        ledger.mark_to_market(trade_date, close_map)
