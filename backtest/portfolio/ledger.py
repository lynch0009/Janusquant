"""账户对象定义。

账户层负责维护：
1. 现金、持仓、订单、成交
2. 收盘权益曲线
3. 公司行为产生的应收现金、应收股份及对应流水
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, time

from backtest.utils.datetime_utils import combine_trade_date
from backtest.utils.metadata import copy_metadata, merge_metadata

from .corporate_actions import (
    CashDividendReceivable,
    CorporateActionEvent,
    CorporateActionRecord,
    StockDividendReceivable,
)
from .orders import StockOrder
from .positions import StockPosition
from .results import EquityPoint, TradeRecord


@dataclass
class PortfolioLedger:
    """回测期间唯一的账户状态容器。"""

    cash: float
    positions: dict[str, StockPosition]
    trades: list[TradeRecord]
    orders: list[StockOrder]
    equity_curve: list[EquityPoint]
    closed_positions: list[StockPosition]
    cash_dividend_receivables: list[CashDividendReceivable]
    stock_dividend_receivables: list[StockDividendReceivable]
    corporate_action_records: list[CorporateActionRecord]

    @classmethod
    def create(cls, initial_cash: float) -> "PortfolioLedger":
        """按初始资金创建一个空账户。"""

        return cls(
            cash=initial_cash,
            positions={},
            trades=[],
            orders=[],
            equity_curve=[],
            closed_positions=[],
            cash_dividend_receivables=[],
            stock_dividend_receivables=[],
            corporate_action_records=[],
        )

    def snapshot(self) -> dict:
        """返回策略选股阶段需要的轻量组合快照。"""

        return {
            "cash": self.cash,
            "held_codes": list(self.positions.keys()),
            "positions": self.positions.copy(),
        }

    def register_order(self, order: StockOrder) -> None:
        """登记订单，无论成交还是跳过都保留。"""

        self.orders.append(order)

    def register_corporate_action_record(self, record: CorporateActionRecord) -> None:
        """登记公司行为流水。"""

        self.corporate_action_records.append(record)

    def open_position(self, position: StockPosition, trade: TradeRecord, cash_delta: float) -> None:
        """登记开仓成交，同时更新现金和持仓。"""

        self.cash += cash_delta
        position.refresh_adjusted_avg_price()
        self.positions[position.code] = position
        self.trades.append(trade)

    def _require_open_position(self, code: str, *, action: str) -> StockPosition:
        """读取并校验目标持仓存在。"""

        position = self.positions.get(code)
        if position is None:
            raise KeyError(f"position not found for {action}: {code}")
        return position

    @staticmethod
    def _merge_entry_metadata(
        position: StockPosition,
        trade: TradeRecord,
        order_id: str,
        incoming_metadata: dict,
    ) -> dict:
        """把加仓信息合并回原持仓 metadata。"""

        merged_metadata = merge_metadata(position.metadata, incoming_metadata)
        entry_type = incoming_metadata.get("entry_type")
        if entry_type == "add_on":
            merged_metadata["add_on_count"] = int(position.metadata.get("add_on_count", 0) or 0) + 1
            merged_metadata["last_add_on_trade_date"] = trade.trade_date
            merged_metadata["last_add_on_price"] = trade.price
            merged_metadata["last_add_on_order_id"] = order_id
            merged_metadata["last_add_on_stage"] = incoming_metadata.get("add_on_stage")
        merged_metadata["last_entry_type"] = entry_type or merged_metadata.get("last_entry_type")
        merged_metadata["last_entry_trade_date"] = trade.trade_date
        merged_metadata["last_entry_price"] = trade.price
        return merged_metadata

    @staticmethod
    def _apply_stop_loss_from_metadata(position: StockPosition, incoming_metadata: dict) -> None:
        """同步加仓后更严格的止损价。"""

        if incoming_metadata.get("current_stop_loss") is not None:
            incoming_stop = float(incoming_metadata["current_stop_loss"])
            if position.current_stop_loss is None:
                position.current_stop_loss = incoming_stop
            else:
                position.current_stop_loss = max(position.current_stop_loss, incoming_stop)
        if position.initial_stop_loss is None and incoming_metadata.get("initial_stop_loss") is not None:
            position.initial_stop_loss = float(incoming_metadata["initial_stop_loss"])

    @staticmethod
    def _apply_risk_budget_from_metadata(
        position: StockPosition,
        incoming_metadata: dict,
        trade_quantity: int,
    ) -> None:
        """把新成交对应的风险预算累计到老仓位。"""

        prior_risk_budget = float(position.risk_budget or 0.0)
        if incoming_metadata.get("risk_budget") is not None:
            position.risk_budget = prior_risk_budget + float(incoming_metadata["risk_budget"])
            return
        if incoming_metadata.get("risk_per_share") is not None:
            position.risk_budget = prior_risk_budget + float(incoming_metadata["risk_per_share"]) * trade_quantity

    @staticmethod
    def _update_position_price_extremes(position: StockPosition, trade_price: float) -> None:
        """维护开仓以来的最高/最低成交价。"""

        if position.highest_price_since_entry is None:
            position.highest_price_since_entry = trade_price
        else:
            position.highest_price_since_entry = max(position.highest_price_since_entry, trade_price)
        if position.lowest_price_since_entry is None:
            position.lowest_price_since_entry = trade_price
        else:
            position.lowest_price_since_entry = min(position.lowest_price_since_entry, trade_price)

    @staticmethod
    def _build_partial_closed_position(
        position: StockPosition,
        trade: TradeRecord,
        order_id: str,
        *,
        reduce_quantity: int,
        exit_trade_index: int | None = None,
        exit_reason: str | None = None,
    ) -> tuple[StockPosition, float, float]:
        """把部分减仓拆成一笔已关闭的派生持仓。"""

        original_quantity = int(position.quantity)
        original_cost_basis = float(position.share_cost_basis or 0.0)
        original_entry_cost = float(position.entry_transaction_cost or 0.0)
        reduce_ratio = reduce_quantity / original_quantity
        reduced_cost_basis = original_cost_basis * reduce_ratio
        reduced_entry_cost = original_entry_cost * reduce_ratio

        reduced_position = replace(
            position,
            position_id=f"{position.position_id}_partial_{reduce_quantity}_{trade.trade_date:%Y%m%d}",
            quantity=reduce_quantity,
            share_cost_basis=reduced_cost_basis,
            adjusted_avg_price=None,
            entry_transaction_cost=reduced_entry_cost,
            last_price=trade.price,
            status="OPEN",
            exit_trade_date=None,
            exit_time=None,
            exit_price=None,
            close_order_id=None,
            exit_trade_index=None,
            holding_trade_days=None,
            exit_reason=None,
            realized_pnl=None,
            realized_return=None,
        )
        reduced_position.refresh_adjusted_avg_price()
        reduced_position.close(
            trade_date=trade.trade_date,
            trade_time=trade.trade_time,
            trade_price=trade.price,
            close_order_id=order_id,
            total_sell_cost=trade.commission + trade.tax,
            exit_trade_index=exit_trade_index,
            exit_reason=exit_reason,
        )
        return reduced_position, reduced_cost_basis, reduced_entry_cost

    @staticmethod
    def _build_corporate_action_record(
        *,
        trade_date: datetime,
        code: str,
        event_type: str,
        stage: str,
        operate_date: datetime,
        settle_date: datetime,
        eligible_quantity: int,
        cash_amount: float = 0.0,
        bonus_quantity: int = 0,
        allocated_cost_basis: float = 0.0,
        note: str = "",
        metadata: dict | None = None,
    ) -> CorporateActionRecord:
        """统一构造公司行为流水对象。"""

        return CorporateActionRecord(
            trade_date=trade_date,
            code=code,
            event_type=event_type,
            stage=stage,
            operate_date=operate_date,
            settle_date=settle_date,
            eligible_quantity=eligible_quantity,
            cash_amount=cash_amount,
            bonus_quantity=bonus_quantity,
            allocated_cost_basis=allocated_cost_basis,
            note=note,
            metadata=copy_metadata(metadata),
        )

    @staticmethod
    def _build_bonus_position_from_receivable(
        receivable: StockDividendReceivable,
        trade_date: datetime,
    ) -> StockPosition:
        """在原仓位已卖出时，把送转股落成一笔派生仓位。"""

        entry_price = receivable.allocated_cost_basis / receivable.bonus_quantity if receivable.bonus_quantity > 0 else 0.0
        position = StockPosition(
            position_id=f"{receivable.origin_position_id}:bonus:{trade_date:%Y%m%d}",
            code=receivable.code,
            quantity=receivable.bonus_quantity,
            entry_trade_date=trade_date,
            entry_time=combine_trade_date(trade_date, time(9, 30)),
            entry_price=entry_price,
            target_exit_trade_date=trade_date,
            signal_date=receivable.signal_date,
            open_order_id=f"corporate_action:{receivable.origin_position_id}",
            score=receivable.score,
            metadata=merge_metadata(
                receivable.metadata,
                {
                    "from_corporate_action": True,
                    "corporate_action_origin_position_id": receivable.origin_position_id,
                },
            ),
            share_cost_basis=receivable.allocated_cost_basis,
            last_price=entry_price,
        )
        position.cum_bonus_quantity = receivable.bonus_quantity
        return position

    def _settle_cash_dividend_receivable(
        self,
        trade_date: datetime,
        receivable: CashDividendReceivable,
    ) -> float:
        """结算单笔现金分红应收。"""

        self.cash += receivable.amount
        self.register_corporate_action_record(
            self._build_corporate_action_record(
                trade_date=trade_date,
                code=receivable.code,
                event_type="cash_dividend",
                stage="settled",
                operate_date=receivable.operate_date,
                settle_date=receivable.settle_date,
                eligible_quantity=receivable.eligible_quantity,
                cash_amount=receivable.amount,
                metadata=receivable.metadata,
            )
        )
        return receivable.amount

    def _settle_stock_dividend_receivable(
        self,
        trade_date: datetime,
        receivable: StockDividendReceivable,
    ) -> int:
        """结算单笔送转股应收。"""

        position = self.positions.get(receivable.code)
        if position is None:
            self.positions[receivable.code] = self._build_bonus_position_from_receivable(receivable, trade_date)
        else:
            position.quantity += receivable.bonus_quantity
            position.share_cost_basis += receivable.allocated_cost_basis
            position.cum_bonus_quantity += receivable.bonus_quantity
            position.refresh_adjusted_avg_price()

        self.register_corporate_action_record(
            self._build_corporate_action_record(
                trade_date=trade_date,
                code=receivable.code,
                event_type="stock_dividend",
                stage="settled",
                operate_date=receivable.operate_date,
                settle_date=receivable.settle_date,
                eligible_quantity=receivable.eligible_quantity,
                bonus_quantity=receivable.bonus_quantity,
                allocated_cost_basis=receivable.allocated_cost_basis,
                metadata=receivable.metadata,
            )
        )
        return receivable.bonus_quantity

    def add_to_position(
        self,
        code: str,
        trade: TradeRecord,
        cash_delta: float,
        order_id: str,
        *,
        score: float | None = None,
        metadata: dict | None = None,
        target_exit_trade_date: datetime | None = None,
    ) -> None:
        """把同一只股票的加仓成交合并进已有持仓。"""

        position = self._require_open_position(code, action="add-on")

        # 同一只票的 pyramiding 加仓在账本里只保留一个聚合持仓。
        # 这样后续估值、止损和收益统计都会把它当作一笔持续演化的真实持仓。
        self.cash += cash_delta
        position.quantity += trade.quantity
        position.share_cost_basis = (position.share_cost_basis or 0.0) + trade.notional
        position.entry_transaction_cost += trade.commission + trade.tax
        position.last_price = trade.price
        if target_exit_trade_date is not None:
            position.target_exit_trade_date = max(position.target_exit_trade_date, target_exit_trade_date)
        if score is not None:
            position.score = score

        incoming_metadata = copy_metadata(metadata)
        position.metadata = self._merge_entry_metadata(position, trade, order_id, incoming_metadata)
        self._apply_stop_loss_from_metadata(position, incoming_metadata)
        self._apply_risk_budget_from_metadata(position, incoming_metadata, trade.quantity)
        self._update_position_price_extremes(position, trade.price)

        position.refresh_adjusted_avg_price()
        self.trades.append(trade)

    def close_position(
        self,
        code: str,
        trade: TradeRecord,
        cash_delta: float,
        order_id: str,
        exit_trade_index: int | None = None,
        *,
        exit_reason: str | None = None,
    ) -> None:
        """登记平仓成交，同时把持仓移入已平仓列表。"""

        self.cash += cash_delta
        position = self.positions.pop(code, None)
        if position is not None:
            position.close(
                trade_date=trade.trade_date,
                trade_time=trade.trade_time,
                trade_price=trade.price,
                close_order_id=order_id,
                total_sell_cost=trade.commission + trade.tax,
                exit_trade_index=exit_trade_index,
                exit_reason=exit_reason,
            )
            self.closed_positions.append(position)
        self.trades.append(trade)

    def reduce_position(
        self,
        code: str,
        trade: TradeRecord,
        cash_delta: float,
        order_id: str,
        *,
        reduce_quantity: int,
        exit_trade_index: int | None = None,
        exit_reason: str | None = None,
    ) -> None:
        """按指定股数部分减仓，并保留剩余仓位继续持有。"""

        position = self._require_open_position(code, action="reduction")
        if reduce_quantity <= 0 or reduce_quantity > position.quantity:
            raise ValueError(f"invalid reduce quantity for {code}: {reduce_quantity}")
        if reduce_quantity == position.quantity:
            self.close_position(
                code,
                trade,
                cash_delta,
                order_id,
                exit_trade_index=exit_trade_index,
                exit_reason=exit_reason,
            )
            return

        self.cash += cash_delta

        reduced_position, reduced_cost_basis, reduced_entry_cost = self._build_partial_closed_position(
            position,
            trade,
            order_id,
            reduce_quantity=reduce_quantity,
            exit_trade_index=exit_trade_index,
            exit_reason=exit_reason,
        )
        self.closed_positions.append(reduced_position)

        position.quantity -= reduce_quantity
        position.share_cost_basis = max(float(position.share_cost_basis or 0.0) - reduced_cost_basis, 0.0)
        position.entry_transaction_cost = max(float(position.entry_transaction_cost or 0.0) - reduced_entry_cost, 0.0)
        position.last_price = trade.price
        position.refresh_adjusted_avg_price()

        self.trades.append(trade)

    def capture_eligible_quantity_snapshot(self, codes: list[str] | None = None) -> dict[str, int]:
        """记录开盘前资格股数快照，用于判断当天是否享有分红送转。"""

        target_codes = set(codes or self.positions.keys())
        return {
            code: int(position.quantity)
            for code, position in self.positions.items()
            if code in target_codes
        }

    def create_cash_dividend_receivable(
        self,
        position: StockPosition,
        event: CorporateActionEvent,
        eligible_quantity: int,
        trade_date: datetime,
    ) -> CashDividendReceivable | None:
        """创建现金分红应收，并把对应经济成本从原持仓中拆出。"""

        if eligible_quantity <= 0 or event.cash_dividend_per_share <= 0:
            return None

        amount = eligible_quantity * event.cash_dividend_per_share
        receivable = CashDividendReceivable(
            code=position.code,
            operate_date=event.operate_date,
            settle_date=event.settle_date,
            eligible_quantity=eligible_quantity,
            cash_dividend_per_share=event.cash_dividend_per_share,
            amount=amount,
            origin_position_id=position.position_id,
            signal_date=position.signal_date,
            score=position.score,
            metadata=merge_metadata(position.metadata, event.metadata),
        )
        self.cash_dividend_receivables.append(receivable)

        # 分红会降低老持仓的经济成本，但现金要到到账日才真正变成可用现金。
        position.cum_cash_dividend += amount
        position.share_cost_basis = max(position.share_cost_basis - amount, 0.0)
        position.refresh_adjusted_avg_price()

        self.register_corporate_action_record(
            self._build_corporate_action_record(
                trade_date=trade_date,
                code=position.code,
                event_type=event.event_type,
                stage="receivable_created",
                operate_date=event.operate_date,
                settle_date=event.settle_date,
                eligible_quantity=eligible_quantity,
                cash_amount=amount,
                note=event.raw_text,
                metadata=event.metadata,
            )
        )
        return receivable

    def create_stock_dividend_receivable(
        self,
        position: StockPosition,
        event: CorporateActionEvent,
        eligible_quantity: int,
        trade_date: datetime,
    ) -> StockDividendReceivable | None:
        """创建送转股应收，并把对应成本从原持仓中拆出。"""

        if eligible_quantity <= 0 or event.stock_dividend_ratio <= 0:
            return None

        bonus_quantity = int(eligible_quantity * event.stock_dividend_ratio)
        if bonus_quantity <= 0:
            return None

        # 送转后总成本不变，只是要重新摊到“原股 + 应收新股”上。
        combined_quantity = position.quantity + bonus_quantity
        adjusted_unit_cost = position.share_cost_basis / combined_quantity if combined_quantity > 0 else 0.0
        allocated_cost_basis = adjusted_unit_cost * bonus_quantity

        receivable = StockDividendReceivable(
            code=position.code,
            operate_date=event.operate_date,
            settle_date=event.settle_date,
            eligible_quantity=eligible_quantity,
            stock_dividend_ratio=event.stock_dividend_ratio,
            bonus_quantity=bonus_quantity,
            allocated_cost_basis=allocated_cost_basis,
            origin_position_id=position.position_id,
            signal_date=position.signal_date,
            score=position.score,
            metadata=merge_metadata(position.metadata, event.metadata),
        )
        self.stock_dividend_receivables.append(receivable)

        position.share_cost_basis = max(position.share_cost_basis - allocated_cost_basis, 0.0)
        position.refresh_adjusted_avg_price()

        self.register_corporate_action_record(
            self._build_corporate_action_record(
                trade_date=trade_date,
                code=position.code,
                event_type=event.event_type,
                stage="receivable_created",
                operate_date=event.operate_date,
                settle_date=event.settle_date,
                eligible_quantity=eligible_quantity,
                bonus_quantity=bonus_quantity,
                allocated_cost_basis=allocated_cost_basis,
                note=event.raw_text,
                metadata=event.metadata,
            )
        )
        return receivable

    def settle_cash_dividends(self, trade_date: datetime) -> float:
        """把今天到账的现金分红转入可用现金。"""

        settled_amount = 0.0
        remaining: list[CashDividendReceivable] = []
        for receivable in self.cash_dividend_receivables:
            if receivable.settle_date <= trade_date:
                settled_amount += self._settle_cash_dividend_receivable(trade_date, receivable)
            else:
                remaining.append(receivable)
        self.cash_dividend_receivables = remaining
        return settled_amount

    def settle_stock_dividends(self, trade_date: datetime) -> int:
        """把今天上市流通的送转股份并入持仓。"""

        settled_quantity = 0
        remaining: list[StockDividendReceivable] = []
        for receivable in self.stock_dividend_receivables:
            if receivable.settle_date <= trade_date:
                settled_quantity += self._settle_stock_dividend_receivable(trade_date, receivable)
            else:
                remaining.append(receivable)
        self.stock_dividend_receivables = remaining
        return settled_quantity

    def valuation_codes(self) -> list[str]:
        """返回收盘估值需要的全部股票代码：持仓 + 未到账送转股应收。"""

        return sorted(
            set(self.positions.keys()) | {receivable.code for receivable in self.stock_dividend_receivables}
        )

    def receivable_market_value(self, close_map: dict[str, float]) -> tuple[float, float]:
        """估算未到账现金和未上市送转股份的权益价值。"""

        cash_receivable_value = sum(receivable.amount for receivable in self.cash_dividend_receivables)
        stock_receivable_value = 0.0
        for receivable in self.stock_dividend_receivables:
            stock_receivable_value += close_map.get(receivable.code, 0.0) * receivable.bonus_quantity
        return cash_receivable_value, stock_receivable_value

    def mark_to_market(self, trade_date: datetime, close_map: dict[str, float]) -> None:
        """按收盘价更新持仓浮盈浮亏，并把应收也纳入权益曲线。"""

        market_value = 0.0
        for code, position in self.positions.items():
            if code in close_map:
                position.last_price = close_map[code]
            market_value += position.market_value
        cash_receivable_value, stock_receivable_value = self.receivable_market_value(close_map)
        self.equity_curve.append(
            EquityPoint(
                trade_date=trade_date,
                cash=self.cash,
                market_value=market_value,
                total_equity=self.cash + market_value + cash_receivable_value + stock_receivable_value,
                position_count=len(self.positions),
                cash_receivable_value=cash_receivable_value,
                stock_receivable_value=stock_receivable_value,
            )
        )
