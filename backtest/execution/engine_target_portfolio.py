"""执行引擎里的 target-portfolio 调仓辅助逻辑。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any
import uuid

import pandas as pd

from backtest.portfolio import PortfolioLedger, StockOrder
from backtest.strategies import DailyCandidate
from backtest.utils.frame_utils import first_sorted_row
from backtest.utils.metadata import copy_metadata


class MissingTargetExposureError(RuntimeError):
    """target-portfolio 策略缺少目标暴露时抛出的显式错误。"""


class EngineTargetPortfolioMixin:
    def _execution_reference_price(self, frame: pd.DataFrame) -> float | None:
        """取调仓目标股数估算用的参考价格。"""

        row = first_sorted_row(frame)
        if row is None:
            return None
        for field in ("open", getattr(self.execution_model, "price_field", None), "close"):
            if not field:
                continue
            if field in row and pd.notna(row[field]):
                price = float(row[field])
                if price > 0:
                    return price
        return None

    def _build_target_specs_from_candidates(
        self,
        trade_dates: list[datetime],
        trade_index: int,
        target_market_value: float,
        candidates: list[DailyCandidate],
    ) -> dict[str, dict[str, Any]]:
        """把调仓候选转换成目标组合规格。"""

        if target_market_value <= 0 or not candidates:
            return {}

        unique_candidates: list[DailyCandidate] = []
        seen_codes: set[str] = set()
        for candidate in candidates:
            if candidate.code in seen_codes:
                continue
            seen_codes.add(candidate.code)
            unique_candidates.append(candidate)
        if not unique_candidates:
            return {}

        raw_weights: list[float] = []
        has_explicit_weight = True
        for candidate in unique_candidates:
            weight = candidate.metadata.get("target_weight")
            if weight is None:
                has_explicit_weight = False
                break
            try:
                weight_value = float(weight)
            except (TypeError, ValueError):
                has_explicit_weight = False
                break
            if weight_value <= 0:
                has_explicit_weight = False
                break
            raw_weights.append(weight_value)
        if not has_explicit_weight or sum(raw_weights) <= 0:
            raw_weights = [1.0] * len(unique_candidates)

        total_weight = float(sum(raw_weights))
        target_specs: dict[str, dict[str, Any]] = {}
        for candidate, raw_weight in zip(unique_candidates, raw_weights):
            metadata_exit_trade_date = candidate.metadata.get("target_exit_trade_date")
            if isinstance(metadata_exit_trade_date, datetime):
                target_exit_trade_date = metadata_exit_trade_date
            else:
                target_exit_trade_date = self._resolve_exit_trade_date(
                    trade_dates,
                    trade_index,
                    candidate.hold_days,
                )
            target_specs[candidate.code] = {
                "candidate": candidate,
                "target_market_value": target_market_value * (raw_weight / total_weight),
                # target-portfolio 模式下它只用于调仓元数据刷新，不触发 scheduled forced exit。
                "target_exit_trade_date": target_exit_trade_date,
            }
        return target_specs

    def _build_target_specs_from_existing_positions(
        self,
        ledger: PortfolioLedger,
        target_market_value: float,
    ) -> dict[str, dict[str, Any]]:
        """按当前老仓市值占比缩放到新的目标总暴露。"""

        current_market_value = sum(position.market_value for position in ledger.positions.values())
        if current_market_value <= 0 or target_market_value < 0:
            return {}

        target_specs: dict[str, dict[str, Any]] = {}
        for position in ledger.positions.values():
            weight = position.market_value / current_market_value if current_market_value > 0 else 0.0
            target_specs[position.code] = {
                "candidate": None,
                "target_market_value": target_market_value * weight,
                "target_exit_trade_date": position.target_exit_trade_date,
            }
        return target_specs

    def _normalize_target_quantity(
        self,
        target_market_value: float,
        reference_price: float | None,
    ) -> tuple[int, int]:
        """按参考价把目标市值换算成整手股数，不足一手直接归零。"""

        if target_market_value <= 0 or reference_price is None or reference_price <= 0:
            return 0, 0
        raw_quantity = int(float(target_market_value) // float(reference_price))
        normalized_quantity = (raw_quantity // self.config.lot_size) * self.config.lot_size
        if normalized_quantity < self.config.lot_size:
            normalized_quantity = 0
        return raw_quantity, normalized_quantity

    @staticmethod
    def _strip_rebalance_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        """去掉调仓链路里容易混淆时点语义的旧字段。"""

        cleaned = dict(metadata or {})
        for key in (
            "regime_state",
            "capital_scale",
            "repair_active",
            "repair_trigger",
            "repair_strong",
            "target_exposure",
            "requested_quantity",
            "execution_regime_state",
            "execution_target_exposure",
            "execution_capital_scale",
        ):
            cleaned.pop(key, None)
        return cleaned

    @staticmethod
    def _build_execution_metadata(regime_state: str, target_exposure: float) -> dict[str, Any]:
        """构造执行日口径的调仓元数据。"""

        exposure = float(target_exposure)
        return {
            "execution_regime_state": str(regime_state),
            "execution_target_exposure": exposure,
            "execution_capital_scale": exposure,
        }

    def _build_rebalance_metadata(
        self,
        base_metadata: dict[str, Any] | None,
        *,
        regime_state: str,
        target_exposure: float,
        target_quantity: int,
        current_quantity: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """统一构造净额调仓订单与持仓的审计元数据。"""

        metadata = self._strip_rebalance_metadata(base_metadata)
        metadata.update(self._build_execution_metadata(regime_state, target_exposure))
        metadata["rebalance_mode"] = "net_delta"
        metadata["target_quantity"] = int(target_quantity)
        metadata["current_quantity"] = int(current_quantity)
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    def _build_target_quantity_map(
        self,
        target_specs: dict[str, dict[str, Any]],
        entry_frames: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
        """把目标市值换算成目标股数，并保留复盘明细。"""

        target_quantities: dict[str, int] = {}
        target_details: dict[str, dict[str, Any]] = {}
        for code, spec in target_specs.items():
            target_market_value = float(spec.get("target_market_value") or 0.0)
            frame = entry_frames.get(code)
            reference_price = self._execution_reference_price(frame)
            raw_quantity, target_quantity = self._normalize_target_quantity(target_market_value, reference_price)
            target_quantities[code] = target_quantity
            target_details[code] = {
                "target_market_value": target_market_value,
                "reference_price": reference_price,
                "raw_target_quantity": raw_quantity,
                "target_quantity": target_quantity,
            }
        return target_quantities, target_details

    def _refresh_position_after_rebalance(
        self,
        position,
        candidate: DailyCandidate,
        target_exit_trade_date: datetime,
        trade_date: datetime,
        *,
        regime_state: str,
        target_exposure: float,
    ) -> None:
        """对调仓后仍保留的重合持仓，刷新其最新信号和退出日。"""

        position.target_exit_trade_date = target_exit_trade_date
        position.signal_date = candidate.signal_date
        if candidate.score is not None:
            position.score = candidate.score
        merged_metadata = self._strip_rebalance_metadata(position.metadata)
        merged_metadata.update(self._strip_rebalance_metadata(candidate.metadata))
        merged_metadata.update(self._build_execution_metadata(regime_state, target_exposure))
        merged_metadata["rebalance_mode"] = "net_delta"
        merged_metadata["last_rebalance_signal_date"] = candidate.signal_date
        merged_metadata["last_rebalance_trade_date"] = trade_date
        position.metadata = merged_metadata
        if candidate.metadata.get("initial_stop_loss") is not None:
            position.initial_stop_loss = float(candidate.metadata["initial_stop_loss"])
        if candidate.metadata.get("current_stop_loss") is not None:
            position.current_stop_loss = float(candidate.metadata["current_stop_loss"])

    def _execute_target_portfolio_rebalance(
        self,
        ledger: PortfolioLedger,
        trade_index: int,
        trade_date: datetime,
        target_specs: dict[str, dict[str, Any]],
        *,
        reason: str,
        log_prefix: str,
        regime_state: str,
        target_exposure: float,
    ) -> None:
        """执行组合目标市值到目标股数的净额调仓。"""

        union_codes = sorted(set(ledger.positions.keys()) | set(target_specs.keys()))
        if not union_codes:
            return

        entry_frames = self._load_execution_frames(sorted(target_specs.keys()), trade_date, for_exit=False) if target_specs else {}
        exit_frames = self._load_execution_frames(union_codes, trade_date, for_exit=True)
        target_quantities, target_quantity_details = self._build_target_quantity_map(target_specs, entry_frames)

        for code, detail in target_quantity_details.items():
            if detail["target_market_value"] <= 0:
                continue
            if detail["reference_price"] is None or float(detail["reference_price"]) <= 0:
                continue
            if int(detail["target_quantity"]) > 0:
                continue
            self._log_event(
                "rebalance_target_below_lot_skipped",
                trade_date=trade_date,
                code=code,
                target_market_value=detail["target_market_value"],
                reference_price=detail["reference_price"],
                raw_target_quantity=detail["raw_target_quantity"],
                lot_size=self.config.lot_size,
                regime_state=regime_state,
                target_exposure=target_exposure,
            )

        # 先卖后买，保证现金先回笼，再去补目标组合的差额。
        for code in union_codes:
            position = ledger.positions.get(code)
            if position is None:
                continue
            current_quantity = int(position.quantity)
            target_quantity = int(target_quantities.get(code, 0))
            sell_quantity = current_quantity - target_quantity
            if sell_quantity <= 0:
                continue

            frame = exit_frames.get(code)
            exit_order = StockOrder.create(
                code=code,
                side="SELL",
                signal_date=position.signal_date,
                scheduled_trade_date=trade_date,
                execution_model=self.execution_model.data_frequency,
                reason=reason,
                requested_quantity=sell_quantity,
                score=position.score,
                metadata=self._build_rebalance_metadata(
                    position.metadata,
                    regime_state=regime_state,
                    target_exposure=target_exposure,
                    target_quantity=target_quantity,
                    current_quantity=current_quantity,
                ),
            )
            if frame is None or frame.empty:
                exit_order.mark_skipped("no_execution_data")
                ledger.register_order(exit_order)
                self._log_event(
                    f"{log_prefix}_sell_skipped",
                    trade_date=trade_date,
                    code=code,
                    skip_reason="no_execution_data",
                    requested_quantity=sell_quantity,
                )
                continue

            slice_position = replace(position, quantity=sell_quantity)
            trade, cash_delta = self.execution_model.execute_exit(
                slice_position,
                trade_date,
                frame,
                self.config,
                reason=reason,
            )
            if trade is None:
                exit_order.mark_skipped("no_fill")
                ledger.register_order(exit_order)
                self._log_event(
                    f"{log_prefix}_sell_skipped",
                    trade_date=trade_date,
                    code=code,
                    skip_reason="no_fill",
                    requested_quantity=sell_quantity,
                )
                continue

            exit_order.mark_filled(trade)
            ledger.register_order(exit_order)
            if sell_quantity == current_quantity:
                ledger.close_position(
                    code,
                    trade,
                    cash_delta,
                    exit_order.order_id,
                    exit_trade_index=trade_index,
                    exit_reason=reason,
                )
            else:
                ledger.reduce_position(
                    code,
                    trade,
                    cash_delta,
                    exit_order.order_id,
                    reduce_quantity=sell_quantity,
                    exit_trade_index=trade_index,
                    exit_reason=reason,
                )
            self._log_event(
                f"{log_prefix}_sell_filled",
                trade_date=trade_date,
                code=code,
                quantity=trade.quantity,
                price=trade.price,
                target_quantity=target_quantity,
                regime_state=regime_state,
                target_exposure=target_exposure,
                cash=ledger.cash,
            )

        for code, spec in target_specs.items():
            target_quantity = int(target_quantities.get(code, 0))
            if target_quantity <= 0:
                continue
            frame = entry_frames.get(code)
            current_position = ledger.positions.get(code)
            current_quantity = int(current_position.quantity) if current_position is not None else 0
            buy_quantity = target_quantity - current_quantity
            candidate = spec.get("candidate")
            target_exit_trade_date = spec.get("target_exit_trade_date")
            if buy_quantity <= 0:
                if current_position is not None and candidate is not None and isinstance(target_exit_trade_date, datetime):
                    self._refresh_position_after_rebalance(
                        current_position,
                        candidate,
                        target_exit_trade_date,
                        trade_date,
                        regime_state=regime_state,
                        target_exposure=target_exposure,
                    )
                continue

            base_signal_metadata = (
                copy_metadata(candidate.metadata)
                if candidate is not None
                else copy_metadata(current_position.metadata) if current_position is not None else {}
            )
            reference_price = self._execution_reference_price(frame) if frame is not None and not frame.empty else None
            requested_budget = 0.0
            if reference_price is not None and reference_price > 0:
                requested_budget = buy_quantity * reference_price * (1 + self.config.buy_commission_rate)

            order_signal_date = candidate.signal_date if candidate is not None else trade_date
            order_score = candidate.score if candidate is not None else (current_position.score if current_position is not None else None)
            order_metadata = self._build_rebalance_metadata(
                base_signal_metadata,
                regime_state=regime_state,
                target_exposure=target_exposure,
                target_quantity=target_quantity,
                current_quantity=current_quantity,
            )

            if frame is None or frame.empty:
                entry_order = StockOrder.create(
                    code=code,
                    side="BUY",
                    signal_date=order_signal_date,
                    scheduled_trade_date=trade_date,
                    execution_model=self.execution_model.data_frequency,
                    reason=reason,
                    requested_budget=requested_budget,
                    requested_quantity=buy_quantity,
                    score=order_score,
                    metadata=order_metadata,
                )
                entry_order.mark_skipped("no_execution_data")
                ledger.register_order(entry_order)
                self._log_event(
                    f"{log_prefix}_buy_skipped",
                    trade_date=trade_date,
                    code=code,
                    skip_reason="no_execution_data",
                    requested_quantity=buy_quantity,
                    requested_budget=requested_budget,
                )
                continue

            candidate_metadata = self._strip_rebalance_metadata(base_signal_metadata)
            candidate_metadata["entry_reason"] = reason
            if candidate is None:
                candidate = DailyCandidate(
                    signal_date=trade_date,
                    code=code,
                    score=(current_position.score if current_position is not None else None),
                    hold_days=1,
                    metadata=candidate_metadata,
                )
            else:
                candidate = DailyCandidate(
                    signal_date=candidate.signal_date,
                    code=candidate.code,
                    score=candidate.score,
                    hold_days=candidate.hold_days,
                    metadata=candidate_metadata,
                )

            target_exit_date = target_exit_trade_date if isinstance(target_exit_trade_date, datetime) else trade_date
            entry_order = StockOrder.create(
                code=code,
                side="BUY",
                signal_date=candidate.signal_date,
                scheduled_trade_date=trade_date,
                execution_model=self.execution_model.data_frequency,
                reason=reason,
                requested_budget=requested_budget,
                requested_quantity=buy_quantity,
                score=candidate.score,
                metadata=order_metadata,
            )
            entry_result = self.execution_model.execute_entry_by_quantity(
                candidate,
                trade_date,
                frame,
                buy_quantity,
                ledger.cash,
                self.config,
                target_exit_date,
            )
            if entry_result.position is None or entry_result.trade is None:
                reject_reason = entry_result.reject_reason or "no_fill"
                entry_order.mark_skipped(reject_reason)
                ledger.register_order(entry_order)
                self._log_event(
                    f"{log_prefix}_buy_skipped",
                    trade_date=trade_date,
                    code=code,
                    skip_reason=reject_reason,
                    requested_quantity=buy_quantity,
                    requested_budget=requested_budget,
                )
                continue

            trade = entry_result.trade
            cash_delta = entry_result.cash_delta
            entry_order.mark_filled(trade)
            ledger.register_order(entry_order)
            existing_position = ledger.positions.get(code)
            if existing_position is not None:
                existing_position.last_price = trade.price
                ledger.add_to_position(
                    code,
                    trade,
                    cash_delta,
                    entry_order.order_id,
                    score=candidate.score,
                    metadata=copy_metadata(trade.metadata),
                    target_exit_trade_date=target_exit_date,
                )
                self._refresh_position_after_rebalance(
                    ledger.positions[code],
                    candidate,
                    target_exit_date,
                    trade_date,
                    regime_state=regime_state,
                    target_exposure=target_exposure,
                )
            else:
                position = entry_result.position
                position.position_id = uuid.uuid4().hex
                position.open_order_id = entry_order.order_id
                position.entry_transaction_cost = trade.commission + trade.tax
                position.entry_trade_index = trade_index
                self._apply_candidate_risk_to_position(position, candidate)
                if self.exit_policy is not None:
                    self.exit_policy.initialize_position(position)
                ledger.open_position(position, trade, cash_delta)
                self._refresh_position_after_rebalance(
                    ledger.positions[code],
                    candidate,
                    target_exit_date,
                    trade_date,
                    regime_state=regime_state,
                    target_exposure=target_exposure,
                )
            self._log_event(
                f"{log_prefix}_buy_filled",
                trade_date=trade_date,
                code=code,
                quantity=trade.quantity,
                price=trade.price,
                target_quantity=target_quantity,
                regime_state=regime_state,
                target_exposure=target_exposure,
                cash=ledger.cash,
            )

        for code, spec in target_specs.items():
            candidate = spec.get("candidate")
            target_exit_trade_date = spec.get("target_exit_trade_date")
            if candidate is None or not isinstance(target_exit_trade_date, datetime):
                continue
            position = ledger.positions.get(code)
            if position is None:
                continue
            self._refresh_position_after_rebalance(
                position,
                candidate,
                target_exit_trade_date,
                trade_date,
                regime_state=regime_state,
                target_exposure=target_exposure,
            )

    def _process_state_change_exposure_rebalance(
        self,
        ledger: PortfolioLedger,
        trade_index: int,
        trade_date: datetime,
        *,
        regime_state: str,
        target_exposure: float,
    ) -> None:
        """非调仓日发生状态切换时，只对老仓按当前市值同比例缩放。"""

        if not ledger.positions:
            self._log_event(
                "regime_state_change_skipped",
                trade_date=trade_date,
                regime_state=regime_state,
                target_exposure=target_exposure,
                skip_reason="no_positions",
            )
            return

        total_equity = self._portfolio_equity(ledger, trade_date)
        current_market_value = sum(position.market_value for position in ledger.positions.values())
        target_market_value = total_equity * target_exposure
        target_specs = self._build_target_specs_from_existing_positions(ledger, target_market_value)
        self._log_event(
            "regime_state_change_ready",
            trade_date=trade_date,
            regime_state=regime_state,
            target_exposure=target_exposure,
            current_market_value=current_market_value,
            target_market_value=target_market_value,
            position_count=len(ledger.positions),
        )
        self._execute_target_portfolio_rebalance(
            ledger,
            trade_index,
            trade_date,
            target_specs,
            reason="regime_state_change_rebalance",
            log_prefix="regime_state_change",
            regime_state=regime_state,
            target_exposure=target_exposure,
        )

    def _process_rebalance_day_target_portfolio(
        self,
        ledger: PortfolioLedger,
        trade_dates: list[datetime],
        trade_index: int,
        trade_date: datetime,
        candidates: list[DailyCandidate],
        *,
        regime_state: str,
        target_exposure: float,
    ) -> None:
        """调仓日先确定目标总仓位，再对旧组合和新组合做净额调仓。"""

        total_equity = self._portfolio_equity(ledger, trade_date)
        target_market_value = total_equity * target_exposure
        target_specs = self._build_target_specs_from_candidates(
            trade_dates,
            trade_index,
            target_market_value,
            candidates,
        )
        self._log_event(
            "rebalance_day_target_ready",
            trade_date=trade_date,
            regime_state=regime_state,
            target_exposure=target_exposure,
            candidate_count=len(candidates),
            candidate_codes=[candidate.code for candidate in candidates],
            target_market_value=target_market_value,
            overlap_count=len(set(ledger.positions.keys()) & set(target_specs.keys())),
        )
        self._execute_target_portfolio_rebalance(
            ledger,
            trade_index,
            trade_date,
            target_specs,
            reason="rebalance_net_delta",
            log_prefix="rebalance_net_delta",
            regime_state=regime_state,
            target_exposure=target_exposure,
        )

    def _resolve_target_portfolio_regime_state(self, trade_date: datetime) -> tuple[str, bool]:
        """提取当前交易日的 regime_state，并标记是否实际取到 regime 记录。"""

        if hasattr(self.strategy, "regime_table"):
            regime = getattr(self.strategy, "regime_table", {}).get(trade_date)
            if regime is not None:
                return str(regime.get("regime_state", "unknown")), True
        return "unknown", False

    def _process_target_portfolio_day(
        self,
        ledger: PortfolioLedger,
        trade_dates: list[datetime],
        trade_index: int,
        trade_date: datetime,
        pending_entries: dict[datetime, list[DailyCandidate]],
    ) -> None:
        """按目标组合和目标暴露驱动调仓，不沿用 scheduled hold_days forced exit。"""

        regime_state, regime_available = self._resolve_target_portfolio_regime_state(trade_date)
        todays_pending_entries = pending_entries.get(trade_date, [])
        target_exposure = self.strategy.target_exposure(trade_date, ledger.snapshot())
        if target_exposure is None:
            raise MissingTargetExposureError(
                "target_exposure is missing for "
                f"trade_date={trade_date:%Y-%m-%d}, "
                f"strategy={type(self.strategy).__name__}, "
                f"pending_entry_count={len(todays_pending_entries)}, "
                f"regime_state={regime_state if regime_available else 'unavailable'}, "
                f"regime_state_available={regime_available}"
            )

        target_exposure = max(min(float(target_exposure), 1.0), 0.0)
        todays_candidates = pending_entries.pop(trade_date, [])
        if todays_candidates:
            self._process_rebalance_day_target_portfolio(
                ledger,
                trade_dates,
                trade_index,
                trade_date,
                todays_candidates,
                regime_state=regime_state,
                target_exposure=target_exposure,
            )
            return

        previous_trade_date = trade_dates[trade_index - 1] if trade_index > 0 else None
        if not self.strategy.should_rebalance_exposure(
            trade_date,
            previous_trade_date,
            ledger.snapshot(),
        ):
            self._log_event(
                "regime_state_unchanged",
                trade_date=trade_date,
                regime_state=regime_state,
                target_exposure=target_exposure,
                position_count=len(ledger.positions),
            )
            return

        self._process_state_change_exposure_rebalance(
            ledger,
            trade_index,
            trade_date,
            regime_state=regime_state,
            target_exposure=target_exposure,
        )
