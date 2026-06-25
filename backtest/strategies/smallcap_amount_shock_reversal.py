from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from backtest.strategies.base import SmallCapSignalTableStrategy
from backtest.strategies.models import DailyCandidate
from backtest.utils.datetime_utils import to_pydatetime


class SmallCapAmountShockReversalStrategy(SmallCapSignalTableStrategy):
    """小市值高放量反转策略。

    策略只验证研究结论中的两个核心变量：高放量冲击 amount_expand 和 10 日反转 ret_10d。
    不复用既有小市值流动性策略的过滤/排序逻辑，避免把旧策略收益来源混进这次验证。
    """

    def __init__(
        self,
        *,
        top_k: int = 10,
        hold_days: int = 10,
        rebalance_every_n_trade_days: int = 10,
        min_listing_trade_days: int = 120,
        candidate_pool_size: int = 150,
        cap_field: str = "liqaMV",
        amount_fast_window: int = 5,
        amount_slow_window: int = 20,
        ret_window: int = 10,
        group_count: int = 5,
        amount_keep_groups: Sequence[int] | str | None = (5,),
        ret_keep_groups: Sequence[int] | str | None = (5,),
        min_research_ret_10d: float | None = None,
        selection_sort: str = "ret_desc",
        signal_price_mode: str = "hfq",
        st_lookback_trade_days: int | None = 100,
        min_signal_close_price: float | None = 1.5,
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.hold_days = int(hold_days)
        self.rebalance_every_n_trade_days = int(rebalance_every_n_trade_days)
        self.min_listing_trade_days = int(min_listing_trade_days)
        self.candidate_pool_size = int(candidate_pool_size)
        self.cap_field = str(cap_field)
        self.amount_fast_window = int(amount_fast_window)
        self.amount_slow_window = int(amount_slow_window)
        self.ret_window = int(ret_window)
        self.group_count = int(group_count)
        self.amount_keep_groups = self._normalize_keep_groups(amount_keep_groups)
        self.ret_keep_groups = self._normalize_keep_groups(ret_keep_groups)
        self.min_research_ret_10d = None if min_research_ret_10d is None else float(min_research_ret_10d)
        self.selection_sort = str(selection_sort or "ret_desc").strip().lower()
        self.signal_price_mode = str(signal_price_mode)
        self.st_lookback_trade_days = None if st_lookback_trade_days is None else int(st_lookback_trade_days)
        self.min_signal_close_price = None if min_signal_close_price is None else float(min_signal_close_price)
        self._validate_parameters()

    def required_feature_fields(self) -> Sequence[str]:
        return ("code", "date", self.cap_field)

    @property
    def amount_fast_col(self) -> str:
        return f"avg_amount_{self.amount_fast_window}d"

    @property
    def amount_slow_col(self) -> str:
        return f"avg_amount_{self.amount_slow_window}d"

    @staticmethod
    def _normalize_keep_groups(raw_groups: Sequence[int] | str | None) -> tuple[int, ...]:
        if raw_groups is None:
            return ()
        if isinstance(raw_groups, str):
            groups = [int(part.strip()) for part in raw_groups.split(",") if part.strip()]
        else:
            groups = [int(group) for group in raw_groups]
        return tuple(sorted(set(groups)))

    def _validate_parameters(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.rebalance_every_n_trade_days < 1:
            raise ValueError("rebalance_every_n_trade_days must be >= 1")
        if self.candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be >= 1")
        if self.amount_fast_window < 1:
            raise ValueError("amount_fast_window must be >= 1")
        if self.amount_slow_window < self.amount_fast_window:
            raise ValueError("amount_slow_window must be >= amount_fast_window")
        if self.ret_window < 1:
            raise ValueError("ret_window must be >= 1")
        if self.group_count < 2:
            raise ValueError("group_count must be >= 2")
        for name, groups in {"amount_keep_groups": self.amount_keep_groups, "ret_keep_groups": self.ret_keep_groups}.items():
            invalid = [group for group in groups if group < 1 or group > self.group_count]
            if invalid:
                raise ValueError(f"{name} must be within 1..{self.group_count}: {invalid}")
        if self.min_research_ret_10d is not None and self.min_research_ret_10d < 0:
            raise ValueError("min_research_ret_10d must be >= 0")
        if self.st_lookback_trade_days is not None and self.st_lookback_trade_days < 0:
            raise ValueError("st_lookback_trade_days must be >= 0")
        if self.min_signal_close_price is not None and self.min_signal_close_price <= 0:
            raise ValueError("min_signal_close_price must be positive")
        if self.selection_sort not in {"ret_desc", "cap_asc"}:
            raise ValueError("selection_sort must be one of: ret_desc, cap_asc")

    def _warmup_trade_dates(self, full_trade_calendar: pd.DatetimeIndex, signal_start: datetime) -> datetime:
        signal_start_ts = pd.Timestamp(signal_start)
        start_index = int(full_trade_calendar.searchsorted(signal_start_ts, side="left"))
        st_window = self.st_lookback_trade_days or 0
        warmup_window = max(self.amount_slow_window, self.ret_window, st_window) + 5
        warmup_index = max(0, start_index - warmup_window)
        return to_pydatetime(full_trade_calendar[warmup_index])

    def _load_or_build_cached_strategy_frame(self, data_portal, stage: str, payload: dict, builder) -> pd.DataFrame:
        frame_cache = getattr(data_portal, "frame_cache", None)
        if frame_cache is None:
            return builder()
        return frame_cache.load_or_build_frame(stage, payload, builder)

    def _prepare_candidate_pool_frame(
        self,
        data_portal,
        trade_dates: Sequence[datetime],
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        base_frame, full_trade_calendar = self._prepare_smallcap_feature_frame(
            data_portal,
            trade_dates,
            feature_fields=["code", "date", self.cap_field],
            cap_field=self.cap_field,
            rebalance_every_n_trade_days=self.rebalance_every_n_trade_days,
            min_listing_trade_days=self.min_listing_trade_days,
            rebalance_only=True,
        )
        if base_frame.empty:
            return pd.DataFrame(), full_trade_calendar
        candidate_pool = (
            base_frame.sort_values(["trade_date", self.cap_field, "code"], ascending=[True, True, True])
            .groupby("trade_date", group_keys=False)
            .head(self.candidate_pool_size)
            .reset_index(drop=True)
        )
        return candidate_pool, full_trade_calendar

    def _prepare_cached_candidate_pool_frame(
        self,
        data_portal,
        trade_dates: Sequence[datetime],
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        # full_trade_calendar 不是表格数据，不能直接塞进 FrameCache；这里只缓存更重的日线指标。
        return self._prepare_candidate_pool_frame(data_portal, trade_dates)

    def _prepare_indicator_frame(
        self,
        data_portal,
        *,
        codes: Sequence[str],
        warmup_start: datetime,
        signal_end: datetime,
        research_store=None,
    ) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame()
        frame_cache = getattr(data_portal, "frame_cache", None)
        if frame_cache is not None:
            payload = {
                "feature_formula_version": "smallcap_amount_shock_indicator_v2",
                "warmup_start": warmup_start,
                "signal_end": signal_end,
                "codes_signature": frame_cache.codes_signature(codes),
                "amount_fast_window": self.amount_fast_window,
                "amount_slow_window": self.amount_slow_window,
                "ret_window": self.ret_window,
                "signal_price_mode": self.signal_price_mode,
                "st_lookback_trade_days": self.st_lookback_trade_days,
                "min_signal_close_price": self.min_signal_close_price,
            }
            return frame_cache.load_or_build_frame(
                "smallcap_amount_shock_indicator",
                payload,
                lambda: self._build_indicator_frame(
                    data_portal,
                    codes=codes,
                    warmup_start=warmup_start,
                    signal_end=signal_end,
                    research_store=research_store,
                ),
            )
        return self._build_indicator_frame(
            data_portal,
            codes=codes,
            warmup_start=warmup_start,
            signal_end=signal_end,
            research_store=research_store,
        )

    def _build_indicator_frame(
        self,
        data_portal,
        *,
        codes: Sequence[str],
        warmup_start: datetime,
        signal_end: datetime,
        research_store=None,
    ) -> pd.DataFrame:
        history_end = to_pydatetime(pd.Timestamp(signal_end) + pd.Timedelta(days=1))
        fields = ["code", "trade_date", "close", "amount", "isST"]
        if research_store is not None:
            daily_history = research_store.load_daily_history(
                warmup_start,
                history_end,
                codes=codes,
                fields=fields,
                include_stopped=False,
                price_mode=self.signal_price_mode,
                batch_size=1000,
            )
        else:
            daily_history = data_portal.get_daily_history(
                warmup_start,
                history_end,
                codes=codes,
                fields=fields,
                include_stopped=False,
                price_mode=self.signal_price_mode,
                batch_size=1000,
            )
        if daily_history.empty:
            return pd.DataFrame()

        daily_history = daily_history.copy()
        daily_history["trade_date"] = pd.to_datetime(daily_history["trade_date"])
        daily_history = daily_history.sort_values(["code", "trade_date"]).reset_index(drop=True)

        grouped = daily_history.groupby("code", sort=False)
        daily_history[self.amount_fast_col] = grouped["amount"].transform(
            lambda values: values.rolling(self.amount_fast_window, min_periods=self.amount_fast_window).mean()
        )
        daily_history[self.amount_slow_col] = grouped["amount"].transform(
            lambda values: values.rolling(self.amount_slow_window, min_periods=self.amount_slow_window).mean()
        )
        amount_slow = daily_history[self.amount_slow_col]
        daily_history["amount_expand"] = daily_history[self.amount_fast_col] / amount_slow.where(amount_slow != 0)
        shifted_close = grouped["close"].shift(self.ret_window)
        daily_history["ret_10d"] = daily_history["close"] / shifted_close.where(shifted_close != 0) - 1.0
        daily_history["research_ret_10d"] = -daily_history["ret_10d"]
        daily_history["isST"] = daily_history["isST"].fillna(False).astype(bool)
        lookback_window = self.st_lookback_trade_days or 0
        if lookback_window > 0:
            daily_history["is_st_lookback_flag"] = (
                grouped["isST"]
                .transform(lambda values: values.astype(float).rolling(lookback_window, min_periods=1).max())
                .fillna(0)
                .astype(bool)
            )
        else:
            daily_history["is_st_lookback_flag"] = False
        return daily_history

    def _assign_groups(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        result = frame.copy()
        grouped = result.groupby("trade_date", sort=False)
        if "amount_group" not in result.columns:
            result["amount_group"] = self._daily_group(grouped, "amount_expand", ascending=True)
        if "ret_group" not in result.columns:
            result["ret_group"] = self._daily_group(grouped, "research_ret_10d", ascending=True)
        return result

    def _daily_group(self, grouped, column: str, *, ascending: bool) -> pd.Series:
        count = grouped[column].transform("count")
        rank = grouped[column].rank(method="first", ascending=ascending)
        bucket = np.floor(((rank - 1.0) * self.group_count) / count) + 1.0
        return pd.array(pd.Series(bucket, index=rank.index).where(count >= self.group_count), dtype="Int8")

    def _apply_signal_filters(self, frame: pd.DataFrame) -> pd.DataFrame:
        filtered = frame.copy()
        if self.amount_keep_groups:
            filtered = filtered[filtered["amount_group"].isin(self.amount_keep_groups)].copy()
        if self.ret_keep_groups:
            filtered = filtered[filtered["ret_group"].isin(self.ret_keep_groups)].copy()
        # 阈值放在分组之后，避免改变 ret_group 的历史实验口径。
        if self.min_research_ret_10d is not None:
            filtered = filtered[filtered["research_ret_10d"] >= self.min_research_ret_10d].copy()
        return filtered

    def _build_signal_table(self, frame: pd.DataFrame) -> dict[datetime, pd.DataFrame]:
        if frame.empty:
            return {}
        required_factor_columns = []
        if self.amount_keep_groups:
            required_factor_columns.append("amount_expand")
        if self.ret_keep_groups or self.min_research_ret_10d is not None:
            required_factor_columns.extend(["ret_10d", "research_ret_10d"])
        final_frame = frame.dropna(subset=required_factor_columns).copy() if required_factor_columns else frame.copy()
        final_frame = self._assign_groups(final_frame)
        final_frame = self._apply_signal_filters(final_frame)
        if final_frame.empty:
            return {}
        if self.selection_sort == "ret_desc" and self.ret_keep_groups:
            final_frame = final_frame.sort_values(
                ["trade_date", "research_ret_10d", self.cap_field],
                ascending=[True, False, True],
            )
        else:
            final_frame = final_frame.sort_values(["trade_date", self.cap_field], ascending=[True, True])
        return self.group_signal_table(final_frame, limit=self.top_k)

    def prepare(self, data_portal, trade_dates: Sequence[datetime], *, research_store=None) -> None:
        self.reset_signal_table()
        if not trade_dates:
            return

        signal_start = to_pydatetime(trade_dates[0])
        signal_end = to_pydatetime(trade_dates[-1])

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

        frame = candidate_pool_frame.merge(indicator_frame, on=["code", "trade_date"], how="inner")
        if frame.empty:
            return
        frame = frame[frame["isST"].fillna(False) == False].copy()
        if self.st_lookback_trade_days:
            frame = frame[frame["is_st_lookback_flag"].fillna(False) == False].copy()
        if self.min_signal_close_price is not None:
            frame = frame[pd.to_numeric(frame["close"], errors="coerce") > self.min_signal_close_price].copy()
        self.set_signal_table(self._build_signal_table(frame))

    @staticmethod
    def _safe_float_metadata(metadata: dict, key: str, value) -> None:
        if pd.notna(value):
            metadata[key] = float(value)

    def generate_candidates(
        self,
        signal_date: datetime,
        feature_slice: pd.DataFrame,
        portfolio_snapshot: dict,
    ) -> list[DailyCandidate]:
        frame = self.signal_frame(signal_date)
        if frame is None or frame.empty:
            return []

        candidates: list[DailyCandidate] = []
        for row in frame.itertuples(index=False):
            metadata = {
                self.cap_field: float(getattr(row, self.cap_field)),
                "close": float(row.close),
            }
            amount_group_value = getattr(row, "amount_group", None)
            ret_group_value = getattr(row, "ret_group", None)
            if pd.notna(amount_group_value):
                metadata["amount_group"] = int(amount_group_value)
            if pd.notna(ret_group_value):
                metadata["ret_group"] = int(ret_group_value)
            for key in ("amount_expand", "ret_10d", "research_ret_10d", self.amount_fast_col, self.amount_slow_col):
                self._safe_float_metadata(metadata, key, getattr(row, key, None))
            if self.min_research_ret_10d is not None:
                metadata["min_research_ret_10d"] = self.min_research_ret_10d
            if self.st_lookback_trade_days:
                metadata["st_lookback_trade_days"] = self.st_lookback_trade_days
            if self.min_signal_close_price is not None:
                metadata["min_signal_close_price"] = self.min_signal_close_price
            candidates.append(
                DailyCandidate(
                    signal_date=signal_date,
                    code=str(row.code),
                    score=float(getattr(row, "research_ret_10d"))
                    if self.ret_keep_groups or self.min_research_ret_10d is not None
                    else float(getattr(row, self.cap_field)),
                    hold_days=self.hold_days,
                    metadata=metadata,
                )
            )
        return candidates
