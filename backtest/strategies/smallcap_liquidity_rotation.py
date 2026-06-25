from __future__ import annotations

from datetime import datetime
import math
from typing import Sequence

import pandas as pd

from backtest.strategies.base import SmallCapSignalTableStrategy
from backtest.strategies.models import DailyCandidate
from backtest.utils.datetime_utils import to_pydatetime


class SmallCapLiquidityRotationStrategy(SmallCapSignalTableStrategy):
    def __init__(
        self,
        *,
        top_k: int = 20,
        hold_days: int = 10,
        rebalance_every_n_trade_days: int = 10,
        min_listing_trade_days: int = 120,
        candidate_pool_size: int = 200,
        cap_field: str = "liqaMV",
        liquidity_window: int = 20,
        min_avg_amount: float | None = 30_000_000,
        min_avg_turn: float | None = 1.0,
        exclude_bottom_liquidity_pct: float = 0.15,
        exclude_bottom_liquidity_metric: str = "amount",
        min_close_price: float | None = None,
        factor_filter_enabled: bool = False,
        hhv_window: int = 60,
        hhv_group_count: int = 5,
        hhv_keep_groups: Sequence[int] = (2, 3, 4),
        amount_expand_fast_window: int = 5,
        amount_expand_slow_window: int = 20,
        factor_sort_enabled: bool = False,
        amount_expand_descending: bool = False,
    ):
        super().__init__()
        self.top_k = top_k
        self.hold_days = hold_days
        self.rebalance_every_n_trade_days = rebalance_every_n_trade_days
        self.min_listing_trade_days = min_listing_trade_days
        self.candidate_pool_size = candidate_pool_size
        self.cap_field = cap_field
        self.liquidity_window = liquidity_window
        self.min_avg_amount = min_avg_amount
        self.min_avg_turn = self._normalize_turn_threshold(min_avg_turn)
        self.exclude_bottom_liquidity_pct = max(min(float(exclude_bottom_liquidity_pct), 1.0), 0.0)
        self.exclude_bottom_liquidity_metric = str(exclude_bottom_liquidity_metric).lower()
        if self.exclude_bottom_liquidity_metric not in {"amount", "volume"}:
            raise ValueError("exclude_bottom_liquidity_metric must be one of: amount, volume")
        self.min_close_price = min_close_price
        self.factor_filter_enabled = bool(factor_filter_enabled)
        self.factor_sort_enabled = bool(factor_sort_enabled)
        self.amount_expand_descending = bool(amount_expand_descending)
        self.hhv_window = int(hhv_window)
        self.hhv_group_count = int(hhv_group_count)
        self.hhv_keep_groups = self._normalize_hhv_keep_groups(hhv_keep_groups)
        self.amount_expand_fast_window = int(amount_expand_fast_window)
        self.amount_expand_slow_window = int(amount_expand_slow_window)
        self._validate_factor_parameters()

    def required_feature_fields(self) -> Sequence[str]:
        return ("code", "date", self.cap_field)

    def _validate_factor_parameters(self) -> None:
        if self.hhv_window < 2:
            raise ValueError("hhv_window must be >= 2")
        if self.hhv_group_count < 3:
            raise ValueError("hhv_group_count must be >= 3")
        if not self.hhv_keep_groups:
            raise ValueError("hhv_keep_groups cannot be empty")
        invalid_groups = [group for group in self.hhv_keep_groups if group < 1 or group > self.hhv_group_count]
        if invalid_groups:
            raise ValueError(f"hhv_keep_groups must be within 1..{self.hhv_group_count}: {invalid_groups}")
        if self.amount_expand_fast_window < 1:
            raise ValueError("amount_expand_fast_window must be >= 1")
        if self.amount_expand_slow_window < self.amount_expand_fast_window:
            raise ValueError("amount_expand_slow_window must be >= amount_expand_fast_window")

    @staticmethod
    def _normalize_hhv_keep_groups(raw_groups: Sequence[int] | str) -> tuple[int, ...]:
        if isinstance(raw_groups, str):
            groups = [int(part.strip()) for part in raw_groups.split(",") if part.strip()]
        else:
            groups = [int(group) for group in raw_groups]
        return tuple(sorted(set(groups)))

    @property
    def hhv_col(self) -> str:
        return f"hhv_{self.hhv_window}d"

    @property
    def distance_to_hhv_col(self) -> str:
        return f"distance_to_hhv{self.hhv_window}"

    @property
    def research_distance_to_hhv_col(self) -> str:
        return f"research_distance_to_hhv{self.hhv_window}"

    @property
    def amount_expand_fast_col(self) -> str:
        return f"avg_amount_{self.amount_expand_fast_window}d"

    @property
    def amount_expand_slow_col(self) -> str:
        return f"avg_amount_{self.amount_expand_slow_window}d"

    @staticmethod
    def _safe_float_metadata(metadata: dict, key: str, value) -> None:
        if pd.notna(value):
            metadata[key] = float(value)

    @staticmethod
    def _normalize_turn_threshold(min_avg_turn: float | None) -> float | None:
        """统一换手率阈值口径。

        当前日线里的 turn 字段按“百分比数值”存储，例如：
        - 1 表示 1%
        - 13.4 表示 13.4%

        为了避免旧参数误传成小数形式，这里兼容：
        - 0.01 -> 1
        - 0.02 -> 2
        """

        if min_avg_turn is None:
            return None
        if 0 < min_avg_turn < 1:
            return min_avg_turn * 100.0
        return min_avg_turn

    def _warmup_trade_dates(
        self,
        full_trade_calendar: pd.DatetimeIndex,
        signal_start: datetime,
    ) -> datetime:
        signal_start_ts = pd.Timestamp(signal_start)
        start_index = int(full_trade_calendar.searchsorted(signal_start_ts, side="left"))
        warmup_index = max(0, start_index - (self._factor_warmup_window() + 5))
        return to_pydatetime(full_trade_calendar[warmup_index])

    def _factor_warmup_window(self) -> int:
        return max(self.liquidity_window, self.hhv_window, self.amount_expand_slow_window)

    def _prepare_base_feature_frame(
        self,
        data_portal,
        trade_dates: Sequence[datetime],
        *,
        rebalance_only: bool = True,
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        return self._prepare_smallcap_feature_frame(
            data_portal,
            trade_dates,
            feature_fields=["code", "date", self.cap_field],
            cap_field=self.cap_field,
            rebalance_every_n_trade_days=self.rebalance_every_n_trade_days,
            min_listing_trade_days=self.min_listing_trade_days,
            rebalance_only=rebalance_only,
        )

    def _load_or_build_cached_strategy_frame(
        self,
        data_portal,
        stage: str,
        payload: dict,
        builder,
    ) -> pd.DataFrame:
        frame_cache = getattr(data_portal, "frame_cache", None)
        if frame_cache is None:
            return builder()
        return frame_cache.load_or_build_frame(stage, payload, builder)

    def _prepare_cached_base_feature_frame(
        self,
        data_portal,
        trade_dates: Sequence[datetime],
        *,
        rebalance_only: bool = True,
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        if not trade_dates:
            return pd.DataFrame(), pd.DatetimeIndex([])

        signal_start = to_pydatetime(trade_dates[0])
        signal_end = to_pydatetime(trade_dates[-1])
        payload = {
            "feature_formula_version": "smallcap_base_v1",
            "start_date": signal_start,
            "end_date": signal_end,
            "cap_field": self.cap_field,
            "rebalance_every_n_trade_days": self.rebalance_every_n_trade_days,
            "min_listing_trade_days": self.min_listing_trade_days,
            "rebalance_only": rebalance_only,
            "smallcap_code_prefixes": self.smallcap_code_prefixes,
        }

        def builder() -> pd.DataFrame:
            base_frame, _ = self._prepare_base_feature_frame(
                data_portal,
                trade_dates,
                rebalance_only=rebalance_only,
            )
            return base_frame

        base_frame = self._load_or_build_cached_strategy_frame(
            data_portal,
            "smallcap_base_feature_frame",
            payload,
            builder,
        )
        if base_frame.empty or "ipoDate" not in base_frame.columns:
            return base_frame, pd.DatetimeIndex([])

        earliest_ipo_date = pd.to_datetime(base_frame["ipoDate"], errors="coerce").dropna().min()
        if pd.isna(earliest_ipo_date):
            return base_frame, pd.DatetimeIndex([])
        full_trade_calendar = pd.DatetimeIndex(
            pd.to_datetime(data_portal.get_trade_calendar(to_pydatetime(earliest_ipo_date), signal_end))
        )
        return base_frame, full_trade_calendar

    def _prepare_cached_candidate_pool_frame(
        self,
        data_portal,
        trade_dates: Sequence[datetime],
        *,
        rebalance_only: bool = True,
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        base_frame, full_trade_calendar = self._prepare_cached_base_feature_frame(
            data_portal,
            trade_dates,
            rebalance_only=rebalance_only,
        )
        if base_frame.empty or full_trade_calendar.empty:
            return pd.DataFrame(), full_trade_calendar

        signal_start = to_pydatetime(trade_dates[0])
        signal_end = to_pydatetime(trade_dates[-1])
        payload = {
            "feature_formula_version": "smallcap_candidate_pool_v1",
            "start_date": signal_start,
            "end_date": signal_end,
            "cap_field": self.cap_field,
            "rebalance_every_n_trade_days": self.rebalance_every_n_trade_days,
            "min_listing_trade_days": self.min_listing_trade_days,
            "candidate_pool_size": self.candidate_pool_size,
            "rebalance_only": rebalance_only,
            "smallcap_code_prefixes": self.smallcap_code_prefixes,
        }
        candidate_pool_frame = self._load_or_build_cached_strategy_frame(
            data_portal,
            "smallcap_candidate_pool",
            payload,
            lambda: self._build_candidate_pool_frame(base_frame),
        )
        return candidate_pool_frame, full_trade_calendar

    def _build_candidate_pool_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        return (
            frame.sort_values(["trade_date", self.cap_field, "code"], ascending=[True, True, True])
            .groupby("trade_date", group_keys=False)
            .head(self.candidate_pool_size)
            .reset_index(drop=True)
        )

    def _build_signal_table(self, frame: pd.DataFrame) -> dict[datetime, pd.DataFrame]:
        if frame.empty:
            return {}

        final_frame = frame.copy()
        if self.factor_sort_enabled:
            required_columns = {"trade_date", "code", self.cap_field, "amount_expand"}
            missing_columns = [column for column in sorted(required_columns) if column not in final_frame.columns]
            if missing_columns:
                raise KeyError(
                    "Factor sort missing required columns: "
                    f"{missing_columns}. Available columns: {sorted(final_frame.columns.tolist())}"
                )
            final_frame = final_frame.dropna(subset=["amount_expand"]).copy()
            final_frame = final_frame.sort_values(
                ["trade_date", "amount_expand", self.cap_field, "code"],
                ascending=[True, not self.amount_expand_descending, True, True],
            )
        else:
            final_frame = final_frame.sort_values(["trade_date", self.cap_field, "code"], ascending=[True, True, True])
        return self.group_signal_table(final_frame, limit=self.top_k)

    def _prepare_liquidity_indicator_frame(
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
                "feature_formula_version": "liquidity_indicator_v2",
                "warmup_start": warmup_start,
                "signal_end": signal_end,
                "codes_signature": frame_cache.codes_signature(codes),
                "liquidity_window": self.liquidity_window,
                "hhv_window": self.hhv_window,
                "amount_expand_fast_window": self.amount_expand_fast_window,
                "amount_expand_slow_window": self.amount_expand_slow_window,
                "price_mode": "raw",
            }
            return frame_cache.load_or_build_frame(
                "liquidity_indicator_frame",
                payload,
                lambda: self._build_liquidity_indicator_frame(
                    data_portal,
                    codes=codes,
                    warmup_start=warmup_start,
                    signal_end=signal_end,
                    research_store=research_store,
                ),
            )

        return self._build_liquidity_indicator_frame(
            data_portal,
            codes=codes,
            warmup_start=warmup_start,
            signal_end=signal_end,
            research_store=research_store,
        )

    def _build_liquidity_indicator_frame(
        self,
        data_portal,
        *,
        codes: Sequence[str],
        warmup_start: datetime,
        signal_end: datetime,
        research_store=None,
    ) -> pd.DataFrame:
        history_end = to_pydatetime(pd.Timestamp(signal_end) + pd.Timedelta(days=1))
        fields = ["code", "trade_date", "close", "amount", "volume", "turn", "isST"]
        if research_store is not None:
            daily_history = research_store.load_daily_history(
                warmup_start,
                history_end,
                codes=codes,
                fields=fields,
                include_stopped=False,
                price_mode="raw",
                batch_size=1000,
            )
        else:
            daily_history = data_portal.get_daily_history(
                warmup_start,
                history_end,
                codes=codes,
                fields=fields,
                include_stopped=False,
                price_mode="raw",
                batch_size=1000,
            )
        if daily_history.empty:
            return pd.DataFrame()

        daily_history = daily_history.copy()
        daily_history["trade_date"] = pd.to_datetime(daily_history["trade_date"])
        daily_history = daily_history.sort_values(["code", "trade_date"]).reset_index(drop=True)

        grouped = daily_history.groupby("code")
        amount_windows = sorted({self.liquidity_window, self.amount_expand_fast_window, self.amount_expand_slow_window})
        for window in amount_windows:
            daily_history[f"avg_amount_{window}d"] = grouped["amount"].transform(
                lambda values, rolling_window=window: values.rolling(
                    rolling_window,
                    min_periods=rolling_window,
                ).mean()
            )
        daily_history[f"avg_volume_{self.liquidity_window}d"] = grouped["volume"].transform(
            lambda values: values.rolling(self.liquidity_window, min_periods=self.liquidity_window).mean()
        )
        daily_history[f"avg_turn_{self.liquidity_window}d"] = grouped["turn"].transform(
            lambda values: values.rolling(self.liquidity_window, min_periods=self.liquidity_window).mean()
        )
        daily_history[self.hhv_col] = grouped["close"].transform(
            lambda values: values.rolling(self.hhv_window, min_periods=self.hhv_window).max()
        )
        hhv_denominator = daily_history[self.hhv_col].replace(0, pd.NA)
        daily_history[self.distance_to_hhv_col] = daily_history["close"] / hhv_denominator - 1.0
        daily_history[self.research_distance_to_hhv_col] = -daily_history[self.distance_to_hhv_col]
        amount_denominator = daily_history[self.amount_expand_slow_col].replace(0, pd.NA)
        daily_history["amount_expand"] = daily_history[self.amount_expand_fast_col] / amount_denominator
        daily_history["research_amount_expand"] = -daily_history["amount_expand"]
        return daily_history

    def _apply_liquidity_filter(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        amount_col = f"avg_amount_{self.liquidity_window}d"
        volume_col = f"avg_volume_{self.liquidity_window}d"
        turn_col = f"avg_turn_{self.liquidity_window}d"
        metric_col = amount_col if self.exclude_bottom_liquidity_metric == "amount" else volume_col
        required_columns = {"trade_date", "code", amount_col, turn_col}
        if self.exclude_bottom_liquidity_pct > 0:
            required_columns.add(metric_col)
        if self.min_close_price is not None:
            required_columns.add("close")
        missing_columns = [column for column in sorted(required_columns) if column not in frame.columns]
        if missing_columns:
            raise KeyError(
                "Liquidity filter missing required columns: "
                f"{missing_columns}. Available columns: {sorted(frame.columns.tolist())}"
            )
        filtered = frame.copy()
        if self.min_avg_amount is not None:
            filtered = filtered[filtered[amount_col] >= self.min_avg_amount].copy()
        if self.min_avg_turn is not None:
            filtered = filtered[filtered[turn_col] >= self.min_avg_turn].copy()
        if self.exclude_bottom_liquidity_pct > 0 and not filtered.empty:
            filtered = filtered.sort_values(["trade_date", metric_col, "code"], ascending=[True, True, True]).copy()
            filtered["_liquidity_rank"] = filtered.groupby("trade_date").cumcount()
            filtered["_liquidity_group_size"] = filtered.groupby("trade_date")["code"].transform("size")
            filtered["_liquidity_exclude_count"] = (
                filtered["_liquidity_group_size"] * self.exclude_bottom_liquidity_pct
            ).apply(math.ceil).astype(int)
            filtered = filtered[filtered["_liquidity_rank"] >= filtered["_liquidity_exclude_count"]].copy()
            filtered = filtered.drop(columns=["_liquidity_rank", "_liquidity_group_size", "_liquidity_exclude_count"])
        if self.min_close_price is not None:
            filtered = filtered[filtered["close"] >= self.min_close_price].copy()
        return filtered

    def _apply_factor_filter(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not self.factor_filter_enabled:
            return frame

        research_col = self.research_distance_to_hhv_col
        required_columns = {"trade_date", "code", research_col}
        missing_columns = [column for column in sorted(required_columns) if column not in frame.columns]
        if missing_columns:
            raise KeyError(
                "Factor filter missing required columns: "
                f"{missing_columns}. Available columns: {sorted(frame.columns.tolist())}"
            )

        filtered = frame.dropna(subset=[research_col]).copy()
        if filtered.empty:
            return filtered

        daily_groups = filtered.groupby("trade_date", sort=False)
        feature_count = daily_groups[research_col].transform("count")
        feature_rank = daily_groups[research_col].rank(method="first")
        raw_bucket = ((feature_rank - 1.0) * self.hhv_group_count // feature_count) + 1.0
        bucket_series = pd.Series(raw_bucket, index=filtered.index).where(feature_count >= self.hhv_group_count)
        filtered["hhv_group"] = pd.array(bucket_series, dtype="Int8")
        filtered = filtered[filtered["hhv_group"].isin(self.hhv_keep_groups)].copy()
        return filtered

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
        indicator_frame = self._prepare_liquidity_indicator_frame(
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
        frame = self._apply_liquidity_filter(frame)
        if frame.empty:
            return
        frame = self._apply_factor_filter(frame)
        if frame.empty:
            return

        self.set_signal_table(self._build_signal_table(frame))

    def generate_candidates(
        self,
        signal_date: datetime,
        feature_slice: pd.DataFrame,
        portfolio_snapshot: dict,
    ) -> list[DailyCandidate]:
        frame = self.signal_frame(signal_date)
        if frame is None or frame.empty:
            return []

        amount_col = f"avg_amount_{self.liquidity_window}d"
        volume_col = f"avg_volume_{self.liquidity_window}d"
        turn_col = f"avg_turn_{self.liquidity_window}d"
        candidates: list[DailyCandidate] = []
        for row in frame.itertuples(index=False):
            metadata = {
                self.cap_field: float(getattr(row, self.cap_field)),
                "close": float(row.close),
            }
            amount_value = getattr(row, amount_col, None)
            volume_value = getattr(row, volume_col, None)
            turn_value = getattr(row, turn_col, None)
            if pd.notna(amount_value):
                metadata[amount_col] = float(amount_value)
            if pd.notna(volume_value):
                metadata[volume_col] = float(volume_value)
            if pd.notna(turn_value):
                metadata[turn_col] = float(turn_value)
            self._safe_float_metadata(metadata, "distance_to_hhv", getattr(row, self.distance_to_hhv_col, None))
            self._safe_float_metadata(metadata, "research_distance_to_hhv", getattr(row, self.research_distance_to_hhv_col, None))
            hhv_group_value = getattr(row, "hhv_group", None)
            if pd.notna(hhv_group_value):
                metadata["hhv_group"] = int(hhv_group_value)
            self._safe_float_metadata(metadata, "amount_expand", getattr(row, "amount_expand", None))
            self._safe_float_metadata(metadata, "research_amount_expand", getattr(row, "research_amount_expand", None))
            self._safe_float_metadata(metadata, self.amount_expand_fast_col, getattr(row, self.amount_expand_fast_col, None))
            self._safe_float_metadata(metadata, self.amount_expand_slow_col, getattr(row, self.amount_expand_slow_col, None))

            candidates.append(
                DailyCandidate(
                    signal_date=signal_date,
                    code=row.code,
                    score=float(getattr(row, "amount_expand")) if self.factor_sort_enabled else float(getattr(row, self.cap_field)),
                    hold_days=self.hold_days,
                    metadata=metadata,
                )
            )
        return candidates
