from __future__ import annotations

from datetime import datetime
from inspect import signature
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd

from backtest.execution.config import EngineConfig
from backtest.feature.minervini_fundamental_feature import (
    FEATURE_VERSION as MINERVINI_FUNDAMENTAL_FEATURE_VERSION,
    TARGET_COLLECTION as MINERVINI_FUNDAMENTAL_FEATURE_COLLECTION,
)
from backtest.utils.datetime_utils import to_pydatetime
from backtest.utils.log import log_event

from .base import SmallCapSignalTableStrategy
from .models import DailyCandidate


MINERVINI_FUNDAMENTAL_STRATEGY_FIELDS = (
    "code",
    "code_name",
    "statDate",
    "pubDate",
    "revisionDate",
    "source",
    "featureVersion",
    "parent_net_profit_single",
    "parent_net_profit_yoy",
    "parent_net_profit_ttm_yoy",
    "eps_yoy",
    "eps_ttm_yoy",
    "revenue_yoy",
    "revenue_ttm_yoy",
    "yoy_extreme_flag",
    "low_base_flag",
    "missing_core_fields",
)


class MinerviniAshareStrategy(SmallCapSignalTableStrategy):
    """贴合 A 股的 Minervini 风格策略，包含突破入场与盈利加仓。

    整体流程刻意拆成两个阶段：
    1. 先用日线确认今天是否出现有效突破。
    2. 再把突破得到的元数据交给执行引擎，在下一交易日完成成交。

    这样策略层专注定义“什么 setup 可以买”，
    执行层专注处理“买多少、怎么落仓、怎么跟踪风险”。
    """

    def __init__(
        self,
        *,
        benchmark_code: str | None = None,
        top_k: int = 10,
        hold_days: int = 20,
        rebalance_every_n_trade_days: int = 5,
        min_listing_trade_days: int = 250,
        min_liqa_mv: float = 2e9,
        max_liqa_mv: float | None = None,
        min_revenue_yoy: float = 0.20,
        min_net_profit_yoy: float = 0.25,
        min_ttm_net_profit_yoy: float = 0.08,
        min_eps_yoy_floor: float = -0.20,
        min_eps_ttm_yoy: float = 0.0,
        allow_missing_revenue_yoy: bool = False,
        min_rps: float = 90.0,
        min_close_to_high_250: float = 0.75,
        min_above_low_250: float = 1.25,
        price_mode: str = "qfq",
        ma_micro_window: int = 5,
        ma_pullback_window: int = 20,
        ma_short_window: int = 50,
        ma_mid_window: int = 150,
        ma_long_window: int = 200,
        ma_long_rise_window: int = 20,
        rps_window_short: int = 120,
        rps_window_long: int = 250,
        breakout_buffer_pct: float = 0.0,
        breakout_volume_window: int = 50,
        min_breakout_volume_ratio: float = 1.5,
        vcp_mode: str = "rolling",
        vcp_breakout_volume_ratio: float = 1.03,
        platform_window: int = 40,
        max_platform_depth: float = 0.20,
        vcp_short_window: int = 5,
        vcp_mid_window: int = 10,
        vcp_base_window: int = 20,
        vcp_long_window: int = 40,
        max_vcp_depth: float = 0.30,
        vcp_swing_window: int = 80,
        vcp_swing_lookback: int = 3,
        vcp_swing_min_depth: float = 0.05,
        vcp_swing_max_depth: float = 0.35,
        vcp_swing_recovery_ratio: float = 0.92,
        stop_atr_multiple: float = 1.5,
        max_initial_stop_pct: float = 0.12,
        risk_fraction: float = 0.005,
        add_on_risk_fraction: float = 0.0025,
        add_on_short_pivot_window: int = 10,
        min_add_on_volume_ratio: float = 1.2,
        add_on_trigger_r_multiples: Sequence[float] = (0.5, 1.0),
        max_add_on_count: int = 2,
        breakout_as_filter: bool = True,
        vcp_breakout_bonus: float = 12.0,
        platform_breakout_bonus: float = 6.0,
        build_execution_raw_bridge: bool = True,
        indicator_code_batch_size: int = 500,
        fundamental_feature_version: str = MINERVINI_FUNDAMENTAL_FEATURE_VERSION,
        progress_logging: bool = True,
        progress_log_every_days: int = 1,
    ):
        super().__init__()
        self.benchmark_code = benchmark_code or EngineConfig().benchmark_code
        self.top_k = top_k
        self.hold_days = hold_days
        self.rebalance_every_n_trade_days = rebalance_every_n_trade_days
        self.min_listing_trade_days = min_listing_trade_days
        self.min_liqa_mv = min_liqa_mv
        self.max_liqa_mv = max_liqa_mv
        self.min_revenue_yoy = min_revenue_yoy
        self.min_net_profit_yoy = min_net_profit_yoy
        self.min_ttm_net_profit_yoy = min_ttm_net_profit_yoy
        self.min_eps_yoy_floor = min_eps_yoy_floor
        self.min_eps_ttm_yoy = min_eps_ttm_yoy
        self.allow_missing_revenue_yoy = allow_missing_revenue_yoy
        self.min_rps = min_rps
        self.min_close_to_high_250 = min_close_to_high_250
        self.min_above_low_250 = min_above_low_250
        self.price_mode = price_mode
        self.ma_micro_window = ma_micro_window
        self.ma_pullback_window = ma_pullback_window
        self.ma_short_window = ma_short_window
        self.ma_mid_window = ma_mid_window
        self.ma_long_window = ma_long_window
        self.ma_long_rise_window = ma_long_rise_window
        self.rps_window_short = rps_window_short
        self.rps_window_long = rps_window_long
        self.breakout_buffer_pct = breakout_buffer_pct
        self.breakout_volume_window = breakout_volume_window
        self.min_breakout_volume_ratio = min_breakout_volume_ratio
        if vcp_mode not in {"rolling", "swing"}:
            raise ValueError("vcp_mode must be 'rolling' or 'swing'")
        self.vcp_mode = vcp_mode
        self.vcp_breakout_volume_ratio = vcp_breakout_volume_ratio
        self.platform_window = platform_window
        self.max_platform_depth = max_platform_depth
        self.vcp_short_window = vcp_short_window
        self.vcp_mid_window = vcp_mid_window
        self.vcp_base_window = vcp_base_window
        self.vcp_long_window = vcp_long_window
        self.max_vcp_depth = max_vcp_depth
        self.vcp_swing_window = vcp_swing_window
        self.vcp_swing_lookback = vcp_swing_lookback
        self.vcp_swing_min_depth = vcp_swing_min_depth
        self.vcp_swing_max_depth = vcp_swing_max_depth
        self.vcp_swing_recovery_ratio = vcp_swing_recovery_ratio
        self.stop_atr_multiple = stop_atr_multiple
        self.max_initial_stop_pct = max_initial_stop_pct
        self.risk_fraction = risk_fraction
        self.add_on_risk_fraction = add_on_risk_fraction
        self.add_on_short_pivot_window = add_on_short_pivot_window
        self.min_add_on_volume_ratio = min_add_on_volume_ratio
        self.add_on_trigger_r_multiples = tuple(add_on_trigger_r_multiples)
        self.max_add_on_count = max_add_on_count
        self.breakout_as_filter = breakout_as_filter
        self.vcp_breakout_bonus = vcp_breakout_bonus
        self.platform_breakout_bonus = platform_breakout_bonus
        self.build_execution_raw_bridge = build_execution_raw_bridge
        self.indicator_code_batch_size = max(int(indicator_code_batch_size), 1)
        self.fundamental_feature_version = fundamental_feature_version
        self.fundamental_missing_records: list[dict[str, object]] = []
        self.fundamental_feature_collection = MINERVINI_FUNDAMENTAL_FEATURE_COLLECTION
        self.progress_logging = progress_logging
        self.progress_log_every_days = max(int(progress_log_every_days), 1)

    def required_feature_fields(self) -> Sequence[str]:
        return ("code", "date", "liqaMV", "totalMV")

    def _prepare_base_feature_frame(
        self,
        data_portal,
        trade_dates: Sequence[datetime],
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        started_at = perf_counter()
        # 先把可交易股票池和完整交易日历准备好，后续指标计算都基于这个底座。
        if not trade_dates:
            if self.progress_logging:
                log_event(
                    "info",
                    "minervini_base_pool_done",
                    reason="trade_dates_empty",
                    elapsed=f"{perf_counter() - started_at:.2f}s",
                )
            return pd.DataFrame(), pd.DatetimeIndex([])

        signal_start = to_pydatetime(trade_dates[0])
        signal_end = to_pydatetime(trade_dates[-1])
        rebalance_dates = pd.to_datetime(trade_dates[:: self.rebalance_every_n_trade_days])
        if self.progress_logging:
            log_event(
                "info",
                "minervini_base_pool_start",
                signal_start=signal_start.date(),
                signal_end=signal_end.date(),
                trade_dates=len(trade_dates),
                rebalance_dates=len(rebalance_dates),
            )

        frame, full_trade_calendar = self._prepare_smallcap_feature_frame(
            data_portal,
            trade_dates,
            feature_fields=["code", "date", "liqaMV", "totalMV"],
            cap_field="liqaMV",
            rebalance_every_n_trade_days=self.rebalance_every_n_trade_days,
            min_listing_trade_days=self.min_listing_trade_days,
            min_cap_value=self.min_liqa_mv,
            max_cap_value=self.max_liqa_mv,
        )
        if frame.empty:
            if self.progress_logging:
                log_event(
                    "info",
                    "minervini_base_pool_done",
                    reason="feature_history_empty",
                    elapsed=f"{perf_counter() - started_at:.2f}s",
                )
            return pd.DataFrame(), pd.DatetimeIndex([])
        if self.progress_logging:
            log_event(
                "info",
                "minervini_feature_history_loaded",
                rows=len(frame),
                codes=frame["code"].nunique(),
                elapsed=f"{perf_counter() - started_at:.2f}s",
            )
        if self.progress_logging:
            log_event(
                "info",
                "minervini_base_pool_filtered",
                rows=len(frame),
                codes=frame["code"].nunique() if not frame.empty else 0,
                trade_calendar_days=len(full_trade_calendar),
                elapsed=f"{perf_counter() - started_at:.2f}s",
            )
        if self.progress_logging:
            log_event(
                "info",
                "minervini_base_pool_done",
                rows=len(frame),
                codes=frame["code"].nunique() if not frame.empty else 0,
                elapsed=f"{perf_counter() - started_at:.2f}s",
            )
        return frame, full_trade_calendar

    def _warmup_start(self, full_trade_calendar: pd.DatetimeIndex, signal_start: datetime) -> datetime:
        # 预加载足够长的历史，覆盖最长的 MA / RPS / VCP 观察窗口。
        signal_start_ts = pd.Timestamp(signal_start)
        start_index = int(full_trade_calendar.searchsorted(signal_start_ts, side="left"))
        lookback_window = max(
            self.ma_long_window,
            self.rps_window_long,
            self.vcp_long_window,
            self.breakout_volume_window,
            250,
        ) + 30
        warmup_index = max(0, start_index - lookback_window)
        return to_pydatetime(full_trade_calendar[warmup_index])

    @staticmethod
    def _rolling_range(values_high: pd.Series, values_low: pd.Series, window: int) -> pd.Series:
        # `shift(1)` 确保突破当天这根 K 线不会反向参与“待突破底部”的计算。
        rolling_high = values_high.shift(1).rolling(window, min_periods=window).max()
        rolling_low = values_low.shift(1).rolling(window, min_periods=window).min()
        return (rolling_high - rolling_low) / rolling_high.replace(0, np.nan)

    @staticmethod
    def _linear_decay_score(values: pd.Series, *, best: float, worst: float) -> pd.Series:
        """把越小越好的指标转成 0-100 分，best 以内满分，worst 以外归零。"""
        numeric = pd.to_numeric(values, errors="coerce")
        if worst <= best:
            raise ValueError("worst must be greater than best")
        score = 100.0 - (numeric - best) / (worst - best) * 100.0
        return score.where(numeric.notna(), 0.0).clip(lower=0.0, upper=100.0)

    def _compute_swing_vcp_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """用轻量局部高低点近似识别 80 日内的 VCP 收缩次数。

        这里刻意不做复杂人工画线：只看已经确认的局部高点 -> 局部低点 -> 回收。
        对第 i 行来说，只使用 i 之前已经能确认的 swing，避免把当天或未来 K 线拿来定义形态。
        """
        group = group.sort_values("trade_date").copy()
        n = len(group)
        index = group.index
        columns = [
            "vcp_swing_count",
            "vcp_swing_count_score",
            "vcp_swing_quality_score",
            "vcp_swing_depth_1",
            "vcp_swing_depth_2",
            "vcp_swing_depth_3",
            "vcp_swing_depth_declining",
            "vcp_last_swing_depth",
            "vcp_last_swing_days",
            "vcp_swing_volume_score",
            "vcp_swing_pivot_proximity_score",
            "vcp_swing_pivot_extension_pct",
        ]
        result = pd.DataFrame(index=index, columns=columns, dtype=float)
        if n <= self.vcp_swing_lookback * 2 + 5:
            return result

        high = pd.to_numeric(group["high"], errors="coerce").to_numpy(dtype=float)
        low = pd.to_numeric(group["low"], errors="coerce").to_numpy(dtype=float)
        close = pd.to_numeric(group["close"], errors="coerce").to_numpy(dtype=float)
        volume_ma_50 = pd.to_numeric(group["volume_ma_50"], errors="coerce").replace(0.0, np.nan)
        dryup_5 = (
            pd.to_numeric(group["volume_ma_5"], errors="coerce") / volume_ma_50
        ).to_numpy(dtype=float)
        dryup_10 = (
            pd.to_numeric(group["volume_ma_10"], errors="coerce") / volume_ma_50
        ).to_numpy(dtype=float)
        pivot_col = f"vcp_pivot_{self.vcp_long_window}"
        pivot = pd.to_numeric(group[pivot_col], errors="coerce").to_numpy(dtype=float)
        lookback = int(self.vcp_swing_lookback)

        local_high = np.zeros(n, dtype=bool)
        local_low = np.zeros(n, dtype=bool)
        for pos in range(lookback, n - lookback):
            high_window = high[pos - lookback : pos + lookback + 1]
            low_window = low[pos - lookback : pos + lookback + 1]
            if np.isfinite(high[pos]) and high[pos] >= np.nanmax(high_window):
                local_high[pos] = True
            if np.isfinite(low[pos]) and low[pos] <= np.nanmin(low_window):
                local_low[pos] = True

        local_high_positions = np.flatnonzero(local_high)
        local_low_positions = np.flatnonzero(local_low)

        for pos in range(n):
            start = max(0, pos - int(self.vcp_swing_window))
            # 局部高低点需要右侧 lookback 根 K 线确认，所以这里只纳入 pos 之前已经确认的点。
            high_positions = local_high_positions[
                (local_high_positions >= start) & (local_high_positions + lookback < pos)
            ]
            low_positions = local_low_positions[
                (local_low_positions >= start) & (local_low_positions + lookback < pos)
            ]
            if len(high_positions) == 0 or len(low_positions) == 0:
                continue

            events = sorted([(item, "high") for item in high_positions] + [(item, "low") for item in low_positions])
            last_high_pos: int | None = None
            contractions: list[tuple[int, int, float]] = []
            for event_pos, event_type in events:
                if event_type == "high":
                    last_high_pos = event_pos
                    continue
                if last_high_pos is None or event_pos <= last_high_pos:
                    continue
                swing_high = high[last_high_pos]
                swing_low = low[event_pos]
                if not np.isfinite(swing_high) or not np.isfinite(swing_low) or swing_high <= 0:
                    continue
                depth = 1.0 - swing_low / swing_high
                if depth < self.vcp_swing_min_depth or depth > self.vcp_swing_max_depth:
                    continue
                recovery_slice = close[event_pos + 1 : pos]
                if recovery_slice.size == 0 or not np.isfinite(recovery_slice).any():
                    continue
                if np.nanmax(recovery_slice) < swing_high * self.vcp_swing_recovery_ratio:
                    continue
                contractions.append((last_high_pos, event_pos, float(depth)))
                last_high_pos = None

            if not contractions:
                continue

            contractions = contractions[-5:]
            depths = [item[2] for item in contractions]
            count = len(depths)
            # Minervini VCP 更重视 3 次左右的有效收缩；次数太多往往说明形态拖沓或震荡松散。
            count_score_map = {
                1: 25.0,
                2: 70.0,
                3: 100.0,
                4: 85.0,
                5: 65.0,
            }
            count_score = count_score_map.get(count, 0.0)
            if count >= 2:
                declining_pairs = [
                    1.0 if depths[idx] <= depths[idx - 1] else 0.0
                    for idx in range(1, count)
                ]
                declining_score = float(np.mean(declining_pairs) * 100.0)
            else:
                declining_score = 0.0
            last_depth = depths[-1]
            if last_depth <= 0.15:
                last_depth_score = 100.0
            else:
                # 最近一轮收缩超过 25% 后，说明买点前的震荡仍偏宽，不再给形态质量分。
                last_depth_score = max(0.0, 100.0 - (last_depth - 0.15) / 0.10 * 100.0)

            dryup_candidates = [value for value in (dryup_5[pos], dryup_10[pos]) if np.isfinite(value)]
            dryup_source = min(dryup_candidates) if dryup_candidates else np.nan
            volume_score = 0.0
            if np.isfinite(dryup_source):
                if 0.6 <= dryup_source <= 1.1:
                    volume_score = 100.0
                elif dryup_source < 0.6:
                    volume_score = max(0.0, dryup_source / 0.6 * 100.0)
                else:
                    volume_score = max(0.0, 100.0 - (dryup_source - 1.1) / 0.7 * 100.0)

            pivot_score = 0.0
            pivot_extension_pct = np.nan
            if np.isfinite(pivot[pos]) and np.isfinite(close[pos]) and pivot[pos] > 0:
                pivot_extension_pct = close[pos] / pivot[pos] - 1.0
                if pivot_extension_pct >= 0:
                    # 突破买点最怕离 pivot 太远，超过 5% 视为已经错过标准 VCP 买点。
                    pivot_score = max(0.0, min(100.0, 100.0 - pivot_extension_pct / 0.05 * 100.0))
                else:
                    # 盘前观察允许尚未突破，但必须足够接近 pivot。
                    pivot_score = max(0.0, min(100.0, 100.0 - abs(pivot_extension_pct) / 0.03 * 100.0))

            quality_score = (
                count_score * 0.25
                + declining_score * 0.20
                + last_depth_score * 0.25
                + volume_score * 0.25
                + pivot_score * 0.05
            )
            last_low_pos = contractions[-1][1]
            result.loc[index[pos], "vcp_swing_count"] = count
            result.loc[index[pos], "vcp_swing_count_score"] = count_score
            result.loc[index[pos], "vcp_swing_quality_score"] = min(max(quality_score, 0.0), 100.0)
            result.loc[index[pos], "vcp_swing_depth_1"] = depths[-1]
            result.loc[index[pos], "vcp_swing_depth_2"] = depths[-2] if count >= 2 else np.nan
            result.loc[index[pos], "vcp_swing_depth_3"] = depths[-3] if count >= 3 else np.nan
            result.loc[index[pos], "vcp_swing_depth_declining"] = declining_score
            result.loc[index[pos], "vcp_last_swing_depth"] = last_depth
            result.loc[index[pos], "vcp_last_swing_days"] = pos - last_low_pos
            result.loc[index[pos], "vcp_swing_volume_score"] = volume_score
            result.loc[index[pos], "vcp_swing_pivot_proximity_score"] = pivot_score
            result.loc[index[pos], "vcp_swing_pivot_extension_pct"] = pivot_extension_pct

        return result

    @staticmethod
    def _rising_streak(values: pd.Series) -> pd.Series:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        out = np.zeros(len(arr), dtype=float)
        streak = 0.0
        for idx in range(1, len(arr)):
            current = arr[idx]
            prev = arr[idx - 1]
            if np.isnan(current) or np.isnan(prev) or current <= prev:
                streak = 0.0
            else:
                streak += 1.0
            out[idx] = streak
        return pd.Series(out, index=values.index)

    @staticmethod
    def _gaussian_target_score(values: pd.Series, *, target: float, bandwidth: float) -> pd.Series:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        numeric = pd.to_numeric(values, errors="coerce")
        normalized_distance = (numeric - target) / bandwidth
        score = 100.0 * np.exp(-2.0 * np.square(normalized_distance))
        score = score.where(numeric.notna(), 0.0)
        return score.clip(lower=0.0, upper=100.0)

    @staticmethod
    def _piecewise_score(values: pd.Series, anchors: Sequence[tuple[float, float]]) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        x = np.asarray([point[0] for point in anchors], dtype=float)
        y = np.asarray([point[1] for point in anchors], dtype=float)
        base = numeric.fillna(x[0]).to_numpy(dtype=float)
        score = np.interp(base, x, y)
        out = pd.Series(score, index=numeric.index, dtype=float)
        out.loc[numeric.isna()] = 0.0
        return out.clip(lower=0.0, upper=100.0)

    def _profit_growth_score(self, values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        score = self._piecewise_score(numeric.clip(lower=0.0), ((0.0, 20.0), (0.10, 40.0), (0.20, 60.0), (0.40, 100.0)))
        score.loc[numeric.le(0) | numeric.isna()] = 0.0
        return score

    def _revenue_growth_score(self, values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        score = self._piecewise_score(numeric.clip(lower=0.0), ((0.0, 20.0), (0.10, 35.0), (0.20, 55.0), (0.40, 100.0)))
        score.loc[numeric.le(0) | numeric.isna()] = 0.0
        return score

    @staticmethod
    def _downcast_float64_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """把大部分技术指标列从 float64 下压到 float32，降低 pandas 合并峰值。"""

        if frame.empty:
            return frame

        float_columns = frame.select_dtypes(include=["float64"]).columns
        if len(float_columns) == 0:
            return frame

        compacted_columns: dict[str, pd.Series] = {}
        for column in frame.columns:
            values = frame[column]
            if column in float_columns:
                compacted_columns[column] = values.astype("float32", copy=False)
            else:
                compacted_columns[column] = values
        return pd.DataFrame(compacted_columns, index=frame.index)

    def _compact_indicator_frame_to_signal_dates(
        self,
        indicator_frame: pd.DataFrame,
        signal_dates: Sequence[datetime] | pd.DatetimeIndex | None,
    ) -> pd.DataFrame:
        """指标层只保留最终需要打信号的日期，避免把 warmup 宽表带到 merge 阶段。"""

        if indicator_frame.empty or signal_dates is None:
            return self._downcast_float64_columns(indicator_frame).reset_index(drop=True)

        needed_dates = pd.Index(pd.to_datetime(signal_dates).unique())
        compact = indicator_frame.loc[indicator_frame["trade_date"].isin(needed_dates)]
        compact = self._downcast_float64_columns(compact)
        return compact.reset_index(drop=True)

    @staticmethod
    def _optimize_daily_history_dtypes(history: pd.DataFrame) -> pd.DataFrame:
        """日线从数据库读出后立刻压缩数值类型，降低后续 rolling 计算的基础内存。"""

        if history.empty:
            return history

        for column in ("close", "high", "low", "volume"):
            if column in history.columns:
                history[column] = pd.to_numeric(history[column], errors="coerce").astype("float32", copy=False)
        if "isST" in history.columns:
            history["isST"] = history["isST"].fillna(False).astype(bool, copy=False)
        return history

    def _prepare_indicator_frame(
        self,
        data_portal,
        *,
        codes: Sequence[str],
        warmup_start: datetime,
        signal_end: datetime,
        research_store=None,
        signal_dates: Sequence[datetime] | pd.DatetimeIndex | None = None,
        _disable_batching: bool = False,
    ) -> pd.DataFrame:
        started_at = perf_counter()
        # 这里集中准备 `prepare()` 后续要用到的价格与技术指标框架。
        if not codes:
            if self.progress_logging:
                log_event(
                    "info",
                    "minervini_indicator_frame_done",
                    reason="codes_empty",
                    elapsed=f"{perf_counter() - started_at:.2f}s",
                )
            return pd.DataFrame()

        history_end = to_pydatetime(pd.Timestamp(signal_end) + pd.Timedelta(days=1))
        all_codes = list(codes) + [self.benchmark_code]
        fields = ["code", "trade_date", "close", "high", "low", "volume", "isST"]
        if self.progress_logging:
            log_event(
                "info",
                "minervini_indicator_frame_start",
                codes=len(codes),
                warmup_start=pd.Timestamp(warmup_start).date(),
                signal_end=pd.Timestamp(signal_end).date(),
                price_mode=self.price_mode,
            )

        def load_history(*, price_mode: str, target_codes: Sequence[str]) -> pd.DataFrame:
            if research_store is not None:
                return research_store.load_daily_history(
                    warmup_start,
                    history_end,
                    codes=target_codes,
                    fields=fields,
                    include_stopped=False,
                    price_mode=price_mode,
                    batch_size=1000,
                )
            return data_portal.get_daily_history(
                warmup_start,
                history_end,
                codes=target_codes,
                fields=fields,
                include_stopped=False,
                price_mode=price_mode,
                batch_size=1000,
            )

        # 主观池只需要信号日期上的指标，但指标计算需要较长 warmup。
        # 这里按股票分批复用同一套计算逻辑：批次内算完整 warmup，马上裁到信号日并下压 float32；
        # 最后再用全市场批次结果统一重算 RPS 横截面排名，避免“批内排名”改变策略含义。
        if (
            not _disable_batching
            and self.indicator_code_batch_size > 0
            and len(codes) > self.indicator_code_batch_size
        ):
            batch_started_at = perf_counter()
            signal_dates_index = pd.Index(pd.to_datetime(signal_dates).unique()) if signal_dates is not None else None
            if self.progress_logging:
                log_event(
                    "info",
                    "minervini_indicator_batches_start",
                    codes=len(codes),
                    batch_size=self.indicator_code_batch_size,
                    signal_dates=len(signal_dates_index) if signal_dates_index is not None else "all",
                )

            batch_frames: list[pd.DataFrame] = []
            code_list = list(codes)
            total_batches = (len(code_list) + self.indicator_code_batch_size - 1) // self.indicator_code_batch_size
            for batch_index, start in enumerate(range(0, len(code_list), self.indicator_code_batch_size), start=1):
                batch_codes = code_list[start : start + self.indicator_code_batch_size]
                one_batch_started_at = perf_counter()
                batch_frame = self._prepare_indicator_frame(
                    data_portal,
                    codes=batch_codes,
                    warmup_start=warmup_start,
                    signal_end=signal_end,
                    research_store=research_store,
                    signal_dates=signal_dates,
                    _disable_batching=True,
                )
                if not batch_frame.empty:
                    batch_frames.append(batch_frame)
                if self.progress_logging:
                    log_event(
                        "info",
                        "minervini_indicator_batch_done",
                        batch_index=batch_index,
                        total_batches=total_batches,
                        codes=len(batch_codes),
                        rows=len(batch_frame),
                        elapsed=f"{perf_counter() - one_batch_started_at:.2f}s",
                    )

            if not batch_frames:
                if self.progress_logging:
                    log_event(
                        "info",
                        "minervini_indicator_frame_done",
                        reason="all_batches_empty",
                        elapsed=f"{perf_counter() - started_at:.2f}s",
                    )
                return pd.DataFrame()

            frame = pd.concat(batch_frames, ignore_index=True, copy=False)
            for window in (self.rps_window_short, self.rps_window_long):
                rs_col = f"rs_{window}"
                rps_col = f"rps_{window}"
                if rs_col in frame.columns:
                    frame[rps_col] = (
                        frame.groupby("trade_date", sort=False)[rs_col].rank(pct=True) * 100.0
                    ).astype("float32", copy=False)
            frame = self._downcast_float64_columns(frame).reset_index(drop=True)
            if self.progress_logging:
                log_event(
                    "info",
                    "minervini_indicator_batches_done",
                    rows=len(frame),
                    columns=len(frame.columns),
                    codes=frame["code"].nunique() if not frame.empty else 0,
                    elapsed=f"{perf_counter() - batch_started_at:.2f}s",
                )
            return frame

        if self.build_execution_raw_bridge:
            raw_started_at = perf_counter()
            raw_history = load_history(price_mode="raw", target_codes=all_codes)
            if raw_history.empty:
                if self.progress_logging:
                    log_event(
                        "info",
                        "minervini_indicator_frame_done",
                        reason="raw_history_empty",
                        elapsed=f"{perf_counter() - started_at:.2f}s",
                    )
                return pd.DataFrame()
            raw_history = raw_history.copy()
            raw_history["trade_date"] = pd.to_datetime(raw_history["trade_date"])
            raw_history = raw_history.sort_values(["code", "trade_date"]).reset_index(drop=True)
            raw_history = self._optimize_daily_history_dtypes(raw_history)
            if self.progress_logging:
                log_event(
                    "info",
                    "minervini_raw_bridge_loaded",
                    rows=len(raw_history),
                    codes=raw_history["code"].nunique(),
                    elapsed=f"{perf_counter() - raw_started_at:.2f}s",
                )

            if self.price_mode == "raw":
                daily_history = raw_history.copy()
                if self.progress_logging:
                    log_event(
                        "info",
                        "minervini_daily_history_loaded",
                        rows=len(daily_history),
                        codes=daily_history["code"].nunique(),
                        source="raw_bridge_reused",
                        elapsed=f"{perf_counter() - raw_started_at:.2f}s",
                    )
            else:
                adjusted_started_at = perf_counter()
                daily_history = data_portal.feature_service.apply_price_mode(
                    raw_history.copy(),
                    price_mode=self.price_mode,
                )
                daily_history["trade_date"] = pd.to_datetime(daily_history["trade_date"])
                daily_history = daily_history.sort_values(["code", "trade_date"]).reset_index(drop=True)
                daily_history = self._optimize_daily_history_dtypes(daily_history)
                if self.progress_logging:
                    log_event(
                        "info",
                        "minervini_daily_history_loaded",
                        rows=len(daily_history),
                        codes=daily_history["code"].nunique(),
                        source="derived_from_raw",
                        elapsed=f"{perf_counter() - adjusted_started_at:.2f}s",
                    )
        else:
            direct_started_at = perf_counter()
            daily_history = load_history(price_mode=self.price_mode, target_codes=all_codes)
            if daily_history.empty:
                if self.progress_logging:
                    log_event(
                        "info",
                        "minervini_indicator_frame_done",
                        reason="daily_history_empty",
                        elapsed=f"{perf_counter() - started_at:.2f}s",
                    )
                return pd.DataFrame()
            if self.progress_logging:
                log_event(
                    "info",
                    "minervini_daily_history_loaded",
                    rows=len(daily_history),
                    codes=daily_history["code"].nunique(),
                    source="direct_price_mode",
                    elapsed=f"{perf_counter() - direct_started_at:.2f}s",
                )

            daily_history = daily_history.copy()
            daily_history["trade_date"] = pd.to_datetime(daily_history["trade_date"])
            daily_history = daily_history.sort_values(["code", "trade_date"]).reset_index(drop=True)
            daily_history = self._optimize_daily_history_dtypes(daily_history)
            raw_history = None

        benchmark = daily_history[daily_history["code"] == self.benchmark_code][["trade_date", "close"]].copy()
        stocks = daily_history[daily_history["code"] != self.benchmark_code].copy()
        if benchmark.empty or stocks.empty:
            if self.progress_logging:
                log_event(
                    "info",
                    "minervini_indicator_frame_done",
                    reason="benchmark_or_stock_empty",
                    benchmark_rows=len(benchmark),
                    stock_rows=len(stocks),
                    elapsed=f"{perf_counter() - started_at:.2f}s",
                )
            return pd.DataFrame()

        same_price_grid = raw_history is None or self.price_mode == "raw"
        if same_price_grid:
            stocks["close_raw"] = stocks["close"]
            stocks["high_raw"] = stocks["high"]
            stocks["low_raw"] = stocks["low"]
        else:
            raw_stocks = raw_history[raw_history["code"] != self.benchmark_code].copy()
            raw_stocks = raw_stocks.rename(
                columns={
                    "close": "close_raw",
                    "high": "high_raw",
                    "low": "low_raw",
                }
            )
            stocks = stocks.merge(
                raw_stocks[["code", "trade_date", "close_raw", "high_raw", "low_raw"]],
                on=["code", "trade_date"],
                how="left",
            )
            if stocks[["close_raw", "high_raw", "low_raw"]].isna().any().any():
                if self.progress_logging:
                    log_event(
                        "info",
                        "minervini_indicator_frame_done",
                        reason="raw_bridge_has_nan",
                        elapsed=f"{perf_counter() - started_at:.2f}s",
                    )
                return pd.DataFrame()

        benchmark = benchmark.sort_values("trade_date")
        benchmark[f"benchmark_ret_{self.rps_window_short}"] = benchmark["close"].pct_change(self.rps_window_short)
        benchmark[f"benchmark_ret_{self.rps_window_long}"] = benchmark["close"].pct_change(self.rps_window_long)
        benchmark = benchmark[
            [
                "trade_date",
                f"benchmark_ret_{self.rps_window_short}",
                f"benchmark_ret_{self.rps_window_long}",
            ]
        ]

        frame = stocks.merge(benchmark, on="trade_date", how="left")
        grouped = frame.groupby("code", group_keys=False, sort=False)
        frame[f"ma_{self.ma_micro_window}"] = grouped["close"].transform(
            lambda values: values.rolling(self.ma_micro_window, min_periods=self.ma_micro_window).mean()
        )
        frame[f"ma_{self.ma_pullback_window}"] = grouped["close"].transform(
            lambda values: values.rolling(self.ma_pullback_window, min_periods=self.ma_pullback_window).mean()
        )
        frame[f"ma_{self.ma_short_window}"] = grouped["close"].transform(
            lambda values: values.rolling(self.ma_short_window, min_periods=self.ma_short_window).mean()
        )
        frame[f"ma_{self.ma_mid_window}"] = grouped["close"].transform(
            lambda values: values.rolling(self.ma_mid_window, min_periods=self.ma_mid_window).mean()
        )
        frame[f"ma_{self.ma_long_window}"] = grouped["close"].transform(
            lambda values: values.rolling(self.ma_long_window, min_periods=self.ma_long_window).mean()
        )
        frame[f"ma_{self.ma_long_window}_prev_{self.ma_long_rise_window}"] = grouped[
            f"ma_{self.ma_long_window}"
        ].shift(self.ma_long_rise_window)
        frame[f"low_{self.ma_long_rise_window}"] = grouped["low"].transform(
            lambda values: values.rolling(self.ma_long_rise_window, min_periods=self.ma_long_rise_window).min()
        )
        frame["high_250"] = grouped["high"].transform(lambda values: values.rolling(250, min_periods=250).max())
        frame["low_250"] = grouped["low"].transform(lambda values: values.rolling(250, min_periods=250).min())
        frame["close_to_high_250"] = frame["close"] / frame["high_250"]
        frame["close_to_low_250"] = frame["close"] / frame["low_250"]
        frame["ma_200_rising_streak_days"] = grouped[f"ma_{self.ma_long_window}"].transform(self._rising_streak)
        frame["ma_200_rising_months"] = frame["ma_200_rising_streak_days"] / 20.0
        frame["ma_200_slope_pct"] = (
            frame[f"ma_{self.ma_long_window}"] / frame[f"ma_{self.ma_long_window}_prev_{self.ma_long_rise_window}"] - 1.0
        ) * 100.0

        for window in (self.rps_window_short, self.rps_window_long):
            frame[f"stock_ret_{window}"] = grouped["close"].pct_change(window)
            frame[f"rs_{window}"] = frame[f"stock_ret_{window}"] - frame[f"benchmark_ret_{window}"]
            frame[f"rps_{window}"] = frame.groupby("trade_date")[f"rs_{window}"].rank(pct=True) * 100
        if self.progress_logging:
            log_event(
                "info",
                "minervini_core_indicators_ready",
                rows=len(frame),
                elapsed=f"{perf_counter() - started_at:.2f}s",
            )

        frame["prev_close"] = grouped["close"].shift(1)
        frame["tr_hl"] = frame["high"] - frame["low"]
        frame["tr_hc"] = (frame["high"] - frame["prev_close"]).abs()
        frame["tr_lc"] = (frame["low"] - frame["prev_close"]).abs()
        frame["true_range"] = frame[["tr_hl", "tr_hc", "tr_lc"]].max(axis=1)
        frame["atr_14"] = grouped["true_range"].transform(lambda values: values.rolling(14, min_periods=14).mean())
        if same_price_grid:
            frame["prev_close_raw"] = frame["prev_close"]
            frame["tr_hl_raw"] = frame["tr_hl"]
            frame["tr_hc_raw"] = frame["tr_hc"]
            frame["tr_lc_raw"] = frame["tr_lc"]
            frame["true_range_raw"] = frame["true_range"]
            frame["atr_14_raw"] = frame["atr_14"]
        else:
            frame["prev_close_raw"] = grouped["close_raw"].shift(1)
            frame["tr_hl_raw"] = frame["high_raw"] - frame["low_raw"]
            frame["tr_hc_raw"] = (frame["high_raw"] - frame["prev_close_raw"]).abs()
            frame["tr_lc_raw"] = (frame["low_raw"] - frame["prev_close_raw"]).abs()
            frame["true_range_raw"] = frame[["tr_hl_raw", "tr_hc_raw", "tr_lc_raw"]].max(axis=1)
            frame["atr_14_raw"] = grouped["true_range_raw"].transform(
                lambda values: values.rolling(14, min_periods=14).mean()
            )
        frame["volume_ma_5"] = grouped["volume"].transform(
            lambda values: values.rolling(5, min_periods=5).mean()
        )
        frame["volume_ma_10"] = grouped["volume"].transform(
            lambda values: values.rolling(10, min_periods=10).mean()
        )
        frame["volume_ma_50"] = grouped["volume"].transform(
            lambda values: values.rolling(self.breakout_volume_window, min_periods=self.breakout_volume_window).mean()
        )
        frame["volume_ratio_50"] = frame["volume"] / frame["volume_ma_50"].replace(0, np.nan)
        bar_range = (frame["high"] - frame["low"]).replace(0, np.nan)
        frame["close_location_score"] = ((frame["close"] - frame["low"]) / bar_range).clip(lower=0.0, upper=1.0)

        frame["high_20_prev"] = grouped["high"].transform(
            lambda values: values.shift(1).rolling(20, min_periods=20).max()
        )
        frame["low_20_prev"] = grouped["low"].transform(
            lambda values: values.shift(1).rolling(20, min_periods=20).min()
        )
        frame["high_55_prev"] = grouped["high"].transform(
            lambda values: values.shift(1).rolling(55, min_periods=55).max()
        )
        frame["low_55_prev"] = grouped["low"].transform(
            lambda values: values.shift(1).rolling(55, min_periods=55).min()
        )
        if same_price_grid:
            frame["high_20_prev_raw"] = frame["high_20_prev"]
            frame["low_20_prev_raw"] = frame["low_20_prev"]
            frame["high_55_prev_raw"] = frame["high_55_prev"]
            frame["low_55_prev_raw"] = frame["low_55_prev"]
        else:
            frame["high_20_prev_raw"] = grouped["high_raw"].transform(
                lambda values: values.shift(1).rolling(20, min_periods=20).max()
            )
            frame["low_20_prev_raw"] = grouped["low_raw"].transform(
                lambda values: values.shift(1).rolling(20, min_periods=20).min()
            )
            frame["high_55_prev_raw"] = grouped["high_raw"].transform(
                lambda values: values.shift(1).rolling(55, min_periods=55).max()
            )
            frame["low_55_prev_raw"] = grouped["low_raw"].transform(
                lambda values: values.shift(1).rolling(55, min_periods=55).min()
            )
        frame["pullback_depth_20"] = 1.0 - frame["low_20_prev"] / frame["high_20_prev"].replace(0, np.nan)

        platform_high_col = f"platform_high_{self.platform_window}"
        platform_low_col = f"platform_low_{self.platform_window}"
        frame[platform_high_col] = grouped["high"].transform(
            lambda values: values.shift(1).rolling(self.platform_window, min_periods=self.platform_window).max()
        )
        frame[platform_low_col] = grouped["low"].transform(
            lambda values: values.shift(1).rolling(self.platform_window, min_periods=self.platform_window).min()
        )
        if same_price_grid:
            frame[f"{platform_high_col}_raw"] = frame[platform_high_col]
            frame[f"{platform_low_col}_raw"] = frame[platform_low_col]
        else:
            frame[f"{platform_high_col}_raw"] = grouped["high_raw"].transform(
                lambda values: values.shift(1).rolling(self.platform_window, min_periods=self.platform_window).max()
            )
            frame[f"{platform_low_col}_raw"] = grouped["low_raw"].transform(
                lambda values: values.shift(1).rolling(self.platform_window, min_periods=self.platform_window).min()
            )
        frame["platform_depth"] = (frame[platform_high_col] - frame[platform_low_col]) / frame[
            platform_high_col
        ].replace(0, np.nan)

        for window in (
            self.vcp_short_window,
            self.vcp_mid_window,
            self.vcp_base_window,
            self.vcp_long_window,
            80,
        ):
            # 这里不用 groupby.apply，避免 pandas 为每个 code 重新拼接大块 DataFrame。
            # rolling range 的口径仍然是“只看前一日之前的窗口”，不会把突破当天 K 线放进底部形态。
            rolling_high = grouped["high"].transform(
                lambda values, rolling_window=window: values.shift(1).rolling(
                    rolling_window,
                    min_periods=rolling_window,
                ).max()
            )
            rolling_low = grouped["low"].transform(
                lambda values, rolling_window=window: values.shift(1).rolling(
                    rolling_window,
                    min_periods=rolling_window,
                ).min()
            )
            frame[f"range_pct_{window}"] = (rolling_high - rolling_low) / rolling_high.replace(0, np.nan)
            del rolling_high, rolling_low

        vcp_pivot_col = f"vcp_pivot_{self.vcp_long_window}"
        vcp_low_col = f"vcp_low_{self.vcp_long_window}"
        frame[vcp_pivot_col] = grouped["high"].transform(
            lambda values: values.shift(1).rolling(self.vcp_long_window, min_periods=self.vcp_long_window).max()
        )
        frame[vcp_low_col] = grouped["low"].transform(
            lambda values: values.shift(1).rolling(self.vcp_long_window, min_periods=self.vcp_long_window).min()
        )
        if same_price_grid:
            frame[f"{vcp_pivot_col}_raw"] = frame[vcp_pivot_col]
            frame[f"{vcp_low_col}_raw"] = frame[vcp_low_col]
        else:
            frame[f"{vcp_pivot_col}_raw"] = grouped["high_raw"].transform(
                lambda values: values.shift(1).rolling(self.vcp_long_window, min_periods=self.vcp_long_window).max()
            )
            frame[f"{vcp_low_col}_raw"] = grouped["low_raw"].transform(
                lambda values: values.shift(1).rolling(self.vcp_long_window, min_periods=self.vcp_long_window).min()
            )

        if self.vcp_mode == "swing":
            swing_started_at = perf_counter()
            swing_features = frame.groupby("code", group_keys=False, sort=False).apply(self._compute_swing_vcp_group)
            if isinstance(swing_features.index, pd.MultiIndex):
                swing_features = swing_features.reset_index(level=0, drop=True)
            swing_features = swing_features.reindex(frame.index)
            for column in swing_features.columns:
                frame[column] = swing_features[column]
            if self.progress_logging:
                log_event(
                    "info",
                    "minervini_swing_vcp_features_ready",
                    rows=len(frame),
                    codes=frame["code"].nunique() if not frame.empty else 0,
                    elapsed=f"{perf_counter() - swing_started_at:.2f}s",
                )

        add_on_pivot_col = f"add_on_pivot_{self.add_on_short_pivot_window}"
        add_on_low_col = f"add_on_low_{self.add_on_short_pivot_window}"
        frame[add_on_pivot_col] = grouped["high"].transform(
            lambda values: values.shift(1).rolling(
                self.add_on_short_pivot_window,
                min_periods=self.add_on_short_pivot_window,
            ).max()
        )
        frame[add_on_low_col] = grouped["low"].transform(
            lambda values: values.shift(1).rolling(
                self.add_on_short_pivot_window,
                min_periods=self.add_on_short_pivot_window,
            ).min()
        )
        if same_price_grid:
            frame[f"{add_on_pivot_col}_raw"] = frame[add_on_pivot_col]
            frame[f"{add_on_low_col}_raw"] = frame[add_on_low_col]
        else:
            frame[f"{add_on_pivot_col}_raw"] = grouped["high_raw"].transform(
                lambda values: values.shift(1).rolling(
                    self.add_on_short_pivot_window,
                    min_periods=self.add_on_short_pivot_window,
                ).max()
            )
            frame[f"{add_on_low_col}_raw"] = grouped["low_raw"].transform(
                lambda values: values.shift(1).rolling(
                    self.add_on_short_pivot_window,
                    min_periods=self.add_on_short_pivot_window,
                ).min()
            )
        frame = self._compact_indicator_frame_to_signal_dates(frame, signal_dates)
        if self.progress_logging:
            log_event(
                "info",
                "minervini_indicator_frame_done",
                rows=len(frame),
                codes=frame["code"].nunique() if not frame.empty else 0,
                elapsed=f"{perf_counter() - started_at:.2f}s",
            )
        return frame

    def _finance_start_pub_date(self, signal_start: datetime) -> datetime:
        """给财报查询留出足够长的发布日期回看窗口。"""

        return to_pydatetime(pd.Timestamp(signal_start) - pd.Timedelta(days=800))

    def _load_fundamental_feature_timeline(
        self,
        data_portal,
        *,
        codes: Sequence[str],
        start_pub_date: datetime,
        end_pub_date: datetime,
        research_store=None,
    ) -> pd.DataFrame:
        started_at = perf_counter()
        if not codes:
            if self.progress_logging:
                log_event(
                    "info",
                    "minervini_fundamental_feature_timeline_done",
                    reason="codes_empty",
                    elapsed=f"{perf_counter() - started_at:.2f}s",
                )
            return pd.DataFrame(columns=list(MINERVINI_FUNDAMENTAL_STRATEGY_FIELDS))

        if self.progress_logging:
            log_event(
                "info",
                "minervini_fundamental_feature_timeline_start",
                codes=len(codes),
                start_pub_date=pd.Timestamp(start_pub_date).date(),
                end_pub_date=pd.Timestamp(end_pub_date).date(),
                feature_version=self.fundamental_feature_version,
            )

        fields = list(MINERVINI_FUNDAMENTAL_STRATEGY_FIELDS)
        if research_store is not None and hasattr(research_store, "load_minervini_fundamental_features"):
            features = research_store.load_minervini_fundamental_features(
                start_pub_date,
                end_pub_date,
                codes=codes,
                fields=fields,
                feature_version=self.fundamental_feature_version,
            )
        else:
            features = data_portal.get_minervini_fundamental_features(
                codes=codes,
                start_pub_date=start_pub_date,
                end_pub_date=end_pub_date,
                fields=fields,
                feature_version=self.fundamental_feature_version,
            )

        if features.empty:
            if self.progress_logging:
                log_event(
                    "info",
                    "minervini_fundamental_feature_timeline_done",
                    reason="features_empty",
                    elapsed=f"{perf_counter() - started_at:.2f}s",
                )
            return pd.DataFrame(columns=list(MINERVINI_FUNDAMENTAL_STRATEGY_FIELDS))

        if self.progress_logging:
            log_event(
                "info",
                "minervini_fundamental_features_loaded",
                rows=len(features),
                codes=features["code"].nunique() if "code" in features.columns else len(codes),
                elapsed=f"{perf_counter() - started_at:.2f}s",
            )

        feature_timeline = self._normalize_fundamental_feature_timeline(features)
        if self.progress_logging:
            log_event(
                "info",
                "minervini_fundamental_feature_timeline_done",
                rows=len(feature_timeline),
                codes=feature_timeline["code"].nunique() if not feature_timeline.empty else 0,
                elapsed=f"{perf_counter() - started_at:.2f}s",
            )
        return feature_timeline

    def _normalize_fundamental_feature_timeline(self, features: pd.DataFrame) -> pd.DataFrame:
        """统一新基本面特征字段，并保留旧策略输出的兼容别名。"""

        if features.empty:
            return pd.DataFrame(columns=list(MINERVINI_FUNDAMENTAL_STRATEGY_FIELDS))

        frame = features.copy()
        for field in ("pubDate", "statDate", "revisionDate", "computedAt"):
            if field in frame.columns:
                frame[field] = pd.to_datetime(frame[field], errors="coerce")
        numeric_fields = [
            field
            for field in frame.columns
            if field not in {"code", "code_name", "source", "featureVersion", "missing_core_fields"}
            and not field.endswith("Date")
            and field not in {"computedAt"}
        ]
        for field in numeric_fields:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")

        frame["net_profit_yoy"] = frame.get("parent_net_profit_yoy", pd.NA)
        return frame.sort_values(["code", "pubDate", "statDate"]).reset_index(drop=True)

    def _merge_fundamental_feature_timeline(self, price_frame: pd.DataFrame, feature_timeline: pd.DataFrame) -> pd.DataFrame:
        """把 Minervini 基本面特征时间线按 `trade_date` 做 as-of 合并。"""

        if price_frame.empty or feature_timeline.empty:
            return pd.DataFrame()

        left = price_frame.dropna(subset=["trade_date"]).copy()
        right = feature_timeline.dropna(subset=["pubDate"]).copy()
        if left.empty or right.empty:
            return pd.DataFrame()

        left = left.sort_values(["trade_date", "code"]).reset_index(drop=True)
        right = right.sort_values(["pubDate", "code", "statDate"]).copy()
        right = right.drop_duplicates(subset=["code", "pubDate"], keep="last").reset_index(drop=True)
        return pd.merge_asof(
            left,
            right,
            left_on="trade_date",
            right_on="pubDate",
            by="code",
            direction="backward",
            allow_exact_matches=True,
        )

    def _pivot_proximity_score(self, close: pd.Series, pivot_price: pd.Series) -> pd.Series:
        """评估当前价格距离 pivot 的远近。

        越贴近 pivot，说明追价越少、盈亏比越友好，得分越高；
        一旦显著高于 pivot，说明已经偏离理想买点，执行分需要下降。
        """
        pivot = pd.to_numeric(pivot_price, errors="coerce")
        last_close = pd.to_numeric(close, errors="coerce")
        valid_mask = pivot.notna() & last_close.notna() & (pivot > 0)
        overshoot = (last_close / pivot) - 1.0
        score = 100.0 - 1000.0 * overshoot.clip(lower=0.0, upper=0.10)
        score = score.where(valid_mask, 0.0)
        score = score.where(~valid_mask | (overshoot >= 0.0), 25.0)
        return score.clip(lower=0.0, upper=100.0)

    def _stop_distance_score(self, risk_ratio: pd.Series) -> pd.Series:
        """评估初始止损宽度是否处于可执行的甜蜜区。"""
        target = min(self.max_initial_stop_pct * 0.5, 0.06)
        bandwidth = max(self.max_initial_stop_pct * 0.20, 0.02)
        return self._gaussian_target_score(
            risk_ratio,
            target=target,
            bandwidth=bandwidth,
        )

    def _load_growth_snapshot(
        self,
        data_portal,
        trade_date: datetime,
        *,
        codes: Sequence[str],
        research_store=None,
    ) -> pd.DataFrame:
        # 兼容旧调用名：现在统一从 Minervini 基本面特征表读取可见快照。
        if not codes:
            return pd.DataFrame(columns=list(MINERVINI_FUNDAMENTAL_STRATEGY_FIELDS))

        if research_store is not None and hasattr(research_store, "load_minervini_fundamental_features"):
            timeline = research_store.load_minervini_fundamental_features(
                None,
                trade_date,
                codes=codes,
                fields=list(MINERVINI_FUNDAMENTAL_STRATEGY_FIELDS),
                feature_version=self.fundamental_feature_version,
            )
        else:
            timeline = data_portal.get_minervini_fundamental_features(
                codes=codes,
                end_pub_date=trade_date,
                fields=list(MINERVINI_FUNDAMENTAL_STRATEGY_FIELDS),
                feature_version=self.fundamental_feature_version,
            )
        timeline = self._normalize_fundamental_feature_timeline(timeline)
        if timeline.empty:
            return timeline
        return timeline.groupby("code", group_keys=False).tail(1).sort_values("code").reset_index(drop=True)

    @staticmethod
    def _missing_core_has_blocker(values: object) -> bool:
        blockers = {
            "parent_net_profit_single",
            "revenue_yoy",
            "parent_net_profit_yoy",
            "parent_net_profit_ttm_yoy",
            "pubDate",
        }
        if values is None or (isinstance(values, float) and pd.isna(values)):
            return False
        if isinstance(values, str):
            text = values.strip()
            if not text or text in {"[]", "None", "nan"}:
                return False
            return any(field in text for field in blockers)
        if isinstance(values, (list, tuple, set)):
            return any(str(item) in blockers for item in values)
        return False

    @staticmethod
    def _coerce_bool_series(values: pd.Series, *, default: bool = False) -> pd.Series:
        if values is None:
            return pd.Series(default)
        if values.dtype == bool:
            return values.fillna(default)
        normalized = values.astype("string").str.strip().str.lower()
        mapped = normalized.map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "y": True,
                "false": False,
                "0": False,
                "no": False,
                "n": False,
            }
        )
        return mapped.fillna(default).astype(bool)

    def _apply_fundamental_scores(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        scored = frame.copy()
        required_numeric_columns = (
            "parent_net_profit_single",
            "parent_net_profit_yoy",
            "eps_yoy",
            "parent_net_profit_ttm_yoy",
            "eps_ttm_yoy",
            "revenue_yoy",
            "revenue_ttm_yoy",
        )
        for column in required_numeric_columns:
            if column not in scored.columns:
                scored[column] = np.nan
        for column in (
            "parent_net_profit_single",
            "parent_net_profit_yoy",
            "eps_yoy",
            "parent_net_profit_ttm_yoy",
            "eps_ttm_yoy",
            "revenue_yoy",
            "revenue_ttm_yoy",
        ):
            if column in scored.columns:
                scored[column] = pd.to_numeric(scored[column], errors="coerce")

        scored["profit_growth_score"] = self._piecewise_score(
            scored["parent_net_profit_yoy"],
            ((0.0, 0.0), (0.10, 20.0), (0.25, 55.0), (0.50, 85.0), (0.80, 100.0)),
        )
        scored["revenue_growth_score"] = self._piecewise_score(
            scored["revenue_yoy"],
            ((0.0, 0.0), (0.10, 25.0), (0.20, 55.0), (0.40, 85.0), (0.70, 100.0)),
        )
        scored["profit_ttm_growth_score"] = self._piecewise_score(
            scored["parent_net_profit_ttm_yoy"],
            ((0.0, 0.0), (0.08, 25.0), (0.20, 55.0), (0.40, 85.0), (0.80, 100.0)),
        )
        scored["revenue_ttm_growth_score"] = self._piecewise_score(
            scored["revenue_ttm_yoy"],
            ((0.0, 0.0), (0.10, 20.0), (0.20, 55.0), (0.50, 85.0), (1.00, 100.0)),
        )
        scored["growth_leadership_score"] = (
            scored["revenue_ttm_growth_score"] * 0.45
            + scored["revenue_growth_score"] * 0.25
            + scored["profit_ttm_growth_score"] * 0.20
            + scored["profit_growth_score"] * 0.10
        )

        if "low_base_flag" not in scored.columns:
            scored["low_base_flag"] = True
        scored["low_base_flag"] = self._coerce_bool_series(scored["low_base_flag"], default=True)
        missing_core_blocker = (
            scored["missing_core_fields"].apply(self._missing_core_has_blocker)
            if "missing_core_fields" in scored.columns
            else pd.Series(False, index=scored.index)
        )
        eps_not_drag = scored["eps_yoy"].ge(self.min_eps_yoy_floor) | scored["eps_ttm_yoy"].gt(self.min_eps_ttm_yoy)
        ttm_sum_positive = scored["parent_net_profit_ttm_yoy"].gt(0.0) | scored["eps_ttm_yoy"].gt(0.0)
        scored["original_like_fundamental_pass"] = (
            scored["parent_net_profit_single"].gt(0.0)
            & scored["revenue_yoy"].ge(self.min_revenue_yoy)
            & scored["parent_net_profit_yoy"].ge(self.min_net_profit_yoy)
            & scored["parent_net_profit_ttm_yoy"].ge(self.min_ttm_net_profit_yoy)
            & eps_not_drag
            & ttm_sum_positive
            & ~scored["low_base_flag"]
            & ~missing_core_blocker
        )
        scored["fundamental_pass"] = scored["original_like_fundamental_pass"]
        penalty = pd.Series(1.0, index=scored.index, dtype=float)
        yoy_extreme = (
            self._coerce_bool_series(scored["yoy_extreme_flag"], default=False)
            if "yoy_extreme_flag" in scored.columns
            else pd.Series(False, index=scored.index)
        )
        penalty = penalty.where(~yoy_extreme, 0.85)
        scored["growth_leadership_score"] = (
            scored["growth_leadership_score"] * penalty
        ).clip(lower=0.0, upper=100.0)
        scored["fundamental_score"] = scored["growth_leadership_score"]
        return scored

    def _apply_finance_filters(self, frame: pd.DataFrame) -> pd.DataFrame:
        # 基本面在这里是硬门槛，不是软加分项。
        if frame.empty:
            return frame

        scored = self._apply_fundamental_scores(frame)
        return scored[scored["fundamental_pass"].fillna(False)].copy()

    def _apply_trend_template(self, frame: pd.DataFrame) -> pd.DataFrame:
        # 经典 Minervini 趋势模板：只有已经走出健康上升趋势的股票，
        # 才值得继续去看底部形态和突破。
        if frame.empty:
            return frame

        ma_micro_col = f"ma_{self.ma_micro_window}"
        ma_pullback_col = f"ma_{self.ma_pullback_window}"
        ma_short_col = f"ma_{self.ma_short_window}"
        ma_mid_col = f"ma_{self.ma_mid_window}"
        ma_long_col = f"ma_{self.ma_long_window}"
        ma_long_prev_col = f"{ma_long_col}_prev_{self.ma_long_rise_window}"
        low_guard_col = f"low_{self.ma_long_rise_window}"
        return frame[
            frame["close"].notna()
            & frame[ma_micro_col].notna()
            & frame[ma_pullback_col].notna()
            & frame[ma_short_col].notna()
            & frame[ma_mid_col].notna()
            & frame[ma_long_col].notna()
            & frame[ma_long_prev_col].notna()
            & frame[low_guard_col].notna()
            & frame["high_250"].notna()
            & frame["low_250"].notna()
            & (frame[ma_micro_col] > frame[ma_pullback_col])
            & (frame["close"] > frame[ma_short_col])
            & (frame["close"] > frame[ma_mid_col])
            & (frame["close"] > frame[ma_long_col])
            & (frame[ma_short_col] > frame[ma_mid_col])
            & (frame[ma_short_col] > frame[ma_long_col])
            & (frame[ma_mid_col] > frame[ma_long_col])
            & (frame[ma_long_col] > frame[ma_long_prev_col])
            & (frame[low_guard_col] > frame[ma_long_prev_col])
            & (frame["close_to_high_250"] >= self.min_close_to_high_250)
            & (frame["close_to_low_250"] >= self.min_above_low_250)
            & (
                (frame[f"rps_{self.rps_window_short}"] >= self.min_rps)
                | (frame[f"rps_{self.rps_window_long}"] >= self.min_rps)
            )
        ].copy()

    def _apply_breakout_filters(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        platform_pivot_col = f"platform_high_{self.platform_window}"
        platform_low_col = f"platform_low_{self.platform_window}"
        vcp_pivot_col = f"vcp_pivot_{self.vcp_long_window}"
        vcp_low_col = f"vcp_low_{self.vcp_long_window}"
        platform_pivot_raw_col = f"{platform_pivot_col}_raw"
        platform_low_raw_col = f"{platform_low_col}_raw"
        vcp_pivot_raw_col = f"{vcp_pivot_col}_raw"
        vcp_low_raw_col = f"{vcp_low_col}_raw"

        breakout_multiplier = 1.0 + self.breakout_buffer_pct
        frame = frame.copy()

        frame["platform_breakout"] = (
            frame[platform_pivot_col].notna()
            & frame[platform_low_col].notna()
            & frame["platform_depth"].notna()
            & (frame["platform_depth"] <= self.max_platform_depth)
            & (frame["close"] >= frame[platform_pivot_col] * breakout_multiplier)
            & (frame["high"] >= frame[platform_pivot_col])
            & (frame["volume_ratio_50"] >= self.min_breakout_volume_ratio)
        )

        frame["vcp_ready"] = (
            frame[f"range_pct_{self.vcp_short_window}"].notna()
            & frame[f"range_pct_{self.vcp_mid_window}"].notna()
            & frame[f"range_pct_{self.vcp_base_window}"].notna()
            & frame[f"range_pct_{self.vcp_long_window}"].notna()
            & (frame[f"range_pct_{self.vcp_short_window}"] <= frame[f"range_pct_{self.vcp_mid_window}"])
            & (frame[f"range_pct_{self.vcp_mid_window}"] <= frame[f"range_pct_{self.vcp_base_window}"])
            & (frame[f"range_pct_{self.vcp_base_window}"] <= frame[f"range_pct_{self.vcp_long_window}"])
            & (frame[f"range_pct_{self.vcp_long_window}"] <= self.max_vcp_depth)
            & (frame[f"range_pct_{self.vcp_short_window}"] <= frame[f"range_pct_{self.vcp_long_window}"] * 0.7)
        )
        vcp_count_parts = [
            frame[f"range_pct_{self.vcp_short_window}"] <= frame[f"range_pct_{self.vcp_mid_window}"],
            frame[f"range_pct_{self.vcp_mid_window}"] <= frame[f"range_pct_{self.vcp_base_window}"],
            frame[f"range_pct_{self.vcp_base_window}"] <= frame[f"range_pct_{self.vcp_long_window}"],
            frame[f"range_pct_{self.vcp_short_window}"] <= frame[f"range_pct_{self.vcp_long_window}"] * 0.7,
            (
                frame[f"range_pct_{self.vcp_long_window}"].notna()
                & frame["range_pct_80"].notna()
                & (frame[f"range_pct_{self.vcp_long_window}"] <= frame["range_pct_80"])
            ),
        ]
        frame["vcp_contraction_count"] = sum(part.fillna(False).astype(int) for part in vcp_count_parts)

        vcp_pivot_distance_pct = frame["close"] / frame[vcp_pivot_col].replace(0, np.nan) - 1.0
        contraction_count_score = (
            frame["vcp_contraction_count"].clip(lower=0, upper=3) / 3.0
        ) * 100.0
        volume_ma_50_safe = frame["volume_ma_50"].replace(0, np.nan)
        volume_dryup_5 = frame["volume_ma_5"] / volume_ma_50_safe
        volume_dryup_10 = frame["volume_ma_10"] / volume_ma_50_safe
        dryup_5_score = ((1.2 - volume_dryup_5) / 0.4 * 100.0).clip(lower=0.0, upper=100.0)
        dryup_10_score = ((1.3 - volume_dryup_10) / 0.4 * 100.0).clip(lower=0.0, upper=100.0)
        frame["vcp_volume_dryup_score"] = pd.concat(
            [dryup_5_score, dryup_10_score],
            axis=1,
        ).max(axis=1, skipna=True).fillna(0.0)
        # VCP 期间量能最好是温和稳定或逐步缩量；持续过热放量反而代表筹码没有安静下来。
        volume_stability_source = volume_dryup_10.fillna(volume_dryup_5)
        frame["vcp_volume_stability_score"] = np.select(
            [
                volume_stability_source.between(0.6, 1.1, inclusive="both"),
                volume_stability_source.lt(0.6),
                volume_stability_source.gt(1.1),
            ],
            [
                100.0,
                (volume_stability_source / 0.6 * 100.0).clip(lower=0.0, upper=100.0),
                (100.0 - (volume_stability_source - 1.1) / 0.7 * 100.0).clip(lower=0.0, upper=100.0),
            ],
            default=0.0,
        )
        tight_score = (100.0 - frame[f"range_pct_{self.vcp_short_window}"] / 0.15 * 100.0).clip(
            lower=0.0,
            upper=100.0,
        )
        pivot_distance_score = (100.0 - vcp_pivot_distance_pct.abs() / 0.03 * 100.0).clip(
            lower=0.0,
            upper=100.0,
        )
        frame["vcp_pivot_proximity_score"] = pivot_distance_score
        frame["vcp_maturity_score"] = (
            pd.Series(frame["vcp_volume_stability_score"], index=frame.index).fillna(0.0) * 0.35
            + frame["vcp_volume_dryup_score"].fillna(0.0) * 0.25
            + tight_score.fillna(0.0) * 0.20
            + pivot_distance_score.fillna(0.0) * 0.15
            + contraction_count_score.fillna(0.0) * 0.05
        ).clip(lower=0.0, upper=100.0)
        if self.vcp_mode == "swing":
            # swing 模式只替换 VCP 主字段，platform / leader / 价格趋势评分仍沿用同一套主观池逻辑。
            for column in (
                "vcp_swing_count",
                "vcp_swing_count_score",
                "vcp_swing_quality_score",
                "vcp_swing_volume_score",
                "vcp_swing_pivot_proximity_score",
                "vcp_swing_pivot_extension_pct",
                "vcp_last_swing_depth",
            ):
                if column not in frame.columns:
                    frame[column] = np.nan
            frame["vcp_contraction_count"] = pd.to_numeric(
                frame["vcp_swing_count"],
                errors="coerce",
            ).fillna(0.0)
            pivot_distance_score = pd.to_numeric(
                frame["vcp_swing_pivot_proximity_score"],
                errors="coerce",
            ).fillna(pivot_distance_score)
            frame["vcp_pivot_proximity_score"] = pivot_distance_score
            frame["vcp_maturity_score"] = pd.to_numeric(
                frame["vcp_swing_quality_score"],
                errors="coerce",
            ).fillna(0.0).clip(lower=0.0, upper=100.0)
            frame["vcp_ready"] = (
                frame[vcp_pivot_col].notna()
                & frame[vcp_low_col].notna()
                & frame[f"range_pct_{self.vcp_long_window}"].notna()
                & (frame[f"range_pct_{self.vcp_long_window}"] <= self.max_vcp_depth)
                & (pd.to_numeric(frame["vcp_swing_count"], errors="coerce") >= 3.0)
                & (frame["vcp_maturity_score"] >= 60.0)
                & (pd.to_numeric(frame["vcp_swing_volume_score"], errors="coerce") >= 70.0)
                & (pd.to_numeric(frame["vcp_last_swing_depth"], errors="coerce") <= 0.25)
            )

        frame["vcp_price_breakout"] = (
            frame["vcp_ready"]
            & frame[vcp_pivot_col].notna()
            & frame[vcp_low_col].notna()
            & (frame["close"] >= frame[vcp_pivot_col] * breakout_multiplier)
            & (frame["high"] >= frame[vcp_pivot_col])
        )
        frame["vcp_breakout_volume_confirm"] = (
            (frame["volume_ratio_50"] >= self.vcp_breakout_volume_ratio)
            & (frame["volume_ratio_50"] <= 2.0)
        )
        frame["vcp_required_volume_ratio"] = self.vcp_breakout_volume_ratio
        frame["vcp_required_day_volume"] = frame["volume_ma_50"] * self.vcp_breakout_volume_ratio
        frame["vcp_breakout_score"] = (
            frame["vcp_maturity_score"].fillna(0.0) * 0.75
            + pivot_distance_score.fillna(0.0) * 0.15
            + frame["close_location_score"].fillna(0.0) * 100.0 * 0.10
        ).clip(lower=0.0, upper=100.0)
        frame["vcp_breakout"] = (
            frame["vcp_price_breakout"]
            & frame["vcp_breakout_volume_confirm"]
            & (frame["vcp_maturity_score"] >= 50.0)
        )
        if self.vcp_mode == "swing":
            pivot_extension = pd.to_numeric(frame["vcp_swing_pivot_extension_pct"], errors="coerce")
            frame["vcp_breakout"] = (
                frame["vcp_price_breakout"]
                & frame["vcp_breakout_volume_confirm"]
                & (frame["vcp_maturity_score"] >= 60.0)
                & (pd.to_numeric(frame["vcp_swing_count"], errors="coerce") >= 3.0)
                & (pd.to_numeric(frame["vcp_swing_volume_score"], errors="coerce") >= 70.0)
                & (pd.to_numeric(frame["vcp_last_swing_depth"], errors="coerce") <= 0.25)
                & pivot_extension.ge(0.0)
                & pivot_extension.le(0.05)
            )

        frame["high_20_reclaim"] = (
            frame["high_20_prev"].notna()
            & (frame["close"] >= frame["high_20_prev"] * breakout_multiplier)
            & (frame["high"] >= frame["high_20_prev"])
        )
        frame["high_55_reclaim"] = (
            frame["high_55_prev"].notna()
            & (frame["close"] >= frame["high_55_prev"] * breakout_multiplier)
            & (frame["high"] >= frame["high_55_prev"])
        )
        frame["new_high_breakout"] = (
            (frame["high_20_reclaim"] | frame["high_55_reclaim"])
            & (frame["volume_ratio_50"] >= 1.2)
        )
        frame["new_high_breakout_55"] = frame["high_55_reclaim"] & (frame["volume_ratio_50"] >= 1.2)

        frame["leader_continuation"] = (
            (frame[f"rps_{self.rps_window_long}"] >= 90.0)
            & (frame[f"rps_{self.rps_window_short}"] >= 92.0)
            & (frame["close_to_high_250"] >= 0.85)
            & (frame[f"ma_{self.ma_micro_window}"] > frame[f"ma_{self.ma_pullback_window}"])
            & (frame[f"ma_{self.ma_pullback_window}"] > frame[f"ma_{self.ma_short_window}"])
            & frame["pullback_depth_20"].notna()
            & (frame["pullback_depth_20"] <= 0.18)
            & frame["high_20_reclaim"]
            & (frame["volume_ratio_50"] >= 1.0)
        )

        frame["new_high_pivot_price"] = np.where(
            frame["new_high_breakout_55"] | frame["high_55_reclaim"],
            frame["high_55_prev"],
            np.where(frame["high_20_reclaim"], frame["high_20_prev"], np.nan),
        )
        frame["new_high_base_low_price"] = np.where(
            frame["new_high_breakout_55"] | frame["high_55_reclaim"],
            frame["low_55_prev"],
            np.where(frame["high_20_reclaim"], frame["low_20_prev"], np.nan),
        )
        frame["new_high_pivot_price_raw"] = np.where(
            frame["new_high_breakout_55"] | frame["high_55_reclaim"],
            frame["high_55_prev_raw"],
            np.where(frame["high_20_reclaim"], frame["high_20_prev_raw"], np.nan),
        )
        frame["new_high_base_low_price_raw"] = np.where(
            frame["new_high_breakout_55"] | frame["high_55_reclaim"],
            frame["low_55_prev_raw"],
            np.where(frame["high_20_reclaim"], frame["low_20_prev_raw"], np.nan),
        )

        frame["setup_type"] = np.select(
            [
                frame["vcp_breakout"],
                frame["platform_breakout"],
                frame["leader_continuation"],
                frame["new_high_breakout"],
            ],
            [
                "vcp_breakout",
                "platform_breakout",
                "leader_continuation",
                "new_high_breakout",
            ],
            default=None,
        )
        frame["pivot_price"] = np.where(
            frame["setup_type"] == "vcp_breakout",
            frame[vcp_pivot_col],
            np.where(
                frame["setup_type"] == "platform_breakout",
                frame[platform_pivot_col],
                frame["new_high_pivot_price"],
            ),
        )
        frame["base_low_price"] = np.where(
            frame["setup_type"] == "vcp_breakout",
            frame[vcp_low_col],
            np.where(
                frame["setup_type"] == "platform_breakout",
                frame[platform_low_col],
                frame["new_high_base_low_price"],
            ),
        )
        frame["pivot_price_raw"] = np.where(
            frame["setup_type"] == "vcp_breakout",
            frame[vcp_pivot_raw_col],
            np.where(
                frame["setup_type"] == "platform_breakout",
                frame[platform_pivot_raw_col],
                frame["new_high_pivot_price_raw"],
            ),
        )
        frame["base_low_price_raw"] = np.where(
            frame["setup_type"] == "vcp_breakout",
            frame[vcp_low_raw_col],
            np.where(
                frame["setup_type"] == "platform_breakout",
                frame[platform_low_raw_col],
                frame["new_high_base_low_price_raw"],
            ),
        )

        stop_floor = pd.concat(
            [
                frame["base_low_price"],
                frame["low"],
                frame["pivot_price"] - frame["atr_14"] * self.stop_atr_multiple,
            ],
            axis=1,
        ).max(axis=1, skipna=True)
        frame["initial_stop_loss"] = stop_floor
        frame["risk_per_share"] = frame["pivot_price"] - frame["initial_stop_loss"]
        frame["signal_pivot_price"] = frame["pivot_price"]
        frame["signal_base_low_price"] = frame["base_low_price"]
        frame["signal_initial_stop_loss"] = frame["initial_stop_loss"]
        frame["signal_risk_per_share"] = frame["risk_per_share"]

        stop_floor_raw = pd.concat(
            [
                frame["base_low_price_raw"],
                frame["low_raw"],
                frame["pivot_price_raw"] - frame["atr_14_raw"] * self.stop_atr_multiple,
            ],
            axis=1,
        ).max(axis=1, skipna=True)
        frame["initial_stop_loss_raw"] = stop_floor_raw
        frame["risk_per_share_raw"] = frame["pivot_price_raw"] - frame["initial_stop_loss_raw"]
        frame["breakout_volume_ratio"] = frame["volume_ratio_50"]
        frame["has_breakout"] = frame["setup_type"].notna()
        if not self.breakout_as_filter:
            return frame

        frame = frame[
            frame["setup_type"].notna()
            & frame["pivot_price"].notna()
            & frame["initial_stop_loss"].notna()
            & frame["atr_14"].notna()
            & (frame["risk_per_share"] > 0)
            & frame["pivot_price_raw"].notna()
            & frame["initial_stop_loss_raw"].notna()
            & frame["atr_14_raw"].notna()
            & (frame["risk_per_share_raw"] > 0)
            & ((frame["risk_per_share_raw"] / frame["pivot_price_raw"]) <= self.max_initial_stop_pct)
        ].copy()
        return frame

    def _apply_selection_score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        scored = frame.copy()
        if "fundamental_score" not in scored.columns:
            scored["profit_growth_score"] = self._profit_growth_score(scored["net_profit_yoy"])
            scored["revenue_growth_score"] = self._revenue_growth_score(scored["revenue_yoy"])
            profit_ttm_source = (
                scored["parent_net_profit_ttm_yoy"]
                if "parent_net_profit_ttm_yoy" in scored.columns
                else scored["net_profit_yoy"]
            )
            scored["profit_ttm_growth_score"] = self._profit_growth_score(profit_ttm_source)
            revenue_ttm_source = (
                scored["revenue_ttm_yoy"]
                if "revenue_ttm_yoy" in scored.columns
                else scored["revenue_yoy"]
            )
            scored["revenue_ttm_growth_score"] = self._revenue_growth_score(revenue_ttm_source)
            scored["growth_leadership_score"] = (
                scored["revenue_ttm_growth_score"] * 0.45
                + scored["revenue_growth_score"] * 0.25
                + scored["profit_ttm_growth_score"] * 0.20
                + scored["profit_growth_score"] * 0.10
            )
            scored["fundamental_score"] = scored["growth_leadership_score"]
        # pool_score 负责“哪些票值得持续放进研究候选池”，
        # 只保留增长质量和 VCP 成熟度，RPS、量能与 setup 只做筛选和解释，不参与排序。
        scored["pool_score"] = (
            scored["growth_leadership_score"] * 0.60
            + scored["vcp_maturity_score"].fillna(0.0) * 0.40
        )

        pivot_reference = scored["pivot_price"].where(scored["pivot_price"].notna(), scored["close"])
        risk_ratio = scored["risk_per_share"] / pivot_reference.replace(0.0, np.nan)
        scored["pivot_proximity_score"] = self._pivot_proximity_score(scored["close"], scored["pivot_price"])
        scored["stop_distance_score"] = self._stop_distance_score(risk_ratio)

        # execution_score 负责“今天这个位置要不要下手”，不再让 setup/RPS/量能参与排序。
        scored["execution_score"] = (
            scored["pool_score"] * 0.50
            + scored["pivot_proximity_score"] * 0.30
            + scored["stop_distance_score"] * 0.20
        )
        # selection_score 保留为兼容字段，继续代表研究层入池排序分。
        scored["selection_score"] = scored["pool_score"]
        return scored

    @staticmethod
    def _compact_indicator_frame_for_merge(indicator_frame: pd.DataFrame, base_frame: pd.DataFrame) -> pd.DataFrame:
        """降低指标层和基础池合并时的内存峰值。

        `indicator_frame` 会包含 warmup 历史，而主观池只需要 `base_frame` 覆盖的信号日期。
        先按日期和列裁掉无关数据，再逐列把 float64 下压到 float32，避免 pandas 在 `.copy()`
        或 merge 前一次性 consolidate 出接近 1GB 的临时数组。
        """
        if indicator_frame.empty or base_frame.empty:
            return indicator_frame

        columns = list(indicator_frame.columns)
        if "trade_date" in indicator_frame.columns and "trade_date" in base_frame.columns:
            needed_dates = pd.Index(pd.to_datetime(base_frame["trade_date"]).unique())
            mask = indicator_frame["trade_date"].isin(needed_dates)
            compact = indicator_frame.loc[mask, columns]
        else:
            compact = indicator_frame.loc[:, columns]

        return MinerviniAshareStrategy._downcast_float64_columns(compact).reset_index(drop=True)

    def _record_missing_fundamentals(self, rows: pd.DataFrame, feature_timeline: pd.DataFrame) -> None:
        """记录没有可见基本面特征的价格候选，供主观池导出诊断。"""

        if rows.empty:
            return

        latest_map: dict[str, tuple[object, object]] = {}
        if not feature_timeline.empty and {"code", "pubDate", "statDate"}.issubset(feature_timeline.columns):
            latest = feature_timeline.dropna(subset=["pubDate"]).sort_values(["code", "pubDate", "statDate"])
            latest_map = {
                item.code: (item.pubDate, item.statDate)
                for item in latest.groupby("code", group_keys=False).tail(1).itertuples(index=False)
            }

        available_columns = [column for column in ("trade_date", "code", "code_name") if column in rows.columns]
        for row in rows[available_columns].itertuples(index=False):
            code = getattr(row, "code")
            latest_pub_date, latest_stat_date = latest_map.get(code, (None, None))
            self.fundamental_missing_records.append(
                {
                    "trade_date": getattr(row, "trade_date", None),
                    "code": code,
                    "code_name": getattr(row, "code_name", ""),
                    "reason": "no_visible_minervini_fundamental_feature",
                    "latest_feature_pubDate": latest_pub_date,
                    "latest_feature_statDate": latest_stat_date,
                }
            )

    def prepare(
        self,
        data_portal,
        trade_dates: Sequence[datetime],
        *,
        research_store=None,
        precomputed_base_frame: pd.DataFrame | None = None,
        precomputed_full_trade_calendar: pd.DatetimeIndex | Sequence[datetime] | None = None,
        precomputed_indicator_frame: pd.DataFrame | None = None,
    ) -> None:
        started_at = perf_counter()
        self.reset_signal_table()
        self.fundamental_missing_records = []
        if not trade_dates:
            if self.progress_logging:
                log_event("info", "minervini_prepare_done", reason="trade_dates_empty", elapsed=f"{perf_counter() - started_at:.2f}s")
            return

        signal_start = to_pydatetime(trade_dates[0])
        signal_end = to_pydatetime(trade_dates[-1])
        if self.progress_logging:
            log_event(
                "info",
                "minervini_prepare_start",
                signal_start=signal_start.date(),
                signal_end=signal_end.date(),
                trade_dates=len(trade_dates),
                rebalance_every=self.rebalance_every_n_trade_days,
            )

        if precomputed_base_frame is not None and precomputed_full_trade_calendar is not None:
            base_frame = precomputed_base_frame.copy()
            full_trade_calendar = pd.DatetimeIndex(pd.to_datetime(precomputed_full_trade_calendar))
            if self.progress_logging:
                log_event(
                    "info",
                    "minervini_base_pool_reused",
                    rows=len(base_frame),
                    codes=base_frame["code"].nunique() if not base_frame.empty and "code" in base_frame.columns else 0,
                    trade_calendar_days=len(full_trade_calendar),
                    elapsed=f"{perf_counter() - started_at:.2f}s",
                )
        else:
            base_frame, full_trade_calendar = self._prepare_base_feature_frame(data_portal, trade_dates)
        if base_frame.empty or full_trade_calendar.empty:
            if self.progress_logging:
                log_event("info", "minervini_prepare_done", reason="base_frame_empty", elapsed=f"{perf_counter() - started_at:.2f}s")
            return

        warmup_start = self._warmup_start(full_trade_calendar, signal_start)
        if self.progress_logging:
            log_event(
                "info",
                "minervini_warmup_ready",
                warmup_start=pd.Timestamp(warmup_start).date(),
                elapsed=f"{perf_counter() - started_at:.2f}s",
            )

        if precomputed_indicator_frame is not None:
            indicator_frame = precomputed_indicator_frame.copy()
            if self.progress_logging:
                log_event(
                    "info",
                    "minervini_indicator_frame_reused",
                    rows=len(indicator_frame),
                    columns=len(indicator_frame.columns),
                    codes=indicator_frame["code"].nunique() if not indicator_frame.empty and "code" in indicator_frame.columns else 0,
                    elapsed=f"{perf_counter() - started_at:.2f}s",
                )
        else:
            indicator_kwargs = {
                "codes": sorted(base_frame["code"].unique().tolist()),
                "warmup_start": warmup_start,
                "signal_end": signal_end,
                "research_store": research_store,
            }
            if "signal_dates" in signature(self._prepare_indicator_frame).parameters:
                indicator_kwargs["signal_dates"] = pd.DatetimeIndex(base_frame["trade_date"].unique())
            indicator_frame = self._prepare_indicator_frame(data_portal, **indicator_kwargs)
        if indicator_frame.empty:
            if self.progress_logging:
                log_event("info", "minervini_prepare_done", reason="indicator_frame_empty", elapsed=f"{perf_counter() - started_at:.2f}s")
            return

        compact_started_at = perf_counter()
        indicator_frame = self._compact_indicator_frame_for_merge(indicator_frame, base_frame)
        if self.progress_logging:
            log_event(
                "info",
                "minervini_indicator_frame_compacted",
                rows=len(indicator_frame),
                columns=len(indicator_frame.columns),
                float32_columns=len(indicator_frame.select_dtypes(include=["float32"]).columns),
                elapsed=f"{perf_counter() - compact_started_at:.2f}s",
            )

        frame = base_frame.merge(indicator_frame, on=["code", "trade_date"], how="inner", copy=False)
        frame = frame[frame["isST"].fillna(False) == False].copy()
        if frame.empty:
            if self.progress_logging:
                log_event("info", "minervini_prepare_done", reason="merged_frame_empty", elapsed=f"{perf_counter() - started_at:.2f}s")
            return
        if self.progress_logging:
            log_event(
                "info",
                "minervini_signal_frame_ready",
                rows=len(frame),
                codes=frame["code"].nunique(),
                trade_dates=frame["trade_date"].nunique(),
                elapsed=f"{perf_counter() - started_at:.2f}s",
            )

        trend_started_at = perf_counter()
        price_frame = self._apply_trend_template(frame)
        if self.progress_logging:
            log_event(
                "info",
                "minervini_trend_template_done",
                input_rows=len(frame),
                rows=len(price_frame),
                codes=price_frame["code"].nunique() if not price_frame.empty else 0,
                elapsed=f"{perf_counter() - trend_started_at:.2f}s",
            )
        if price_frame.empty:
            if self.progress_logging:
                log_event("info", "minervini_prepare_done", reason="price_trend_empty", elapsed=f"{perf_counter() - started_at:.2f}s")
            return

        breakout_started_at = perf_counter()
        price_frame = self._apply_breakout_filters(price_frame)
        if self.progress_logging:
            log_event(
                "info",
                "minervini_breakout_annotation_done",
                rows=len(price_frame),
                codes=price_frame["code"].nunique() if not price_frame.empty else 0,
                breakout_rows=int(price_frame["has_breakout"].fillna(False).sum()) if "has_breakout" in price_frame.columns else 0,
                elapsed=f"{perf_counter() - breakout_started_at:.2f}s",
            )
        if price_frame.empty:
            if self.progress_logging:
                log_event("info", "minervini_prepare_done", reason="breakout_frame_empty", elapsed=f"{perf_counter() - started_at:.2f}s")
            return

        finance_start_pub_date = self._finance_start_pub_date(signal_start)
        fundamental_timeline = self._load_fundamental_feature_timeline(
            data_portal,
            codes=sorted(price_frame["code"].unique().tolist()),
            start_pub_date=finance_start_pub_date,
            end_pub_date=signal_end,
            research_store=research_store,
        )
        if fundamental_timeline.empty:
            self._record_missing_fundamentals(price_frame, fundamental_timeline)
            if self.progress_logging:
                log_event("info", "minervini_prepare_done", reason="fundamental_timeline_empty", elapsed=f"{perf_counter() - started_at:.2f}s")
            return

        finance_merge_started_at = perf_counter()
        merged = self._merge_fundamental_feature_timeline(price_frame, fundamental_timeline)
        if merged.empty:
            self._record_missing_fundamentals(price_frame, fundamental_timeline)
            if self.progress_logging:
                log_event("info", "minervini_prepare_done", reason="fundamental_asof_merge_empty", elapsed=f"{perf_counter() - started_at:.2f}s")
            return
        self._record_missing_fundamentals(merged[merged["pubDate"].isna()].copy(), fundamental_timeline)
        merged = merged[merged["pubDate"].notna()].copy()
        if merged.empty:
            if self.progress_logging:
                log_event("info", "minervini_prepare_done", reason="fundamental_asof_missing_all", elapsed=f"{perf_counter() - started_at:.2f}s")
            return
        if self.progress_logging:
            log_event(
                "info",
                "minervini_fundamental_feature_asof_merged",
                rows=len(merged),
                matched_rows=int(merged["pubDate"].notna().sum()) if "pubDate" in merged.columns else 0,
                missing_rows=len(self.fundamental_missing_records),
                elapsed=f"{perf_counter() - finance_merge_started_at:.2f}s",
            )

        finance_filter_started_at = perf_counter()
        merged = self._apply_finance_filters(merged)
        if self.progress_logging:
            log_event(
                "info",
                "minervini_finance_filter_done",
                rows=len(merged),
                codes=merged["code"].nunique() if not merged.empty else 0,
                elapsed=f"{perf_counter() - finance_filter_started_at:.2f}s",
            )
        if merged.empty:
            if self.progress_logging:
                log_event("info", "minervini_prepare_done", reason="finance_filtered_empty", elapsed=f"{perf_counter() - started_at:.2f}s")
            return

        total_score_started_at = perf_counter()
        merged = self._apply_selection_score(merged)
        if self.progress_logging:
            log_event(
                "info",
                "minervini_total_score_ready",
                rows=len(merged),
                codes=merged["code"].nunique() if not merged.empty else 0,
                elapsed=f"{perf_counter() - total_score_started_at:.2f}s",
            )

        merged = merged.sort_values(
            [
                "trade_date",
                "pool_score",
                "execution_score",
                "growth_leadership_score",
                "vcp_maturity_score",
                "fundamental_score",
                "liqaMV",
            ],
            ascending=[True, False, False, False, False, False, False],
        )

        signal_table = self.group_signal_table(merged)
        self.set_signal_table(signal_table)
        if self.progress_logging:
            log_event(
                "info",
                "minervini_prepare_done",
                signal_dates=len(signal_table),
                selected_rows=sum(len(value) for value in signal_table.values()),
                elapsed=f"{perf_counter() - started_at:.2f}s",
            )

    def _build_candidate_metadata(self, row, *, entry_type: str) -> dict[str, object]:
        # 这里把执行层后面会用到的关键信息全部序列化进 metadata。
        # 包括 pivot、stop、risk、RPS、成长因子和加仓状态。
        ma_short_col = f"ma_{self.ma_short_window}"
        ma_mid_col = f"ma_{self.ma_mid_window}"
        ma_long_col = f"ma_{self.ma_long_window}"
        return {
            "entry_type": entry_type,
            "entry_reason": str(getattr(row, "setup_type", "breakout_entry") or "breakout_entry"),
            "setup_type": getattr(row, "setup_type", None),
            "signal_price_mode": self.price_mode,
            "execution_price_mode": "raw",
            "selection_score": float(row.selection_score),
            "pool_score": float(row.pool_score) if pd.notna(row.pool_score) else float(row.selection_score),
            "execution_score": float(row.execution_score) if pd.notna(row.execution_score) else None,
            "growth_leadership_score": (
                float(row.growth_leadership_score) if pd.notna(getattr(row, "growth_leadership_score", np.nan)) else None
            ),
            "fundamental_score": float(row.fundamental_score) if pd.notna(row.fundamental_score) else None,
            "pivot_proximity_score": (
                float(row.pivot_proximity_score) if pd.notna(getattr(row, "pivot_proximity_score", np.nan)) else None
            ),
            "stop_distance_score": (
                float(row.stop_distance_score) if pd.notna(getattr(row, "stop_distance_score", np.nan)) else None
            ),
            "profit_growth_score": float(row.profit_growth_score) if pd.notna(row.profit_growth_score) else None,
            "revenue_growth_score": float(row.revenue_growth_score) if pd.notna(row.revenue_growth_score) else None,
            "pivot_price": float(row.pivot_price_raw) if pd.notna(row.pivot_price_raw) else None,
            "base_low_price": float(row.base_low_price_raw) if pd.notna(row.base_low_price_raw) else None,
            "initial_stop_loss": float(row.initial_stop_loss_raw) if pd.notna(row.initial_stop_loss_raw) else None,
            "current_stop_loss": float(row.initial_stop_loss_raw) if pd.notna(row.initial_stop_loss_raw) else None,
            "risk_per_share": float(row.risk_per_share_raw) if pd.notna(row.risk_per_share_raw) else None,
            "entry_reference_price": float(row.pivot_price_raw) if pd.notna(row.pivot_price_raw) else float(row.close_raw),
            "signal_pivot_price": float(row.signal_pivot_price) if pd.notna(row.signal_pivot_price) else None,
            "signal_base_low_price": float(row.signal_base_low_price) if pd.notna(row.signal_base_low_price) else None,
            "signal_initial_stop_loss": (
                float(row.signal_initial_stop_loss) if pd.notna(row.signal_initial_stop_loss) else None
            ),
            "signal_risk_per_share": float(row.signal_risk_per_share) if pd.notna(row.signal_risk_per_share) else None,
            "risk_fraction": self.risk_fraction if entry_type == "initial" else self.add_on_risk_fraction,
            "atr_14": float(row.atr_14) if pd.notna(row.atr_14) else None,
            "atr_14_raw": float(row.atr_14_raw) if pd.notna(row.atr_14_raw) else None,
            "breakout_volume_ratio": float(row.breakout_volume_ratio) if pd.notna(row.breakout_volume_ratio) else None,
            "close_to_high_250": float(row.close_to_high_250) if pd.notna(row.close_to_high_250) else None,
            "close_to_low_250": float(row.close_to_low_250) if pd.notna(row.close_to_low_250) else None,
            "revenue_yoy": float(row.revenue_yoy) if pd.notna(row.revenue_yoy) else None,
            "net_profit_yoy": float(row.net_profit_yoy) if pd.notna(row.net_profit_yoy) else None,
            "liqaMV": float(row.liqaMV) if pd.notna(row.liqaMV) else None,
            "totalMV": float(row.totalMV) if pd.notna(row.totalMV) else None,
            f"rps_{self.rps_window_short}": float(getattr(row, f"rps_{self.rps_window_short}")),
            f"rps_{self.rps_window_long}": float(getattr(row, f"rps_{self.rps_window_long}")),
            f"ma_{self.ma_short_window}": float(getattr(row, ma_short_col)),
            f"ma_{self.ma_mid_window}": float(getattr(row, ma_mid_col)),
            f"ma_{self.ma_long_window}": float(getattr(row, ma_long_col)),
            "max_add_on_count": self.max_add_on_count,
            "add_on_trigger_r_multiples": list(self.add_on_trigger_r_multiples),
            "add_on_count": 0,
        }

    def _build_add_on_candidate(
        self,
        signal_date: datetime,
        row,
        position,
    ) -> DailyCandidate | None:
        # 只有已经盈利推进的持仓才允许加仓，加仓触发用的仍然是首仓。
        # 初始风险单位，保证 pyramiding 是“加在强势上”，不是越跌越补。
        add_on_count = int(position.metadata.get("add_on_count", 0) or 0)
        if add_on_count >= self.max_add_on_count:
            return None
        if add_on_count >= len(self.add_on_trigger_r_multiples):
            return None

        initial_risk_per_share = position.metadata.get("initial_risk_per_share")
        if initial_risk_per_share is None:
            initial_risk_per_share = position.metadata.get("risk_per_share")
        if initial_risk_per_share is None or float(initial_risk_per_share) <= 0:
            return None

        trigger_multiple = float(self.add_on_trigger_r_multiples[add_on_count])
        trigger_price = position.entry_price + float(initial_risk_per_share) * trigger_multiple
        add_on_pivot_col = f"add_on_pivot_{self.add_on_short_pivot_window}"
        add_on_low_col = f"add_on_low_{self.add_on_short_pivot_window}"
        add_on_pivot = getattr(row, f"{add_on_pivot_col}_raw", np.nan)
        add_on_low = getattr(row, f"{add_on_low_col}_raw", np.nan)
        row_close_raw = getattr(row, "close_raw", np.nan)
        row_atr_raw = getattr(row, "atr_14_raw", np.nan)
        if pd.isna(add_on_pivot) or pd.isna(add_on_low) or pd.isna(row_atr_raw):
            return None
        # 这里把 raw close 和 raw 风险单位对齐，避免加仓触发再次混用复权价。
        if pd.isna(row_close_raw) or float(row_close_raw) < trigger_price:
            return None
        if float(row_close_raw) < float(add_on_pivot) * (1.0 + self.breakout_buffer_pct):
            return None
        if pd.isna(row.volume_ratio_50) or float(row.volume_ratio_50) < self.min_add_on_volume_ratio:
            return None

        # 加仓后的 stop 只能维持不变或继续收紧，绝不能因为加仓而放宽风险。
        existing_stop = position.current_stop_loss
        if existing_stop is None:
            existing_stop = position.initial_stop_loss
        if existing_stop is None:
            existing_stop = float(add_on_low)
        add_on_stop = max(
            float(add_on_low),
            float(add_on_pivot) - float(row_atr_raw) * self.stop_atr_multiple,
            float(existing_stop),
        )
        risk_per_share = float(add_on_pivot) - add_on_stop
        if risk_per_share <= 0:
            return None
        if risk_per_share / float(add_on_pivot) > self.max_initial_stop_pct:
            return None

        metadata = self._build_candidate_metadata(row, entry_type="add_on")
        metadata.update(
            {
                "entry_reason": "add_on_breakout",
                "setup_type": "add_on_breakout",
                "pivot_price": float(add_on_pivot),
                "base_low_price": float(add_on_low),
                "initial_stop_loss": add_on_stop,
                "current_stop_loss": add_on_stop,
                "risk_per_share": risk_per_share,
                "entry_reference_price": float(add_on_pivot),
                "signal_pivot_price": float(getattr(row, add_on_pivot_col))
                if pd.notna(getattr(row, add_on_pivot_col))
                else None,
                "signal_base_low_price": float(getattr(row, add_on_low_col))
                if pd.notna(getattr(row, add_on_low_col))
                else None,
                "risk_fraction": self.add_on_risk_fraction,
                "add_on_stage": add_on_count + 1,
                "add_on_trigger_price": trigger_price,
                "breakout_volume_ratio": float(row.volume_ratio_50),
            }
        )
        return DailyCandidate(
            signal_date=signal_date,
            code=row.code,
            score=float(getattr(row, "execution_score", row.selection_score)) + 0.01 * (add_on_count + 1),
            hold_days=self.hold_days,
            metadata=metadata,
        )

    def generate_candidates(
        self,
        signal_date: datetime,
        feature_slice: pd.DataFrame,
        portfolio_snapshot: dict,
    ) -> list[DailyCandidate]:
        # 输出两类候选：
        # 1. ???????????
        # 2. ?????????????
        frame = self.signal_frame(signal_date)
        if frame is None or frame.empty:
            return []
        frame = frame.sort_values(
            [
                "execution_score",
                "selection_score",
                "growth_leadership_score",
                "fundamental_score",
                "liqaMV",
            ],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)

        positions = portfolio_snapshot.get("positions", {}) or {}
        held_codes = set(portfolio_snapshot.get("held_codes", []))
        candidates: list[DailyCandidate] = []
        new_entry_count = 0
        for row in frame.itertuples(index=False):
            position = positions.get(row.code)
            if row.code in held_codes and position is not None:
                # ????????????????????? continuation add-on?
                add_on_candidate = self._build_add_on_candidate(signal_date, row, position)
                if add_on_candidate is not None:
                    candidates.append(add_on_candidate)
                continue

            if new_entry_count >= self.top_k:
                continue
            metadata = self._build_candidate_metadata(row, entry_type="initial")
            metadata["initial_risk_per_share"] = metadata["risk_per_share"]
            candidates.append(
                DailyCandidate(
                    signal_date=signal_date,
                    code=row.code,
                    score=float(getattr(row, "execution_score", row.selection_score)),
                    hold_days=self.hold_days,
                    metadata=metadata,
                )
            )
            new_entry_count += 1
        return candidates
