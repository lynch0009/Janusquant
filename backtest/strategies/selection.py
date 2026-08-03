"""具体选股策略实现。

这里放的是从“特征截面/历史行情”到“候选股列表”的转换逻辑。
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from backtest.execution.config import EngineConfig
from backtest.utils.datetime_utils import to_pydatetime

from .base import BaseSelectionStrategy, SignalTableSelectionStrategy
from .models import DailyCandidate


class StaticUniverseStrategy(BaseSelectionStrategy):
    """静态股票池策略：每天都从固定股票列表里生成候选。"""

    def __init__(self, codes: Sequence[str], *, hold_days: int = 1):
        """接收一组固定股票代码和默认持有交易日数。"""

        self.codes = list(codes)
        self.hold_days = hold_days

    def required_feature_fields(self) -> Sequence[str]:
        """静态股票池不依赖任何特征字段。"""

        return ()

    def generate_candidates(
        self,
        signal_date: datetime,
        feature_slice: pd.DataFrame,
        portfolio_snapshot: dict,
    ) -> list[DailyCandidate]:
        """把未持有的固定股票池成员直接转成候选股。"""

        held_codes = set(portfolio_snapshot.get("held_codes", []))
        candidates: list[DailyCandidate] = []
        for code in self.codes:
            if code in held_codes:
                continue
            candidates.append(
                DailyCandidate(
                    signal_date=signal_date,
                    code=code,
                    hold_days=self.hold_days,
                )
            )
        return candidates


class TopKFeatureStrategy(BaseSelectionStrategy):
    """
    通用的截面 Top-K 选股策略。

    它会在每天或每周的某个调仓日，从特征截面里按排序字段取出前 K 名，
    生成后续由执行器负责成交的候选股票。
    """

    def __init__(
        self,
        ranking_field: str,
        *,
        top_k: int,
        ascending: bool = False,
        hold_days: int = 1,
        rebalance_frequency: str = "D",
        rebalance_weekday: int = 0,
        min_liqa_mv: float | None = None,
        extra_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        allow_reentry: bool = False,
    ):
        """配置排序字段、调仓频率、流动性过滤和是否允许重复入场。"""

        self.ranking_field = ranking_field
        self.top_k = top_k
        self.ascending = ascending
        self.hold_days = hold_days
        self.rebalance_frequency = rebalance_frequency.upper()
        self.rebalance_weekday = rebalance_weekday
        self.min_liqa_mv = min_liqa_mv
        self.extra_filter = extra_filter
        self.allow_reentry = allow_reentry

    def required_feature_fields(self) -> Sequence[str]:
        """根据配置动态声明所需字段。"""

        fields = {"code", "date", self.ranking_field}
        if self.min_liqa_mv is not None:
            fields.add("liqaMV")
        return tuple(sorted(fields))

    def _should_rebalance(self, signal_date: datetime) -> bool:
        """判断某个交易日是否应该执行调仓。"""

        if self.rebalance_frequency == "D":
            return True
        if self.rebalance_frequency == "W":
            return signal_date.weekday() == self.rebalance_weekday
        raise ValueError(f"unsupported rebalance frequency: {self.rebalance_frequency}")

    def generate_candidates(
        self,
        signal_date: datetime,
        feature_slice: pd.DataFrame,
        portfolio_snapshot: dict,
    ) -> list[DailyCandidate]:
        """对截面打分排序，筛出当日应入场的 Top-K 候选。"""

        if not self._should_rebalance(signal_date):
            return []
        if feature_slice.empty or self.ranking_field not in feature_slice.columns:
            return []

        frame = feature_slice.copy()
        frame = frame[frame[self.ranking_field].notna()].copy()

        if self.min_liqa_mv is not None and "liqaMV" in frame.columns:
            frame = frame[frame["liqaMV"] >= self.min_liqa_mv].copy()

        if self.extra_filter is not None:
            frame = self.extra_filter(frame)

        if not self.allow_reentry:
            held_codes = set(portfolio_snapshot.get("held_codes", []))
            frame = frame[~frame["code"].isin(held_codes)].copy()

        if frame.empty:
            return []

        frame = frame.sort_values(self.ranking_field, ascending=self.ascending).head(self.top_k)
        candidates: list[DailyCandidate] = []
        for row in frame.itertuples(index=False):
            candidates.append(
                DailyCandidate(
                    signal_date=signal_date,
                    code=row.code,
                    score=getattr(row, self.ranking_field),
                    hold_days=self.hold_days,
                    metadata={"ranking_field": self.ranking_field},
                )
            )
        return candidates


class TrainLeaderStrategy(SignalTableSelectionStrategy):
    """
    火车头选股示例策略。

    核心思路来自原 notebook：
    1. 先按流动性做预筛。
    2. 再计算相对强弱、均线、创新高、回撤等指标。
    3. 最后按规则组合筛出候选，并按信号分数排序。
    """

    def __init__(
        self,
        *,
        benchmark_code: str | None = None,
        top_k: int = 10,
        hold_days: int = 3,
        min_liqa_mv: float = 5e9,
        max_liqa_mv: float | None = None,
        max_turn_mrgc: float = 25.0,
        max_turn_sxhcg: float = 15.0,
        preload_days: int = 420,
        preselect_top_n: int | None = 800,
        history_batch_size: int = 300,
    ):
        """配置基准、持有期、流动性门槛和预加载窗口。"""

        super().__init__()
        self.benchmark_code = benchmark_code or EngineConfig().benchmark_code
        self.top_k = top_k
        self.hold_days = hold_days
        self.min_liqa_mv = min_liqa_mv
        self.max_liqa_mv = max_liqa_mv
        self.max_turn_mrgc = max_turn_mrgc
        self.max_turn_sxhcg = max_turn_sxhcg
        self.preload_days = preload_days
        self.preselect_top_n = preselect_top_n
        self.history_batch_size = history_batch_size

    def required_feature_fields(self) -> Sequence[str]:
        """主循环里只需要保留策略会直接读取的特征字段。"""

        return ("code", "date", "liqaMV", "totalMV")

    @staticmethod
    def _rolling_max_drawdown(high_values: np.ndarray, low_values: np.ndarray, window: int) -> np.ndarray:
        """计算滚动窗口内的最大回撤近似值。"""

        result = np.full(len(high_values), np.nan)
        for idx in range(window - 1, len(high_values)):
            start_idx = idx - window + 1
            window_highs = high_values[start_idx : idx + 1]
            window_lows = low_values[start_idx : idx + 1]
            peak_idx = int(np.argmax(window_highs))
            peak_value = window_highs[peak_idx]
            if peak_idx < len(window_highs) - 1 and peak_value > 0:
                trough_value = np.min(window_lows[peak_idx:])
                result[idx] = (peak_value - trough_value) / peak_value
            else:
                result[idx] = 0.0
        return result

    def _build_indicator_frame(self, daily_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        """
        基于日线历史构建策略用到的技术指标表。

        这里会一次性算出相对强弱、ATR、均线、创新高、滚动回撤等指标，
        供后续选股规则直接使用。
        """

        benchmark = benchmark_df.sort_values("trade_date").copy()
        for window in (50, 120, 250):
            benchmark[f"benchmark_ret_{window}"] = benchmark["close"].pct_change(window)

        benchmark = benchmark[["trade_date", "benchmark_ret_50", "benchmark_ret_120", "benchmark_ret_250"]]

        frame = daily_df.sort_values(["code", "trade_date"]).copy()
        frame = frame.merge(benchmark, on="trade_date", how="left")
        grouped = frame.groupby("code", group_keys=False)

        for window in (50, 120, 250):
            frame[f"stock_ret_{window}"] = grouped["close"].pct_change(window)
            frame[f"RS_{window}"] = frame[f"stock_ret_{window}"] - frame[f"benchmark_ret_{window}"]
            frame[f"RS_rank_{window}"] = frame.groupby("trade_date")[f"RS_{window}"].rank(pct=True) * 100

        frame["prev_close"] = grouped["close"].shift(1)
        frame["tr_hl"] = frame["high"] - frame["low"]
        frame["tr_hc"] = (frame["high"] - frame["prev_close"]).abs()
        frame["tr_lc"] = (frame["low"] - frame["prev_close"]).abs()
        frame["true_range"] = frame[["tr_hl", "tr_hc", "tr_lc"]].max(axis=1)
        frame["atr_14"] = grouped["true_range"].transform(lambda s: s.rolling(14, min_periods=14).mean())

        frame["ma10"] = grouped["close"].transform(lambda s: s.rolling(10, min_periods=10).mean())
        frame["ma20"] = grouped["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
        frame["ma200"] = grouped["close"].transform(lambda s: s.rolling(200, min_periods=200).mean())
        frame["ma250"] = grouped["close"].transform(lambda s: s.rolling(250, min_periods=250).mean())
        frame["high_250"] = grouped["high"].transform(lambda s: s.rolling(250, min_periods=250).max())
        frame["close_250_max"] = grouped["close"].transform(lambda s: s.rolling(250, min_periods=250).max())

        frame["close_gt_ma20"] = (frame["close"] > frame["ma20"]).astype(int)
        frame["close_gt_ma10"] = (frame["close"] > frame["ma10"]).astype(int)
        frame["close_gt_ma250"] = (frame["close"] > frame["ma250"]).astype(int)
        frame["close_gt_ma200"] = (frame["close"] > frame["ma200"]).astype(int)

        frame["close_gt_ma250_count_30"] = grouped["close_gt_ma250"].transform(lambda s: s.rolling(30, min_periods=30).sum())
        frame["close_gt_ma200_count_30"] = grouped["close_gt_ma200"].transform(lambda s: s.rolling(30, min_periods=30).sum())
        frame["close_gt_ma20_count_10"] = grouped["close_gt_ma20"].transform(lambda s: s.rolling(10, min_periods=10).sum())
        frame["close_gt_ma10_count_4"] = grouped["close_gt_ma10"].transform(lambda s: s.rolling(4, min_periods=4).sum())
        frame["close_gt_ma20_count_4"] = grouped["close_gt_ma20"].transform(lambda s: s.rolling(4, min_periods=4).sum())

        frame["ma10_up"] = grouped["ma10"].diff().gt(0).astype(int)
        frame["ma20_up"] = grouped["ma20"].diff().gt(0).astype(int)
        frame["ma10_gt_ma20"] = (frame["ma10"] > frame["ma20"]).astype(int)

        frame["new_high_250"] = (frame["close"] == frame["close_250_max"]).astype(int)
        frame["new_high_250_count_5"] = grouped["new_high_250"].transform(lambda s: s.rolling(5, min_periods=1).sum())
        frame["close_to_high_250"] = frame["close"] / frame["high_250"]

        frame["max_drawdown_120"] = grouped.apply(
            lambda g: pd.Series(
                self._rolling_max_drawdown(g["high"].to_numpy(), g["low"].to_numpy(), 120),
                index=g.index,
            )
        ).reset_index(level=0, drop=True)
        frame["max_drawdown_20"] = grouped.apply(
            lambda g: pd.Series(
                self._rolling_max_drawdown(g["high"].to_numpy(), g["low"].to_numpy(), 20),
                index=g.index,
            )
        ).reset_index(level=0, drop=True)
        return frame

    def _build_feature_universe(self, feature_history: pd.DataFrame) -> pd.DataFrame:
        """按日期和流动性对股票池做预筛选，减少后续指标计算规模。"""

        if feature_history.empty:
            return feature_history

        frame = feature_history.sort_values(["date", "liqaMV"], ascending=[True, False]).copy()
        if self.preselect_top_n is not None:
            frame = (
                frame.groupby("date", group_keys=False)
                .head(self.preselect_top_n)
                .reset_index(drop=True)
            )
        return frame

    def prepare(self, data_portal, trade_dates: Sequence[datetime], *, research_store=None) -> None:
        """
        在回测开始前预计算整个区间的信号表。

        这样回测主循环里只需要按日期读取当天信号，而不用重复做大规模指标计算。
        """

        if not trade_dates:
            self.reset_signal_table()
            return

        signal_start = to_pydatetime(trade_dates[0])
        signal_end = to_pydatetime(trade_dates[-1])
        preload_start = signal_start - pd.Timedelta(days=self.preload_days)
        preload_end = signal_end + pd.Timedelta(days=1)

        feature_history = data_portal.get_feature_history(
            signal_start,
            signal_end,
            fields=["code", "date", "liqaMV", "totalMV"],
        )
        if feature_history.empty:
            self.reset_signal_table()
            return

        if self.min_liqa_mv is not None:
            feature_history = feature_history[feature_history["liqaMV"] >= self.min_liqa_mv].copy()
        if self.max_liqa_mv is not None:
            feature_history = feature_history[feature_history["liqaMV"] <= self.max_liqa_mv].copy()
        if feature_history.empty:
            self.reset_signal_table()
            return

        feature_history = self._build_feature_universe(feature_history)
        if feature_history.empty:
            self.reset_signal_table()
            return

        universe_codes = sorted(feature_history["code"].unique().tolist())
        daily_history = data_portal.get_daily_history(
            preload_start,
            preload_end,
            codes=universe_codes + [self.benchmark_code],
            fields=["code", "trade_date", "open", "high", "low", "close", "turn"],
            batch_size=self.history_batch_size,
        )
        if daily_history.empty:
            self.reset_signal_table()
            return

        benchmark_df = daily_history[daily_history["code"] == self.benchmark_code][["trade_date", "close"]].copy()
        stock_df = daily_history[daily_history["code"] != self.benchmark_code].copy()
        if benchmark_df.empty or stock_df.empty:
            self.reset_signal_table()
            return

        indicator_df = self._build_indicator_frame(stock_df, benchmark_df)
        feature_history = feature_history.rename(columns={"date": "trade_date"})
        frame = indicator_df.merge(
            feature_history[["code", "trade_date", "liqaMV", "totalMV"]],
            on=["code", "trade_date"],
            how="inner",
        )
        if frame.empty:
            self.reset_signal_table()
            return

        rps120 = frame["RS_rank_120"].fillna(0)
        rps250 = frame["RS_rank_250"].fillna(0)
        rps50 = frame["RS_rank_50"].fillna(0)
        turn = frame["turn"].fillna(100)
        close_to_high = frame["close_to_high_250"].fillna(0)
        dd120 = frame["max_drawdown_120"].fillna(1)
        dd20 = frame["max_drawdown_20"].fillna(1)

        xg1 = (frame["new_high_250_count_5"].fillna(0) >= 1) & (
            (rps120 > 95.99) | (rps250 > 95.99) | ((rps120 > 94.99) & (rps50 > 94.99))
        )
        xg2 = (close_to_high >= 0.85) & ((rps120 > 96.99) | (rps250 > 96.99))
        xg3 = (close_to_high >= 0.7) & ((rps120 > 97.99) | (rps250 > 97.99))
        xg4 = (dd120 <= 0.35) & (close_to_high >= 0.8) & ((rps120 > 94.99) | (rps250 > 94.99))

        mrgc = (turn < self.max_turn_mrgc) & (dd120 <= 0.5) & (close_to_high >= 0.7) & (xg1 | xg2 | xg3 | xg4)

        sxhcg = (
            ((rps120 + rps250) > 185)
            & (frame["close_gt_ma20"] == 1)
            & (frame["close_gt_ma250_count_30"].fillna(0) >= 25)
            & (frame["close_gt_ma200_count_30"].fillna(0) >= 25)
            & (
                (frame["close_gt_ma20_count_10"].fillna(0) >= 9)
                | (
                    (frame["close_gt_ma10_count_4"].fillna(0) >= 3)
                    & (frame["close_gt_ma20_count_4"].fillna(0) >= 3)
                )
            )
            & (dd20 <= 0.25)
            & (close_to_high > 0.8)
            & (frame["ma10_up"] == 1)
            & (frame["ma20_up"] == 1)
            & (frame["ma10_gt_ma20"] == 1)
            & (turn < self.max_turn_sxhcg)
        )

        frame["mrgc_selected"] = mrgc
        frame["sxhcg_selected"] = sxhcg
        frame["selected"] = frame["mrgc_selected"] | frame["sxhcg_selected"]
        frame["signal_score"] = rps120 + rps250 + (rps50 * 0.5)

        # 先筛出满足规则的股票，再按日期和分数降序保留每日 top_k。
        selected = frame[frame["selected"]].copy()
        if selected.empty:
            self.reset_signal_table()
            return

        selected = selected.sort_values(["trade_date", "signal_score"], ascending=[True, False])
        self.set_signal_table(self.group_signal_table(selected, limit=self.top_k))

    def generate_candidates(
        self,
        signal_date: datetime,
        feature_slice: pd.DataFrame,
        portfolio_snapshot: dict,
    ) -> list[DailyCandidate]:
        """从预计算好的 signal_table 中取出当天候选，并剔除已持仓股票。"""

        frame = self.signal_frame(signal_date)
        if frame is None or frame.empty:
            return []

        held_codes = set(portfolio_snapshot.get("held_codes", []))
        candidates: list[DailyCandidate] = []
        for row in frame.itertuples(index=False):
            if row.code in held_codes:
                continue
            candidates.append(
                DailyCandidate(
                    signal_date=signal_date,
                    code=row.code,
                    score=float(row.signal_score),
                    hold_days=self.hold_days,
                    metadata={
                        "mrgc_selected": bool(row.mrgc_selected),
                        "sxhcg_selected": bool(row.sxhcg_selected),
                        "liqaMV": float(row.liqaMV) if not pd.isna(row.liqaMV) else None,
                        "totalMV": float(row.totalMV) if not pd.isna(row.totalMV) else None,
                        "close_to_high_250": float(row.close_to_high_250) if not pd.isna(row.close_to_high_250) else None,
                        "atr_14": float(row.atr_14) if hasattr(row, "atr_14") and not pd.isna(row.atr_14) else None,
                        "RS_rank_50": float(row.RS_rank_50) if not pd.isna(row.RS_rank_50) else None,
                        "RS_rank_120": float(row.RS_rank_120) if not pd.isna(row.RS_rank_120) else None,
                        "RS_rank_250": float(row.RS_rank_250) if not pd.isna(row.RS_rank_250) else None,
                    },
                )
            )
        return candidates
