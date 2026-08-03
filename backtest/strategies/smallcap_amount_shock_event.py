from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from backtest.strategies.models import DailyCandidate
from backtest.strategies.smallcap_amount_shock_reversal import SmallCapAmountShockReversalStrategy


@dataclass(frozen=True)
class AmountShockEventWindowRule:
    """高放量反转事件窗口规则。"""

    name: str
    min_signal_count: int
    pool_ret_20d_max: float | None = None
    pool_up_ratio_min: float | None = None


DEFAULT_AMOUNT_SHOCK_EVENT_RULES: tuple[AmountShockEventWindowRule, ...] = (
    AmountShockEventWindowRule(
        name="high_win_siggte4_ret20ltem12_upgtep60",
        min_signal_count=4,
        pool_ret_20d_max=-0.12,
        pool_up_ratio_min=0.60,
    ),
    AmountShockEventWindowRule(
        name="balanced_siggte3_ret20ltem12_upgtep40",
        min_signal_count=3,
        pool_ret_20d_max=-0.12,
        pool_up_ratio_min=0.40,
    ),
    AmountShockEventWindowRule(
        name="deep_wide_siggte2_ret20ltem15_upgtep30",
        min_signal_count=2,
        pool_ret_20d_max=-0.15,
        pool_up_ratio_min=0.30,
    ),
)


EVENT_SELECTION_SORTS = {
    "amount_expand_desc",
    "ret_desc",
    "cap_asc",
    "amount_then_ret",
    "ret_then_amount",
    "composite_zscore",
    "amount_ret_zscore",
}


class SmallCapAmountShockEventStrategy(SmallCapAmountShockReversalStrategy):
    """小市值高放量反转事件驱动策略。"""

    def __init__(
        self,
        *,
        top_k: int = 4,
        hold_days: int = 20,
        min_listing_trade_days: int = 120,
        candidate_pool_size: int = 100,
        cap_field: str = "liqaMV",
        amount_fast_window: int = 5,
        amount_slow_window: int = 20,
        ret_window: int = 10,
        group_count: int = 5,
        amount_keep_groups: Sequence[int] | str | None = (5,),
        ret_keep_groups: Sequence[int] | str | None = (3, 4, 5),
        min_research_ret_10d: float | None = 0.12,
        ret_top_pct: float = 0.40,
        selection_sort: str = "amount_expand_desc",
        signal_price_mode: str = "qfq",
        st_lookback_trade_days: int | None = 100,
        min_signal_close_price: float | None = 1.5,
        window_rules: Sequence[AmountShockEventWindowRule] | None = None,
        fixed_event_dates: Sequence[datetime | str] | None = None,
        precomputed_candidate_indicator_frame: pd.DataFrame | None = None,
    ) -> None:
        # 父类会先做通用小市值高放量反转参数校验；事件策略自己接管最终排序逻辑。
        super().__init__(
            top_k=top_k,
            hold_days=hold_days,
            rebalance_every_n_trade_days=1,
            min_listing_trade_days=min_listing_trade_days,
            candidate_pool_size=candidate_pool_size,
            cap_field=cap_field,
            amount_fast_window=amount_fast_window,
            amount_slow_window=amount_slow_window,
            ret_window=ret_window,
            group_count=group_count,
            amount_keep_groups=amount_keep_groups,
            ret_keep_groups=ret_keep_groups,
            min_research_ret_10d=min_research_ret_10d,
            selection_sort="ret_desc",
            signal_price_mode=signal_price_mode,
            st_lookback_trade_days=st_lookback_trade_days,
            min_signal_close_price=min_signal_close_price,
        )
        self.ret_top_pct = float(ret_top_pct)
        self.event_selection_sort = str(selection_sort or "amount_expand_desc").strip().lower()
        self.window_rules = tuple(window_rules or DEFAULT_AMOUNT_SHOCK_EVENT_RULES)
        self.fixed_event_dates = self._normalize_fixed_event_dates(fixed_event_dates)
        self.precomputed_candidate_indicator_frame = precomputed_candidate_indicator_frame
        self.event_signal_frame = pd.DataFrame()
        self.event_daily_window_features = pd.DataFrame()
        self._validate_event_parameters()

    @staticmethod
    def _normalize_fixed_event_dates(raw_dates: Sequence[datetime | str] | None) -> frozenset[pd.Timestamp] | None:
        if raw_dates is None:
            return None
        dates = pd.to_datetime(list(raw_dates), errors="coerce")
        dates = dates[pd.notna(dates)]
        return frozenset(pd.Timestamp(value).normalize() for value in dates)

    def _validate_event_parameters(self) -> None:
        if not 0 < self.ret_top_pct <= 1:
            raise ValueError("ret_top_pct must be within (0, 1].")
        if self.event_selection_sort not in EVENT_SELECTION_SORTS:
            allowed = ", ".join(sorted(EVENT_SELECTION_SORTS))
            raise ValueError(f"selection_sort must be one of: {allowed}")
        for rule in self.window_rules:
            if rule.min_signal_count < 1:
                raise ValueError("window rule min_signal_count must be >= 1")

    def _build_indicator_frame(
        self,
        data_portal,
        *,
        codes: Sequence[str],
        warmup_start: datetime,
        signal_end: datetime,
        research_store=None,
    ) -> pd.DataFrame:
        frame = super()._build_indicator_frame(
            data_portal,
            codes=codes,
            warmup_start=warmup_start,
            signal_end=signal_end,
            research_store=research_store,
        )
        if frame.empty:
            return frame
        frame = frame.sort_values(["code", "trade_date"]).reset_index(drop=True)
        grouped = frame.groupby("code", sort=False)
        for horizon in (1, 5, 20):
            shifted = grouped["close"].shift(horizon)
            frame[f"ret_{horizon}d"] = frame["close"] / shifted.where(shifted != 0) - 1.0
        return frame

    def _build_pool_window_features(self, candidate_indicator_frame: pd.DataFrame) -> pd.DataFrame:
        if candidate_indicator_frame.empty:
            return pd.DataFrame()
        frame = candidate_indicator_frame.copy()
        for column in ("ret_1d", "ret_5d", "ret_20d"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["pool_up_flag_1d"] = frame["ret_1d"] > 0
        return (
            frame.groupby("trade_date", sort=True)
            .agg(
                pool_constituent_count=("code", "size"),
                pool_ret_5d_equal=("ret_5d", "mean"),
                pool_ret_20d_equal=("ret_20d", "mean"),
                pool_up_ratio_1d=("pool_up_flag_1d", "mean"),
            )
            .reset_index()
        )

    @staticmethod
    def _apply_window_rule(daily_windows: pd.DataFrame, rule: AmountShockEventWindowRule) -> pd.Series:
        mask = pd.to_numeric(daily_windows["signal_count"], errors="coerce").fillna(0) >= rule.min_signal_count
        if rule.pool_ret_20d_max is not None:
            mask &= pd.to_numeric(daily_windows["pool_ret_20d_equal"], errors="coerce") <= rule.pool_ret_20d_max
        if rule.pool_up_ratio_min is not None:
            mask &= pd.to_numeric(daily_windows["pool_up_ratio_1d"], errors="coerce") >= rule.pool_up_ratio_min
        return mask.fillna(False)

    def _annotate_window_rules(self, daily_windows: pd.DataFrame) -> pd.DataFrame:
        windows = daily_windows.copy()
        if windows.empty:
            windows["matched_rules"] = ""
            windows["matched_rule_count"] = 0
            return windows

        rule_columns: list[str] = []
        for rule in self.window_rules:
            column = f"rule_{rule.name}"
            windows[column] = self._apply_window_rule(windows, rule)
            rule_columns.append(column)

        def matched_names(row: pd.Series) -> str:
            names = [rule.name for rule in self.window_rules if bool(row.get(f"rule_{rule.name}", False))]
            return ";".join(names)

        windows["matched_rules"] = windows.apply(matched_names, axis=1)
        windows["matched_rule_count"] = windows["matched_rules"].map(lambda value: 0 if not value else len(str(value).split(";")))
        return windows

    def _apply_ret_top_pct_filter(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        selected = []
        for _, daily in frame.groupby("trade_date", sort=True):
            sorted_daily = daily.sort_values(["research_ret_10d", "code"], ascending=[False, True], kind="mergesort").copy()
            keep_count = max(1, int(np.ceil(len(sorted_daily) * self.ret_top_pct)))
            sorted_daily["ret_top_rank"] = np.arange(1, len(sorted_daily) + 1)
            sorted_daily["ret_top_keep_count"] = keep_count
            selected.append(sorted_daily.head(keep_count))
        return pd.concat(selected, ignore_index=True) if selected else frame.iloc[0:0].copy()

    @staticmethod
    def _daily_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        grouped = numeric.groupby(frame["trade_date"])
        mean = grouped.transform("mean")
        std = grouped.transform(lambda values: values.std(ddof=0))
        score = (numeric - mean) / std.where(std != 0)
        return score.fillna(0.0)

    def _add_selection_sort_score(self, frame: pd.DataFrame, *, score: pd.Series) -> pd.DataFrame:
        result = frame.copy()
        result["selection_sort_score"] = score
        return result

    def _sort_event_signals(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.event_selection_sort in {"amount_expand_desc", "amount_then_ret"}:
            sorted_frame = frame.sort_values(
                ["trade_date", "amount_expand", "research_ret_10d", self.cap_field, "code"],
                ascending=[True, False, False, True, True],
                kind="mergesort",
            )
            return self._add_selection_sort_score(sorted_frame, score=pd.to_numeric(sorted_frame["amount_expand"], errors="coerce"))
        if self.event_selection_sort in {"ret_desc", "ret_then_amount"}:
            sorted_frame = frame.sort_values(
                ["trade_date", "research_ret_10d", "amount_expand", self.cap_field, "code"],
                ascending=[True, False, False, True, True],
                kind="mergesort",
            )
            return self._add_selection_sort_score(sorted_frame, score=pd.to_numeric(sorted_frame["research_ret_10d"], errors="coerce"))
        if self.event_selection_sort == "composite_zscore":
            working = self._ensure_raw_close_column(frame).copy()
            working["selection_sort_score"] = (
                self._daily_zscore(working, "research_ret_10d")
                + self._daily_zscore(working, "amount_expand")
                - self._daily_zscore(working, self.cap_field)
                - self._daily_zscore(working, "raw_close")
            )
            return working.sort_values(
                ["trade_date", "selection_sort_score", "research_ret_10d", "amount_expand", self.cap_field, "code"],
                ascending=[True, False, False, False, True, True],
                kind="mergesort",
            )
        if self.event_selection_sort == "amount_ret_zscore":
            working = frame.copy()
            working["selection_sort_score"] = self._daily_zscore(working, "amount_expand") + self._daily_zscore(working, "research_ret_10d")
            return working.sort_values(
                ["trade_date", "selection_sort_score", "amount_expand", "research_ret_10d", self.cap_field, "code"],
                ascending=[True, False, False, False, True, True],
                kind="mergesort",
            )
        sorted_frame = frame.sort_values(["trade_date", self.cap_field, "code"], ascending=[True, True, True], kind="mergesort")
        return self._add_selection_sort_score(sorted_frame, score=-pd.to_numeric(sorted_frame[self.cap_field], errors="coerce"))

    def _limit_daily_signals(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        limited = frame.groupby("trade_date", group_keys=False, sort=True).head(self.top_k).copy()
        limited["event_signal_rank"] = limited.groupby("trade_date").cumcount() + 1
        return limited.reset_index(drop=True)

    def _slice_precomputed_candidate_indicator(self, trade_dates: Sequence[datetime]) -> pd.DataFrame:
        if self.precomputed_candidate_indicator_frame is None:
            return pd.DataFrame()
        frame = self.precomputed_candidate_indicator_frame
        if frame.empty:
            return pd.DataFrame()

        trade_date_set = set(pd.to_datetime(list(trade_dates)).normalize())
        working = frame.copy()
        working["trade_date"] = pd.to_datetime(working["trade_date"]).dt.normalize()
        working = working[working["trade_date"].isin(trade_date_set)].copy()
        working = self._ensure_raw_close_column(working)
        if working.empty:
            return working
        return (
            working.sort_values(["trade_date", self.cap_field, "code"], ascending=[True, True, True], kind="mergesort")
            .groupby("trade_date", group_keys=False, sort=True)
            .head(self.candidate_pool_size)
            .reset_index(drop=True)
        )

    def _build_signal_table(self, frame: pd.DataFrame) -> dict[datetime, pd.DataFrame]:
        # 事件策略在 prepare 中已经完成窗口规则、ret_top_pct 和 top_k 截断。
        return self.group_signal_table(frame, limit=None)

    def prepare(self, data_portal, trade_dates: Sequence[datetime], *, research_store=None) -> None:
        self.reset_signal_table()
        self.event_signal_frame = pd.DataFrame()
        self.event_daily_window_features = pd.DataFrame()
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

        candidate_indicator = self._ensure_raw_close_column(candidate_indicator)
        pool_windows = self._build_pool_window_features(candidate_indicator)
        frame = candidate_indicator.copy()
        frame = frame[frame["isST"].fillna(False) == False].copy()
        if self.st_lookback_trade_days:
            frame = frame[frame["is_st_lookback_flag"].fillna(False) == False].copy()
        if self.min_signal_close_price is not None:
            frame = frame[pd.to_numeric(frame["raw_close"], errors="coerce") > self.min_signal_close_price].copy()

        required_factor_columns = ["amount_expand", "ret_10d", "research_ret_10d"]
        frame = frame.dropna(subset=required_factor_columns).copy()
        if frame.empty:
            return
        frame = self._assign_groups(frame)
        frame = self._apply_signal_filters(frame)

        signal_counts = frame.groupby("trade_date").size().rename("signal_count").reset_index() if not frame.empty else pd.DataFrame(columns=["trade_date", "signal_count"])
        daily_windows = pool_windows.merge(signal_counts, on="trade_date", how="left")
        daily_windows["signal_count"] = daily_windows["signal_count"].fillna(0).astype(int)
        daily_windows = self._annotate_window_rules(daily_windows)
        if self.fixed_event_dates is not None:
            daily_windows["fixed_event_date"] = pd.to_datetime(daily_windows["trade_date"]).dt.normalize().isin(self.fixed_event_dates)
        self.event_daily_window_features = daily_windows.copy()

        if frame.empty:
            return
        rule_columns = ["trade_date", "matched_rules", "matched_rule_count"]
        rule_columns.extend([column for column in daily_windows.columns if column.startswith("rule_")])
        window_feature_columns = [
            "pool_constituent_count",
            "pool_ret_5d_equal",
            "pool_ret_20d_equal",
            "pool_up_ratio_1d",
            "signal_count",
            "fixed_event_date",
        ]
        rule_columns.extend([column for column in window_feature_columns if column in daily_windows.columns])
        rule_columns = list(dict.fromkeys(rule_columns))
        frame = frame.merge(daily_windows[rule_columns], on="trade_date", how="left")
        if self.fixed_event_dates is not None:
            frame = frame[pd.to_datetime(frame["trade_date"]).dt.normalize().isin(self.fixed_event_dates)].copy()
        else:
            frame = frame[pd.to_numeric(frame["matched_rule_count"], errors="coerce").fillna(0) > 0].copy()
        if frame.empty:
            return

        frame = self._apply_ret_top_pct_filter(frame)
        frame = self._sort_event_signals(frame).copy()
        frame["event_signal_rank"] = frame.groupby("trade_date").cumcount() + 1
        frame["ret_top_pct"] = self.ret_top_pct
        frame["selection_sort"] = self.event_selection_sort
        self.event_signal_frame = frame.copy()
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

        held_codes = set(portfolio_snapshot.get("held_codes", []))
        candidates: list[DailyCandidate] = []
        for row in frame.itertuples(index=False):
            code = str(row.code)
            if code in held_codes:
                continue
            metadata = {
                self.cap_field: float(getattr(row, self.cap_field)),
                "close": float(row.close),
                "entry_reason": "amount_shock_event_entry",
                "ret_top_pct": self.ret_top_pct,
                "selection_sort": self.event_selection_sort,
                "matched_rules": str(getattr(row, "matched_rules", "")),
                "matched_rule_count": int(getattr(row, "matched_rule_count", 0)),
                "event_signal_rank": int(getattr(row, "event_signal_rank", 0)),
            }
            self._safe_float_metadata(metadata, "raw_close", getattr(row, "raw_close", None))
            for key in (
                "amount_group",
                "ret_group",
                "signal_count",
                "pool_constituent_count",
                "ret_top_rank",
                "ret_top_keep_count",
            ):
                value = getattr(row, key, None)
                if pd.notna(value):
                    metadata[key] = int(value)
            for key in (
                "amount_expand",
                "ret_10d",
                "research_ret_10d",
                "pool_ret_5d_equal",
                "pool_ret_20d_equal",
                "pool_up_ratio_1d",
                "selection_sort_score",
                self.amount_fast_col,
                self.amount_slow_col,
            ):
                self._safe_float_metadata(metadata, key, getattr(row, key, None))
            candidates.append(
                DailyCandidate(
                    signal_date=signal_date,
                    code=code,
                    score=float(getattr(row, "selection_sort_score", getattr(row, "amount_expand"))),
                    hold_days=self.hold_days,
                    metadata=metadata,
                )
            )
            if len(candidates) >= self.top_k:
                break
        return candidates
