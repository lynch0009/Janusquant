"""Cross-sectional scores and condition flags for weekly regime research."""

from __future__ import annotations

import pandas as pd

from .config import CONDITION_NAMES
from research.models import ResearchRequest
from research.validation import require_columns


def daily_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    grouped = numeric.groupby(frame["trade_date"])
    mean = grouped.transform("mean")
    std = grouped.transform("std", ddof=0)
    return ((numeric - mean) / std.where(std != 0)).fillna(0.0)


def add_factor_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["current_composite_zscore"] = (
        daily_zscore(result, "research_ret_10d")
        + daily_zscore(result, "amount_expand")
        - daily_zscore(result, "liqaMV")
        - daily_zscore(result, "close")
    )
    result["ret20_desc"] = pd.to_numeric(result["ret_20d"], errors="coerce")
    result["ret60_desc"] = pd.to_numeric(result["ret_60d"], errors="coerce")
    result["rs20_desc"] = pd.to_numeric(result["rs20_vs_399303"], errors="coerce")
    result["amount_expand_desc"] = pd.to_numeric(result["amount_expand"], errors="coerce")
    result["smallcap_trend"] = (
        daily_zscore(result, "ret_20d")
        + daily_zscore(result, "rs20_vs_399303")
        + daily_zscore(result, "amount_expand")
        - daily_zscore(result, "liqaMV")
    )
    result["trend_composite"] = (
        daily_zscore(result, "ret_20d")
        + daily_zscore(result, "ret_60d")
        + daily_zscore(result, "rs20_vs_399303")
        + daily_zscore(result, "amount_expand")
        - daily_zscore(result, "liqaMV")
    )
    result["trend_no_amount"] = (
        daily_zscore(result, "ret_20d")
        + daily_zscore(result, "ret_60d")
        + daily_zscore(result, "rs20_vs_399303")
        - daily_zscore(result, "liqaMV")
    )
    result["trend_no_cap"] = (
        daily_zscore(result, "ret_20d")
        + daily_zscore(result, "ret_60d")
        + daily_zscore(result, "rs20_vs_399303")
        + daily_zscore(result, "amount_expand")
    )
    return result


def add_condition_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    close = pd.to_numeric(result["close"], errors="coerce")
    ma10 = pd.to_numeric(result["ma10"], errors="coerce")
    ma20 = pd.to_numeric(result["ma20"], errors="coerce")
    ma20_slope = pd.to_numeric(result["ma20_slope_5d"], errors="coerce")
    ret20 = pd.to_numeric(result["ret_20d"], errors="coerce")
    rs20 = pd.to_numeric(result["rs20_vs_399303"], errors="coerce")
    result["condition_none"] = True
    result["condition_close_gt_ma20"] = close > ma20
    result["condition_close_gt_ma10_and_ma20"] = (close > ma10) & (close > ma20)
    result["condition_ma20_slope_gt_0"] = ma20_slope > 0
    result["condition_ret20_gt_0"] = ret20 > 0
    result["condition_rs20_gt_0"] = rs20 > 0
    result["condition_close_gt_ma20_and_rs20_gt_0"] = (close > ma20) & (rs20 > 0)
    result["condition_trend_quality_all"] = (
        (close > ma10) & (close > ma20) & (ma20_slope > 0) & (ret20 > 0) & (rs20 > 0)
    )
    for name in CONDITION_NAMES:
        result[f"condition_{name}"] = result[f"condition_{name}"].fillna(False).astype(bool)
    return result


class WeeklyScoreTransformer:
    version = "weekly_scores_v2"
    produced_fields = (
        "research_ret_10d",
        "rs20_vs_399303",
        "current_composite_zscore",
        "ret20_desc",
        "ret60_desc",
        "rs20_desc",
        "amount_expand_desc",
        "smallcap_trend",
        "trend_composite",
        "trend_no_amount",
        "trend_no_cap",
    )

    @staticmethod
    def stable_config() -> dict:
        return {}

    def required_fields(self, request: ResearchRequest) -> tuple[str, ...]:
        return (
            "ret_10d", "ret_20d", "ret_60d", "amount_expand", "liqaMV", "close",
            "rs20_vs_index",
        )

    def transform(self, panel: pd.DataFrame, request: ResearchRequest) -> pd.DataFrame:
        require_columns(panel, self.required_fields(request), context="weekly score transformer")
        result = panel.copy()
        result["research_ret_10d"] = -pd.to_numeric(result["ret_10d"], errors="coerce")
        # Keep score formulas independent of the concrete index code.
        result["rs20_vs_399303"] = pd.to_numeric(result["rs20_vs_index"], errors="coerce")
        return add_factor_scores(result)


class ConditionFlagTransformer:
    version = "weekly_conditions_v2"
    produced_fields = tuple(f"condition_{name}" for name in CONDITION_NAMES)

    @staticmethod
    def stable_config() -> dict:
        return {}

    def required_fields(self, request: ResearchRequest) -> tuple[str, ...]:
        return ("close", "ma10", "ma20", "ma20_slope_5d", "ret_20d", "rs20_vs_index")

    def transform(self, panel: pd.DataFrame, request: ResearchRequest) -> pd.DataFrame:
        require_columns(panel, self.required_fields(request), context="weekly condition transformer")
        result = panel.copy()
        result["rs20_vs_399303"] = pd.to_numeric(result["rs20_vs_index"], errors="coerce")
        return add_condition_flags(result)


__all__ = [
    "ConditionFlagTransformer",
    "WeeklyScoreTransformer",
    "add_condition_flags",
    "add_factor_scores",
    "daily_zscore",
]
