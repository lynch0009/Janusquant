from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


FEATURE_VERSION = "minervini_fundamental_v1"
SOURCE_COLLECTION = "A_stock_market_akshare_quarterly_finance"
TARGET_COLLECTION = "A_stock_market_minervini_fundamental_feature"

SOURCE_FIELDS = (
    "code",
    "code_name",
    "statDate",
    "fiscalYear",
    "fiscalQuarter",
    "noticeDate",
    "updateDate",
    "source",
    "revenue_single",
    "parent_net_profit_single",
    "deduct_parent_net_profit_single",
    "basic_eps_single",
    "roe_diluted",
    "gross_margin",
    "net_margin",
    "revenue_yoy",
    "parent_net_profit_yoy",
    "deduct_parent_net_profit_yoy",
    "revenue_qoq",
    "parent_net_profit_qoq",
    "deduct_parent_net_profit_qoq",
)

FEATURE_FIELDS = (
    "code",
    "code_name",
    "statDate",
    "pubDate",
    "revisionDate",
    "fiscalYear",
    "fiscalQuarter",
    "source",
    "featureVersion",
    "computedAt",
    "revenue_single",
    "parent_net_profit_single",
    "deduct_parent_net_profit_single",
    "basic_eps_single",
    "roe_diluted",
    "gross_margin",
    "net_margin",
    "revenue_yoy",
    "parent_net_profit_yoy",
    "deduct_parent_net_profit_yoy",
    "eps_yoy",
    "revenue_qoq",
    "parent_net_profit_qoq",
    "deduct_parent_net_profit_qoq",
    "eps_qoq",
    "revenue_ttm",
    "parent_net_profit_ttm",
    "deduct_parent_net_profit_ttm",
    "eps_ttm",
    "revenue_ttm_yoy",
    "parent_net_profit_ttm_yoy",
    "deduct_parent_net_profit_ttm_yoy",
    "eps_ttm_yoy",
    "revenue_yoy_acceleration_1q",
    "profit_yoy_acceleration_1q",
    "deduct_profit_yoy_acceleration_1q",
    "eps_yoy_acceleration_1q",
    "revenue_yoy_positive_count_3q",
    "profit_yoy_positive_count_3q",
    "deduct_profit_yoy_positive_count_3q",
    "eps_yoy_positive_count_3q",
    "revenue_yoy_accelerating_count_3q",
    "profit_yoy_accelerating_count_3q",
    "deduct_profit_yoy_accelerating_count_3q",
    "eps_yoy_accelerating_count_3q",
    "has_negative_profit",
    "has_negative_deduct_profit",
    "yoy_extreme_flag",
    "low_base_flag",
    "missing_core_fields",
)

DATE_FIELDS = ("statDate", "noticeDate", "updateDate")
NUMERIC_FIELDS = tuple(field for field in SOURCE_FIELDS if field not in {"code", "code_name", "source", *DATE_FIELDS})
CORE_FIELDS = (
    "revenue_single",
    "parent_net_profit_single",
    "basic_eps_single",
    "revenue_yoy",
    "parent_net_profit_yoy",
    "roe_diluted",
    "pubDate",
)
YOY_FIELDS = (
    "revenue_yoy",
    "parent_net_profit_yoy",
    "deduct_parent_net_profit_yoy",
    "eps_yoy",
    "revenue_ttm_yoy",
    "parent_net_profit_ttm_yoy",
    "deduct_parent_net_profit_ttm_yoy",
    "eps_ttm_yoy",
)

LOW_BASE_REVENUE_ABS = 10_000_000.0
LOW_BASE_PROFIT_ABS = 1_000_000.0
LOW_BASE_EPS_ABS = 0.02
YOY_EXTREME_ABS = 10.0


def empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(FEATURE_FIELDS))


def _coerce_dates(frame: pd.DataFrame, fields: Sequence[str]) -> None:
    for field in fields:
        if field in frame.columns:
            frame[field] = pd.to_datetime(frame[field], errors="coerce")


def _coerce_numeric(frame: pd.DataFrame, fields: Sequence[str]) -> None:
    for field in fields:
        if field in frame.columns:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")


def _safe_ratio_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    current_numeric = pd.to_numeric(current, errors="coerce")
    previous_numeric = pd.to_numeric(previous, errors="coerce")
    ratio = (current_numeric / previous_numeric) - 1.0
    return ratio.where(previous_numeric.notna() & previous_numeric.ne(0.0))


def _previous_same_quarter(frame: pd.DataFrame, value_column: str) -> pd.Series:
    if value_column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    current = frame[["code", "statDate", value_column]].copy()
    current["_row_index"] = frame.index
    current["prev_statDate"] = current["statDate"] - pd.DateOffset(years=1)

    previous = frame[["code", "statDate", value_column]].rename(
        columns={"statDate": "prev_statDate", value_column: "_previous_value"}
    )
    merged = current.merge(previous, on=["code", "prev_statDate"], how="left")
    return merged.set_index("_row_index")["_previous_value"].reindex(frame.index)


def _previous_quarter(frame: pd.DataFrame, value_column: str) -> pd.Series:
    if value_column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return frame.groupby("code", sort=False)[value_column].shift(1)


def _ttm_sum(frame: pd.DataFrame, value_column: str) -> pd.Series:
    if value_column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    def calculate(group: pd.DataFrame) -> pd.Series:
        rolling_sum = group[value_column].rolling(4, min_periods=4).sum()
        consecutive = (group["_quarter_index"] - group["_quarter_index"].shift(3)).eq(3)
        return rolling_sum.where(consecutive)

    return frame.groupby("code", group_keys=False, sort=False).apply(calculate)


def _ttm_yoy(frame: pd.DataFrame, ttm_column: str) -> pd.Series:
    previous_ttm = frame.groupby("code", sort=False)[ttm_column].shift(4)
    consecutive = (frame["_quarter_index"] - frame.groupby("code", sort=False)["_quarter_index"].shift(4)).eq(4)
    return _safe_ratio_change(frame[ttm_column], previous_ttm).where(consecutive)


def _positive_count_3q(frame: pd.DataFrame, value_column: str) -> pd.Series:
    if value_column not in frame.columns:
        return pd.Series(0, index=frame.index, dtype="int64")
    positive = pd.to_numeric(frame[value_column], errors="coerce").gt(0.0).astype(int)
    return (
        positive.groupby(frame["code"], sort=False)
        .transform(lambda values: values.rolling(3, min_periods=1).sum())
        .fillna(0)
        .astype(int)
    )


def _accelerating_count_3q(frame: pd.DataFrame, acceleration_column: str) -> pd.Series:
    if acceleration_column not in frame.columns:
        return pd.Series(0, index=frame.index, dtype="int64")
    accelerating = pd.to_numeric(frame[acceleration_column], errors="coerce").gt(0.0).astype(int)
    return (
        accelerating.groupby(frame["code"], sort=False)
        .transform(lambda values: values.rolling(3, min_periods=1).sum())
        .fillna(0)
        .astype(int)
    )


def _missing_core_fields(row: pd.Series) -> list[str]:
    missing: list[str] = []
    for field in CORE_FIELDS:
        value = row.get(field)
        if pd.isna(value):
            missing.append(field)
    return missing


def _normalize_source_frame(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()

    frame = records.copy()
    for field in SOURCE_FIELDS:
        if field not in frame.columns:
            frame[field] = pd.NA

    _coerce_dates(frame, DATE_FIELDS)
    _coerce_numeric(frame, NUMERIC_FIELDS)
    frame = frame.dropna(subset=["code", "statDate", "noticeDate"]).copy()
    if frame.empty:
        return frame

    frame["pubDate"] = frame["noticeDate"]
    frame["revisionDate"] = frame["updateDate"]
    frame["fiscalYear"] = frame["fiscalYear"].fillna(frame["statDate"].dt.year).astype(int)
    frame["fiscalQuarter"] = frame["fiscalQuarter"].fillna(((frame["statDate"].dt.month - 1) // 3) + 1).astype(int)
    frame["_quarter_index"] = frame["fiscalYear"] * 4 + frame["fiscalQuarter"]

    frame = frame.drop_duplicates(subset=["code", "statDate"], keep="last")
    return frame.sort_values(["code", "statDate", "pubDate"]).reset_index(drop=True)


def build_minervini_fundamental_features(
    records: pd.DataFrame,
    *,
    computed_at: datetime | None = None,
    feature_version: str = FEATURE_VERSION,
    write_start_date: datetime | str | None = None,
    write_end_date: datetime | str | None = None,
    target_stat_dates: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Build stable Minervini fundamental features from AkShare quarterly facts."""

    frame = _normalize_source_frame(records)
    if frame.empty:
        return empty_feature_frame()

    computed_timestamp = pd.Timestamp(computed_at or datetime.now()).to_pydatetime()

    previous_revenue = _previous_same_quarter(frame, "revenue_single")
    previous_profit = _previous_same_quarter(frame, "parent_net_profit_single")
    previous_deduct_profit = _previous_same_quarter(frame, "deduct_parent_net_profit_single")
    previous_eps = _previous_same_quarter(frame, "basic_eps_single")

    frame["eps_yoy"] = _safe_ratio_change(frame["basic_eps_single"], previous_eps)
    frame["eps_qoq"] = _safe_ratio_change(frame["basic_eps_single"], _previous_quarter(frame, "basic_eps_single"))

    for value_column, ttm_column in (
        ("revenue_single", "revenue_ttm"),
        ("parent_net_profit_single", "parent_net_profit_ttm"),
        ("deduct_parent_net_profit_single", "deduct_parent_net_profit_ttm"),
        ("basic_eps_single", "eps_ttm"),
    ):
        frame[ttm_column] = _ttm_sum(frame, value_column)
        frame[f"{ttm_column}_yoy"] = _ttm_yoy(frame, ttm_column)

    frame["revenue_yoy_acceleration_1q"] = frame.groupby("code", sort=False)["revenue_yoy"].diff()
    frame["profit_yoy_acceleration_1q"] = frame.groupby("code", sort=False)["parent_net_profit_yoy"].diff()
    frame["deduct_profit_yoy_acceleration_1q"] = frame.groupby("code", sort=False)["deduct_parent_net_profit_yoy"].diff()
    frame["eps_yoy_acceleration_1q"] = frame.groupby("code", sort=False)["eps_yoy"].diff()

    frame["revenue_yoy_positive_count_3q"] = _positive_count_3q(frame, "revenue_yoy")
    frame["profit_yoy_positive_count_3q"] = _positive_count_3q(frame, "parent_net_profit_yoy")
    frame["deduct_profit_yoy_positive_count_3q"] = _positive_count_3q(frame, "deduct_parent_net_profit_yoy")
    frame["eps_yoy_positive_count_3q"] = _positive_count_3q(frame, "eps_yoy")

    frame["revenue_yoy_accelerating_count_3q"] = _accelerating_count_3q(frame, "revenue_yoy_acceleration_1q")
    frame["profit_yoy_accelerating_count_3q"] = _accelerating_count_3q(frame, "profit_yoy_acceleration_1q")
    frame["deduct_profit_yoy_accelerating_count_3q"] = _accelerating_count_3q(frame, "deduct_profit_yoy_acceleration_1q")
    frame["eps_yoy_accelerating_count_3q"] = _accelerating_count_3q(frame, "eps_yoy_acceleration_1q")

    frame["has_negative_profit"] = frame["parent_net_profit_single"].lt(0.0)
    frame["has_negative_deduct_profit"] = frame["deduct_parent_net_profit_single"].lt(0.0)

    yoy_abs = pd.concat([pd.to_numeric(frame[field], errors="coerce").abs() for field in YOY_FIELDS], axis=1)
    frame["yoy_extreme_flag"] = yoy_abs.gt(YOY_EXTREME_ABS).any(axis=1)
    frame["low_base_flag"] = (
        (frame["revenue_yoy"].abs().gt(YOY_EXTREME_ABS) & previous_revenue.abs().le(LOW_BASE_REVENUE_ABS))
        | (frame["parent_net_profit_yoy"].abs().gt(YOY_EXTREME_ABS) & previous_profit.abs().le(LOW_BASE_PROFIT_ABS))
        | (
            frame["deduct_parent_net_profit_yoy"].abs().gt(YOY_EXTREME_ABS)
            & previous_deduct_profit.abs().le(LOW_BASE_PROFIT_ABS)
        )
        | (frame["eps_yoy"].abs().gt(YOY_EXTREME_ABS) & previous_eps.abs().le(LOW_BASE_EPS_ABS))
    ).fillna(False)
    frame["missing_core_fields"] = frame.apply(_missing_core_fields, axis=1)

    frame["featureVersion"] = feature_version
    frame["computedAt"] = computed_timestamp

    if write_start_date is not None:
        frame = frame[frame["statDate"] >= pd.Timestamp(write_start_date)].copy()
    if write_end_date is not None:
        frame = frame[frame["statDate"] <= pd.Timestamp(write_end_date)].copy()
    if target_stat_dates is not None:
        target_dates = {pd.Timestamp(value).normalize() for value in target_stat_dates}
        frame = frame[frame["statDate"].dt.normalize().isin(target_dates)].copy()

    if frame.empty:
        return empty_feature_frame()

    available_fields = [field for field in FEATURE_FIELDS if field in frame.columns]
    return frame[available_fields].sort_values(["code", "statDate"]).reset_index(drop=True)
