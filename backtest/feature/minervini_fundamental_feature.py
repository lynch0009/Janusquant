from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


FEATURE_VERSION = "minervini_fundamental_v4"
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
    "availableDate",
    "source",
    "revenue_single",
    "parent_net_profit_single",
    "deduct_parent_net_profit_single",
    "basic_eps_single",
    "revenue_ytd",
    "parent_net_profit_ytd",
    "deduct_parent_net_profit_ytd",
    "basic_eps_ytd",
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
    "noticeDate",
    "availableDate",
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
    "revenue_annual",
    "parent_net_profit_annual",
    "deduct_parent_net_profit_annual",
    "eps_annual",
    "revenue_annual_yoy",
    "parent_net_profit_annual_yoy",
    "deduct_parent_net_profit_annual_yoy",
    "eps_annual_yoy",
    "revenue_effective_long_term_yoy",
    "parent_net_profit_effective_long_term_yoy",
    "deduct_parent_net_profit_effective_long_term_yoy",
    "eps_effective_long_term_yoy",
    "revenue_growth_basis",
    "parent_net_profit_growth_basis",
    "deduct_parent_net_profit_growth_basis",
    "eps_growth_basis",
    "revenue_annual_yoy_acceleration_1y",
    "profit_annual_yoy_acceleration_1y",
    "deduct_profit_annual_yoy_acceleration_1y",
    "eps_annual_yoy_acceleration_1y",
    "revenue_annual_positive_count_3y",
    "profit_annual_positive_count_3y",
    "deduct_profit_annual_positive_count_3y",
    "eps_annual_positive_count_3y",
    "revenue_annual_accelerating_count_3y",
    "profit_annual_accelerating_count_3y",
    "deduct_profit_annual_accelerating_count_3y",
    "eps_annual_accelerating_count_3y",
    "revenue_turnaround",
    "profit_turnaround",
    "deduct_profit_turnaround",
    "eps_turnaround",
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

DATE_FIELDS = ("statDate", "noticeDate", "updateDate", "availableDate")
NUMERIC_FIELDS = tuple(field for field in SOURCE_FIELDS if field not in {"code", "code_name", "source", *DATE_FIELDS})
# 季度单季值允许缺失；真正的数值门槛由策略使用“精确 TTM / 连续年报”口径判断。
# 这里仅把公告可见日期作为结构性必需字段，避免把可兜底的数据误标成核心缺失。
CORE_FIELDS = ("pubDate",)
YOY_FIELDS = (
    "revenue_yoy",
    "parent_net_profit_yoy",
    "deduct_parent_net_profit_yoy",
    "eps_yoy",
    "revenue_ttm_yoy",
    "parent_net_profit_ttm_yoy",
    "deduct_parent_net_profit_ttm_yoy",
    "eps_ttm_yoy",
    "revenue_annual_yoy",
    "parent_net_profit_annual_yoy",
    "deduct_parent_net_profit_annual_yoy",
    "eps_annual_yoy",
    "revenue_effective_long_term_yoy",
    "parent_net_profit_effective_long_term_yoy",
    "deduct_parent_net_profit_effective_long_term_yoy",
    "eps_effective_long_term_yoy",
)

LOW_BASE_REVENUE_ABS = 10_000_000.0
LOW_BASE_PROFIT_ABS = 1_000_000.0
LOW_BASE_EPS_ABS = 0.02
YOY_EXTREME_ABS = 10.0
# 三个连续年度加速度最多依赖五份年报，故保守覆盖二十个财季的修订可见时间。
FEATURE_AVAILABILITY_LOOKBACK_QUARTERS = 20


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
    return ratio.where(previous_numeric.notna() & previous_numeric.gt(0.0))


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
    current = frame[["code", "_quarter_index"]].copy()
    current["_row_index"] = frame.index
    current["_previous_quarter_index"] = current["_quarter_index"] - 1
    previous = frame[["code", "_quarter_index", value_column]].rename(
        columns={"_quarter_index": "_previous_quarter_index", value_column: "_previous_value"}
    )
    merged = current.merge(previous, on=["code", "_previous_quarter_index"], how="left", validate="many_to_one")
    return merged.set_index("_row_index")["_previous_value"].reindex(frame.index)


def _ttm_sum(frame: pd.DataFrame, value_column: str) -> pd.Series:
    if value_column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    grouped_values = frame.groupby("code", sort=False)[value_column]
    rolling_sum = grouped_values.transform(lambda values: values.rolling(4, min_periods=4).sum())
    grouped_quarters = frame.groupby("code", sort=False)["_quarter_index"]
    consecutive = (frame["_quarter_index"] - grouped_quarters.shift(3)).eq(3)
    return rolling_sum.where(consecutive)


def _ttm_yoy(frame: pd.DataFrame, ttm_column: str) -> pd.Series:
    # 按“同一财季、相隔一年”精确关联，缺季度时不能用行号 shift 冒充同比。
    previous_ttm = _previous_same_quarter(frame, ttm_column)
    return _safe_ratio_change(frame[ttm_column], previous_ttm)


def _lookup_by_fiscal_key(
    frame: pd.DataFrame,
    value_column: str,
    *,
    year_offset: int,
    same_quarter: bool,
) -> pd.Series:
    """按财年和季度键精确取历史值，避免缺期后发生错位。"""

    if value_column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    current = frame[["code", "fiscalYear", "fiscalQuarter"]].copy()
    current["_row_index"] = frame.index
    current["_lookup_year"] = current["fiscalYear"] + year_offset
    keys = ["code", "_lookup_year"]
    previous = frame[["code", "fiscalYear", "fiscalQuarter", value_column]].rename(
        columns={"fiscalYear": "_lookup_year", value_column: "_lookup_value"}
    )
    if same_quarter:
        current["_lookup_quarter"] = current["fiscalQuarter"]
        previous = previous.rename(columns={"fiscalQuarter": "_lookup_quarter"})
        keys.append("_lookup_quarter")
    else:
        previous = previous[previous["fiscalQuarter"].eq(4)].drop(columns="fiscalQuarter")
    merged = current.merge(previous, on=keys, how="left", validate="many_to_one")
    return merged.set_index("_row_index")["_lookup_value"].reindex(frame.index)


def _build_ttm_from_cumulative(
    frame: pd.DataFrame,
    *,
    ytd_column: str,
    annual_column: str,
) -> pd.Series:
    """用累计口径构造精确 TTM，不做均值、插值或简单年化。"""

    current_ytd = pd.to_numeric(frame[ytd_column], errors="coerce")
    previous_same_ytd = _lookup_by_fiscal_key(
        frame,
        ytd_column,
        year_offset=-1,
        same_quarter=True,
    )
    previous_annual = _lookup_by_fiscal_key(
        frame,
        annual_column,
        year_offset=-1,
        same_quarter=False,
    )
    cumulative_ttm = current_ytd + previous_annual - previous_same_ytd
    # 第四季度累计值本身就是完整年报值，不需要再套累计 TTM 公式。
    return cumulative_ttm.where(frame["fiscalQuarter"].ne(4), current_ytd)


def _build_annual_fallback_metrics(
    frame: pd.DataFrame,
    annual_column: str,
) -> pd.DataFrame:
    """把最近连续年报的增长指标映射到每个财季，年度缺口会中断连续统计。"""

    result = pd.DataFrame(index=frame.index)
    for column in ("value", "yoy", "acceleration", "positive_count", "accelerating_count"):
        result[column] = np.nan

    for _, group in frame.groupby("code", sort=False):
        annual = group.loc[group["fiscalQuarter"].eq(4), ["fiscalYear", annual_column]].copy()
        annual = annual.dropna(subset=[annual_column]).drop_duplicates("fiscalYear", keep="last")
        annual = annual.sort_values("fiscalYear").reset_index(drop=True)
        if annual.empty:
            continue

        value_by_year = dict(zip(annual["fiscalYear"].astype(int), annual[annual_column]))
        yoy_by_year: dict[int, float] = {}
        acceleration_by_year: dict[int, float] = {}
        positive_count_by_year: dict[int, int] = {}
        accelerating_count_by_year: dict[int, int] = {}
        for year in sorted(value_by_year):
            previous = value_by_year.get(year - 1)
            current = value_by_year[year]
            if previous is not None and pd.notna(previous) and float(previous) > 0.0:
                yoy_by_year[year] = float(current) / float(previous) - 1.0
            if year in yoy_by_year and year - 1 in yoy_by_year:
                acceleration_by_year[year] = yoy_by_year[year] - yoy_by_year[year - 1]

            # 计数窗口最多三年，但窗口内财年必须逐年连续；遇到缺失年立即重新起算。
            yoy_years = [candidate for candidate in (year - 2, year - 1, year) if candidate in yoy_by_year]
            if yoy_years and yoy_years == list(range(yoy_years[0], year + 1)):
                positive_count_by_year[year] = sum(yoy_by_year[candidate] > 0.0 for candidate in yoy_years)
            acceleration_years = [candidate for candidate in (year - 2, year - 1, year) if candidate in acceleration_by_year]
            if acceleration_years and acceleration_years == list(range(acceleration_years[0], year + 1)):
                accelerating_count_by_year[year] = sum(
                    acceleration_by_year[candidate] > 0.0 for candidate in acceleration_years
                )

        for row_index, row in group.iterrows():
            # 非年报季只能看到上一财年的年报；否则会把当年尚未发布的 Q4 倒灌到 Q1-Q3。
            latest_allowed_year = int(row["fiscalYear"]) if int(row["fiscalQuarter"]) == 4 else int(row["fiscalYear"]) - 1
            eligible_years = [year for year in value_by_year if year <= latest_allowed_year]
            if not eligible_years:
                continue
            latest_year = max(eligible_years)
            result.at[row_index, "value"] = value_by_year[latest_year]
            result.at[row_index, "yoy"] = yoy_by_year.get(latest_year, np.nan)
            result.at[row_index, "acceleration"] = acceleration_by_year.get(latest_year, np.nan)
            result.at[row_index, "positive_count"] = positive_count_by_year.get(latest_year, np.nan)
            result.at[row_index, "accelerating_count"] = accelerating_count_by_year.get(latest_year, np.nan)
    return result


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
    frame["availableDate"] = frame[["availableDate", "noticeDate", "updateDate"]].max(axis=1).dt.normalize()
    frame = frame.dropna(subset=["code", "statDate", "availableDate"]).copy()
    if frame.empty:
        return frame

    frame["pubDate"] = frame["availableDate"]
    frame["revisionDate"] = frame["updateDate"]
    frame["fiscalYear"] = frame["fiscalYear"].fillna(frame["statDate"].dt.year).astype(int)
    frame["fiscalQuarter"] = frame["fiscalQuarter"].fillna(((frame["statDate"].dt.month - 1) // 3) + 1).astype(int)
    frame["_quarter_index"] = frame["fiscalYear"] * 4 + frame["fiscalQuarter"]

    frame = frame.sort_values(["code", "statDate", "availableDate", "revisionDate"])
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

    # 同比只允许同一报告期精确比较，且负数/零基数不计算普通增速。
    frame["revenue_yoy"] = _safe_ratio_change(frame["revenue_single"], previous_revenue)
    frame["parent_net_profit_yoy"] = _safe_ratio_change(frame["parent_net_profit_single"], previous_profit)
    frame["deduct_parent_net_profit_yoy"] = _safe_ratio_change(
        frame["deduct_parent_net_profit_single"], previous_deduct_profit
    )
    frame["eps_yoy"] = _safe_ratio_change(frame["basic_eps_single"], previous_eps)
    frame["revenue_qoq"] = _safe_ratio_change(frame["revenue_single"], _previous_quarter(frame, "revenue_single"))
    frame["parent_net_profit_qoq"] = _safe_ratio_change(
        frame["parent_net_profit_single"], _previous_quarter(frame, "parent_net_profit_single")
    )
    frame["deduct_parent_net_profit_qoq"] = _safe_ratio_change(
        frame["deduct_parent_net_profit_single"], _previous_quarter(frame, "deduct_parent_net_profit_single")
    )
    frame["eps_qoq"] = _safe_ratio_change(frame["basic_eps_single"], _previous_quarter(frame, "basic_eps_single"))

    metric_specs = (
        ("revenue", "revenue_single", "revenue_ytd", "revenue"),
        ("parent_net_profit", "parent_net_profit_single", "parent_net_profit_ytd", "profit"),
        (
            "deduct_parent_net_profit",
            "deduct_parent_net_profit_single",
            "deduct_parent_net_profit_ytd",
            "deduct_profit",
        ),
        ("eps", "basic_eps_single", "basic_eps_ytd", "eps"),
    )
    for metric, single_column, ytd_column, annual_stat_prefix in metric_specs:
        ttm_column = f"{metric}_ttm"
        annual_column = f"{metric}_annual"
        strict_ttm = _ttm_sum(frame, single_column)

        # 年报值优先取第四季度累计口径；缺失时才使用四个连续单季之和。
        frame[annual_column] = pd.to_numeric(frame[ytd_column], errors="coerce").where(
            frame["fiscalQuarter"].eq(4)
        )
        frame[annual_column] = frame[annual_column].combine_first(strict_ttm.where(frame["fiscalQuarter"].eq(4)))
        cumulative_ttm = _build_ttm_from_cumulative(
            frame,
            ytd_column=ytd_column,
            annual_column=annual_column,
        )
        frame[ttm_column] = cumulative_ttm.combine_first(strict_ttm)
        frame[f"{ttm_column}_yoy"] = _ttm_yoy(frame, ttm_column)

        annual_metrics = _build_annual_fallback_metrics(frame, annual_column)
        frame[annual_column] = annual_metrics["value"]
        frame[f"{metric}_annual_yoy"] = annual_metrics["yoy"]
        frame[f"{annual_stat_prefix}_annual_yoy_acceleration_1y"] = annual_metrics["acceleration"]
        frame[f"{annual_stat_prefix}_annual_positive_count_3y"] = annual_metrics["positive_count"]
        frame[f"{annual_stat_prefix}_annual_accelerating_count_3y"] = annual_metrics["accelerating_count"]

        effective_column = f"{metric}_effective_long_term_yoy"
        frame[effective_column] = frame[f"{ttm_column}_yoy"].combine_first(frame[f"{metric}_annual_yoy"])
        frame[f"{metric}_growth_basis"] = np.select(
            [frame[f"{ttm_column}_yoy"].notna(), frame[f"{metric}_annual_yoy"].notna()],
            ["ttm_exact", "annual_fallback"],
            default="unavailable",
        )

    frame["revenue_turnaround"] = previous_revenue.le(0.0) & frame["revenue_single"].gt(0.0)
    frame["profit_turnaround"] = previous_profit.le(0.0) & frame["parent_net_profit_single"].gt(0.0)
    frame["deduct_profit_turnaround"] = previous_deduct_profit.le(0.0) & frame[
        "deduct_parent_net_profit_single"
    ].gt(0.0)
    frame["eps_turnaround"] = previous_eps.le(0.0) & frame["basic_eps_single"].gt(0.0)

    # 季节性行业不按相邻季度判断“连续增长/加速”；旧字段保留为空仅为兼容历史表结构。
    for field in (
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
    ):
        frame[field] = np.nan

    profit_level = frame["parent_net_profit_single"].combine_first(frame["parent_net_profit_annual"])
    deduct_profit_level = frame["deduct_parent_net_profit_single"].combine_first(
        frame["deduct_parent_net_profit_annual"]
    )
    frame["has_negative_profit"] = profit_level.lt(0.0)
    frame["has_negative_deduct_profit"] = deduct_profit_level.lt(0.0)

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

    # 特征行包含 TTM、同比和年报连续性指标，可见时间不得早于任何参与计算的源报告。
    # 上游目前只保存最新修订版，因此用保守滚动窗口阻止后续修订倒灌至历史信号日。
    # DuckDB 返回的日期列通常是 datetime64[us]，而普通 pandas 输入可能是 datetime64[ns]。
    # 先显式统一到纳秒精度，避免把微秒整数按纳秒还原后落到 1970 年。
    availability_ns = frame["availableDate"].astype("datetime64[ns]").astype("int64")
    latest_available_ns = availability_ns.groupby(frame["code"], sort=False).transform(
        lambda values: values.rolling(FEATURE_AVAILABILITY_LOOKBACK_QUARTERS, min_periods=1).max()
    )
    # 财务公告从下一自然日开始可见，交易端再通过 asof 自动落到下一交易日。
    frame["availableDate"] = pd.to_datetime(latest_available_ns.astype("int64"), unit="ns") + pd.Timedelta(days=1)
    frame["pubDate"] = frame["availableDate"]
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
