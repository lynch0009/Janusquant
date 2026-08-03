from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


REVENUE_ALIASES = (
    "MBRevenue",
    "mainBusinessRevenue",
    "operatingRevenue",
    "totalOperatingRevenue",
)
NET_PROFIT_ALIASES = (
    "netProfit",
    "parentNetProfit",
    "netProfitBelongToParent",
)
ROE_ALIASES = (
    "dupontROE",
    "roeAvg",
    "roe",
)
CFO_TO_NP_ALIASES = (
    "CFOToNP",
    "cfoToNp",
)
DIRECT_REVENUE_YOY_ALIASES = (
    "YOYOR",
    "MBRevenueYOY",
    "operatingRevenueYOY",
    "revenueYOY",
)
DIRECT_NET_PROFIT_YOY_ALIASES = (
    "YOYNI",
    "YOYPNI",
    "netProfitYOY",
    "profitYOY",
)
OPERATING_CASHFLOW_ALIASES = (
    "netOperateCashFlow",
    "operateNetCashFlow",
    "netCashFlowFromOperatingActivities",
)

FINANCE_GROWTH_SOURCE_FIELDS = tuple(
    sorted(
        {
            "code",
            "pubDate",
            "statDate",
            *REVENUE_ALIASES,
            *NET_PROFIT_ALIASES,
            *ROE_ALIASES,
            *CFO_TO_NP_ALIASES,
            *DIRECT_REVENUE_YOY_ALIASES,
            *DIRECT_NET_PROFIT_YOY_ALIASES,
            *OPERATING_CASHFLOW_ALIASES,
        }
    )
)

FINANCE_GROWTH_FACTOR_FIELDS = (
    "code",
    "pubDate",
    "statDate",
    "revenue",
    "net_profit",
    "roe",
    "cfo_to_np",
    "revenue_yoy",
    "net_profit_yoy",
    "revenue_acceleration",
    "net_profit_acceleration",
)


def _first_available_column(columns: Sequence[str], aliases: Sequence[str]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def _coerce_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _as_decimal_ratio(series: pd.Series) -> pd.Series:
    """Return finance ratio fields in their fixed decimal-unit form.

    The live finance table uses decimal ratios for YOY and ROE fields:
    0.20 == 20%, 0.08 == 8%.
    We therefore do not use any column-wide heuristic here.
    """
    return pd.to_numeric(series, errors="coerce")


def _derive_same_quarter_yoy_decimal(frame: pd.DataFrame, value_column: str) -> pd.Series:
    if value_column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="float64")

    current = frame[["code", "statDate", value_column]].copy()
    previous = current.rename(
        columns={
            "statDate": "prev_stat_date",
            value_column: "prev_value",
        }
    )
    current["prev_stat_date"] = current["statDate"] - pd.DateOffset(years=1)
    merged = current.merge(previous, on=["code", "prev_stat_date"], how="left")

    prev_value = pd.to_numeric(merged["prev_value"], errors="coerce")
    current_value = pd.to_numeric(merged[value_column], errors="coerce")
    yoy = (current_value / prev_value) - 1.0
    yoy = yoy.where(prev_value.notna() & prev_value.ne(0))
    return yoy


def build_growth_factor_timeline(reports: pd.DataFrame) -> pd.DataFrame:
    """Build a finance growth timeline in decimal units."""

    if reports.empty:
        return pd.DataFrame(columns=list(FINANCE_GROWTH_FACTOR_FIELDS))

    frame = reports.copy()
    frame["pubDate"] = pd.to_datetime(frame["pubDate"], errors="coerce")
    frame["statDate"] = pd.to_datetime(frame["statDate"], errors="coerce")
    frame = frame.dropna(subset=["code", "pubDate", "statDate"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=list(FINANCE_GROWTH_FACTOR_FIELDS))

    frame = frame.drop_duplicates(subset=["code", "pubDate", "statDate"], keep="last")
    frame = frame.sort_values(["code", "pubDate", "statDate"]).reset_index(drop=True)

    numeric_candidates = [
        column
        for column in FINANCE_GROWTH_SOURCE_FIELDS
        if column not in {"code", "pubDate", "statDate"} and column in frame.columns
    ]
    _coerce_numeric(frame, numeric_candidates)

    revenue_column = _first_available_column(frame.columns, REVENUE_ALIASES)
    net_profit_column = _first_available_column(frame.columns, NET_PROFIT_ALIASES)
    roe_column = _first_available_column(frame.columns, ROE_ALIASES)
    cfo_to_np_column = _first_available_column(frame.columns, CFO_TO_NP_ALIASES)
    operating_cashflow_column = _first_available_column(frame.columns, OPERATING_CASHFLOW_ALIASES)
    direct_revenue_yoy_column = _first_available_column(frame.columns, DIRECT_REVENUE_YOY_ALIASES)
    direct_net_profit_yoy_column = _first_available_column(frame.columns, DIRECT_NET_PROFIT_YOY_ALIASES)

    frame["revenue"] = frame[revenue_column] if revenue_column else pd.NA
    frame["net_profit"] = frame[net_profit_column] if net_profit_column else pd.NA
    frame["roe"] = _as_decimal_ratio(frame[roe_column]) if roe_column else pd.NA

    if cfo_to_np_column:
        frame["cfo_to_np"] = frame[cfo_to_np_column]
    elif operating_cashflow_column and net_profit_column:
        operating_cashflow = pd.to_numeric(frame[operating_cashflow_column], errors="coerce")
        net_profit = pd.to_numeric(frame[net_profit_column], errors="coerce")
        frame["cfo_to_np"] = (operating_cashflow / net_profit).where(net_profit.notna() & net_profit.ne(0))
    else:
        frame["cfo_to_np"] = pd.NA

    if direct_revenue_yoy_column:
        frame["revenue_yoy"] = _as_decimal_ratio(frame[direct_revenue_yoy_column])
    elif revenue_column:
        frame["revenue_yoy"] = _derive_same_quarter_yoy_decimal(frame, revenue_column)
    else:
        frame["revenue_yoy"] = pd.NA

    if direct_net_profit_yoy_column:
        frame["net_profit_yoy"] = _as_decimal_ratio(frame[direct_net_profit_yoy_column])
    elif net_profit_column:
        frame["net_profit_yoy"] = _derive_same_quarter_yoy_decimal(frame, net_profit_column)
    else:
        frame["net_profit_yoy"] = pd.NA

    frame["revenue_acceleration"] = frame.groupby("code")["revenue_yoy"].diff()
    frame["net_profit_acceleration"] = frame.groupby("code")["net_profit_yoy"].diff()

    ordered_columns = [column for column in FINANCE_GROWTH_FACTOR_FIELDS if column in frame.columns]
    return frame[ordered_columns].sort_values(["code", "pubDate", "statDate"]).reset_index(drop=True)


def build_visible_growth_factor_frame(reports: pd.DataFrame) -> pd.DataFrame:
    """Return the latest visible finance growth snapshot in decimal units."""

    timeline = build_growth_factor_timeline(reports)
    if timeline.empty:
        return timeline

    latest = timeline.groupby("code", group_keys=False).tail(1).copy()
    latest = latest.sort_values("code").reset_index(drop=True)
    return latest[[column for column in FINANCE_GROWTH_FACTOR_FIELDS if column in latest.columns]]
