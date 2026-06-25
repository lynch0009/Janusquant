"""Generic sample filters, factor directions and cross-sectional grouping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GroupFilterSpec, ResearchSpec, ValueFilterSpec
from .validation import require_columns


def research_column(feature_name: str) -> str:
    return f"research_{feature_name}"


def apply_sample_filter(frame: pd.DataFrame, sample_filter) -> pd.DataFrame:
    if sample_filter is None or frame.empty:
        return frame.copy()
    require_columns(frame, (sample_filter.feature,), context="filter panel")
    values = pd.to_numeric(frame[sample_filter.feature], errors="coerce")
    if isinstance(sample_filter, ValueFilterSpec):
        masks = {
            "==": values == sample_filter.value,
            "!=": values != sample_filter.value,
            ">": values > sample_filter.value,
            ">=": values >= sample_filter.value,
            "<": values < sample_filter.value,
            "<=": values <= sample_filter.value,
        }
        mask = masks[sample_filter.operator] & values.notna()
        return frame.loc[mask.fillna(False)].copy()
    if isinstance(sample_filter, GroupFilterSpec):
        working = frame.copy()
        score = values * sample_filter.direction
        groups = working.assign(_filter_score=score).groupby("trade_date", sort=False, observed=True)
        count = groups["_filter_score"].transform("count")
        rank = groups["_filter_score"].rank(method="first")
        bucket = np.floor((rank - 1) * sample_filter.group_count / count) + 1
        mask = pd.Series(bucket, index=working.index).where(count >= sample_filter.group_count)
        return working.loc[(mask == sample_filter.target_group).fillna(False)].copy()
    raise TypeError(f"unsupported filter: {type(sample_filter)!r}")


def prepare_job_frame(
    panel: pd.DataFrame,
    spec: ResearchSpec,
    *,
    feature_directions: dict[str, int],
) -> pd.DataFrame:
    labels = tuple(f"fwd_ret_{value}d" for value in spec.horizons)
    extra = (spec.sample_filter.feature,) if spec.sample_filter and spec.sample_filter.feature not in spec.features else ()
    require_columns(panel, ("code", "trade_date", *spec.features, *extra, *labels), context="prepared research panel")
    result = panel[["code", "trade_date", *spec.features, *extra, *labels, *[
        column for column in panel.columns if column not in {"code", "trade_date", *spec.features, *extra, *labels}
    ]]].copy()
    result = apply_sample_filter(result, spec.sample_filter)
    groups = result.groupby("trade_date", sort=False, observed=True)
    for feature in spec.features:
        research = pd.to_numeric(result[feature], errors="coerce") * feature_directions[feature]
        result[research_column(feature)] = research
        count = research.groupby(result["trade_date"], sort=False, observed=True).transform("count")
        rank = research.groupby(result["trade_date"], sort=False, observed=True).rank(method="first")
        bucket = np.floor((rank - 1) * spec.group_count / count) + 1
        result[f"feature_group_{feature}"] = pd.array(
            pd.Series(bucket, index=result.index).where(count >= spec.group_count), dtype="Int16"
        )
    return result.sort_values(["trade_date", "code"], kind="mergesort").reset_index(drop=True)


def prepare_true_double_sort_frame(frame: pd.DataFrame, spec: ResearchSpec) -> pd.DataFrame:
    if frame.empty or spec.research_mode != "double_sort":
        return pd.DataFrame()
    primary, secondary = str(spec.primary_feature), str(spec.secondary_feature)
    labels = tuple(f"fwd_ret_{value}d" for value in spec.horizons)
    require_columns(frame, ("trade_date", primary, secondary, *labels), context="double-sort panel")
    result = frame[["trade_date", primary, secondary, *labels]].copy()
    result["primary_research_value"] = pd.to_numeric(result[primary], errors="coerce") * int(spec.primary_direction)
    result["secondary_research_value"] = pd.to_numeric(result[secondary], errors="coerce") * int(spec.secondary_direction)
    daily = result.groupby("trade_date", sort=False, observed=True)
    count = daily["primary_research_value"].transform("count")
    rank = daily["primary_research_value"].rank(method="first")
    result["primary_group"] = pd.array(
        pd.Series(np.floor((rank - 1) * spec.group_count / count) + 1, index=result.index).where(count >= spec.group_count),
        dtype="Int16",
    )
    secondary_groups = result.groupby(["trade_date", "primary_group"], sort=False, observed=True)
    secondary_count = secondary_groups["secondary_research_value"].transform("count")
    secondary_rank = secondary_groups["secondary_research_value"].rank(method="first")
    result["secondary_group"] = pd.array(
        pd.Series(np.floor((secondary_rank - 1) * spec.group_count / secondary_count) + 1, index=result.index).where(
            secondary_count >= spec.group_count
        ),
        dtype="Int16",
    )
    return result
