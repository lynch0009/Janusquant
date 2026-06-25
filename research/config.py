"""Research configuration models and registry-independent normalization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

from .errors import ResearchConfigError


FilterOperator = Literal["==", "!=", ">", ">=", "<", "<="]


@dataclass(frozen=True)
class ValueFilterSpec:
    feature: str
    operator: FilterOperator
    value: float


@dataclass(frozen=True)
class GroupFilterSpec:
    feature: str
    direction: Literal[-1, 1]
    target_group: int
    group_count: int


SampleFilterSpec = ValueFilterSpec | GroupFilterSpec


@dataclass(frozen=True)
class ResearchSpec:
    start_date: datetime
    end_date: datetime
    horizons: tuple[int, ...] = (5, 10, 20)
    features: tuple[str, ...] = ("amount_expand",)
    group_count: int = 5
    top_pct: float = 0.2
    feature_directions: dict[str, int] | None = None
    research_mode: Literal["single_factor", "double_sort"] = "single_factor"
    primary_feature: str | None = None
    secondary_feature: str | None = None
    primary_direction: int | None = None
    secondary_direction: int | None = None
    sample_filter: SampleFilterSpec | None = None
    holding_source_dir: str | None = None
    job_name: str = "study"
    job_index: int = 1


def _direction(value: int | str | float | None, *, required: bool = False) -> int | None:
    if value is None or not str(value).strip():
        if required:
            raise ResearchConfigError("direction cannot be empty")
        return None
    numeric = float(str(value).strip())
    if not numeric.is_integer() or int(numeric) not in (-1, 1):
        raise ResearchConfigError(f"direction must be 1 or -1: {value}")
    return int(numeric)


def normalize_filter(sample_filter: SampleFilterSpec | None) -> SampleFilterSpec | None:
    if sample_filter is None:
        return None
    feature = str(sample_filter.feature).strip().lower()
    if not feature:
        raise ResearchConfigError("filter feature cannot be empty")
    if isinstance(sample_filter, ValueFilterSpec):
        operator = str(sample_filter.operator).strip()
        if operator in {"=", "==="}:
            operator = "=="
        if operator not in {"==", "!=", ">", ">=", "<", "<="}:
            raise ResearchConfigError(f"unsupported filter operator: {sample_filter.operator}")
        try:
            value = float(sample_filter.value)
        except (TypeError, ValueError) as exc:
            raise ResearchConfigError("value filter requires a numeric value") from exc
        return ValueFilterSpec(feature, operator, value)
    if isinstance(sample_filter, GroupFilterSpec):
        direction = _direction(sample_filter.direction, required=True)
        try:
            target_numeric = float(sample_filter.target_group)
            group_numeric = float(sample_filter.group_count)
        except (TypeError, ValueError) as exc:
            raise ResearchConfigError("group filter values must be integers") from exc
        if not target_numeric.is_integer() or not group_numeric.is_integer():
            raise ResearchConfigError("group filter values must be integers")
        target_group, group_count = int(target_numeric), int(group_numeric)
        if group_count < 2:
            raise ResearchConfigError("filter group_count must be >= 2")
        if not 1 <= target_group <= group_count:
            raise ResearchConfigError("target_group must be within 1..group_count")
        return GroupFilterSpec(feature, direction, target_group, group_count)
    raise ResearchConfigError(f"unsupported sample filter: {type(sample_filter)!r}")


def normalize_spec(spec: ResearchSpec) -> ResearchSpec:
    if spec.start_date > spec.end_date:
        raise ResearchConfigError("start_date must be <= end_date")
    try:
        raw_horizons = tuple(int(value) for value in spec.horizons)
    except (TypeError, ValueError) as exc:
        raise ResearchConfigError("horizons must contain positive integers") from exc
    if not raw_horizons or any(value <= 0 for value in raw_horizons):
        raise ResearchConfigError("horizons must contain positive integers")
    horizons = tuple(sorted(set(raw_horizons)))
    features = tuple(dict.fromkeys(str(value).strip().lower() for value in spec.features if str(value).strip()))
    if not features:
        raise ResearchConfigError("features cannot be empty")
    if int(spec.group_count) < 2:
        raise ResearchConfigError("group_count must be >= 2")
    if not 0.0 < float(spec.top_pct) <= 1.0:
        raise ResearchConfigError("top_pct must be within (0, 1]")
    if int(spec.job_index) <= 0:
        raise ResearchConfigError("job_index must be > 0")
    job_name = str(spec.job_name).strip()
    if not job_name:
        raise ResearchConfigError("job_name cannot be empty")
    mode = str(spec.research_mode).strip().lower()
    if mode not in {"single_factor", "double_sort"}:
        raise ResearchConfigError(f"unsupported research_mode: {spec.research_mode}")
    directions = None
    if spec.feature_directions is not None:
        directions = {
            str(name).strip().lower(): int(_direction(value, required=True))
            for name, value in spec.feature_directions.items()
        }
    primary = str(spec.primary_feature).strip().lower() if spec.primary_feature else None
    secondary = str(spec.secondary_feature).strip().lower() if spec.secondary_feature else None
    primary_direction = _direction(spec.primary_direction)
    secondary_direction = _direction(spec.secondary_direction)
    if mode == "double_sort":
        if not primary or not secondary or primary == secondary:
            raise ResearchConfigError("double_sort requires two different factor names")
        if primary not in features or secondary not in features:
            raise ResearchConfigError("double_sort factors must be included in features")
    return replace(
        spec,
        horizons=horizons,
        features=features,
        group_count=int(spec.group_count),
        top_pct=float(spec.top_pct),
        feature_directions=directions,
        research_mode=mode,
        primary_feature=primary,
        secondary_feature=secondary,
        primary_direction=primary_direction,
        secondary_direction=secondary_direction,
        sample_filter=normalize_filter(spec.sample_filter),
        job_name=job_name,
        job_index=int(spec.job_index),
    )


def required_research_features(spec: ResearchSpec) -> tuple[str, ...]:
    names = list(spec.features)
    if spec.sample_filter and spec.sample_filter.feature not in names:
        names.append(spec.sample_filter.feature)
    return tuple(names)
