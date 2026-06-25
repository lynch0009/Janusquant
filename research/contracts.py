"""Public extension protocols."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .models import DataRequirements, MetricResult, ResearchDataset, ResearchRequest, StudyResult


class RequirementProvider(Protocol):
    def required_fields(self, request: ResearchRequest) -> tuple[str, ...]: ...


@runtime_checkable
class DatasetBuilder(Protocol):
    cache_identity: str

    def build(self, request: ResearchRequest, requirements: DataRequirements) -> ResearchDataset: ...

    def stable_config(self, dataset_config: object) -> object: ...


@runtime_checkable
class LabelBuilder(Protocol):
    version: str

    def required_fields(self) -> tuple[str, ...]: ...

    def required_future_window(self, horizons: tuple[int, ...]) -> int: ...

    def build(self, history: pd.DataFrame, horizons: tuple[int, ...], *, key_columns: tuple[str, ...]) -> pd.DataFrame: ...


@runtime_checkable
class SampleSelector(Protocol):
    def required_fields(self, request: ResearchRequest) -> tuple[str, ...]: ...

    def select(self, panel: pd.DataFrame, request: ResearchRequest) -> pd.DataFrame: ...


@runtime_checkable
class PanelTransformer(Protocol):
    def required_fields(self, request: ResearchRequest) -> tuple[str, ...]: ...

    def transform(self, panel: pd.DataFrame, request: ResearchRequest) -> pd.DataFrame: ...


@runtime_checkable
class ResearchMetric(Protocol):
    name: str
    required_columns: tuple[str, ...]
    output_kind: str

    def compute(self, context: Any, request: ResearchRequest) -> MetricResult: ...


@runtime_checkable
class MetricSuite(Protocol):
    def data_fields(self, request: ResearchRequest) -> tuple[str, ...]: ...

    def required_fields(self, request: ResearchRequest) -> tuple[str, ...]: ...

    def compute(self, panel: pd.DataFrame, request: ResearchRequest) -> tuple[dict[str, MetricResult], dict[str, Any]]: ...


@runtime_checkable
class Reporter(Protocol):
    def export(self, result: StudyResult, request: ResearchRequest) -> Path: ...
