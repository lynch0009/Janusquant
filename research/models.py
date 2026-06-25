"""Structured requests, datasets and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ResearchSpec


@dataclass(frozen=True)
class DataRequirements:
    fields: tuple[str, ...]
    features: tuple[str, ...]
    horizons: tuple[int, ...]
    warmup_window: int
    future_window: int
    factor_version: str
    label_version: str


@dataclass(frozen=True)
class ResearchRequest:
    study: ResearchSpec
    dataset: object
    output_dir: Path
    selectors: tuple[Any, ...] = ()
    transformers: tuple[Any, ...] = ()
    metric_suite: Any | None = None
    render_charts: bool = False
    export_panel: bool = False


@dataclass(frozen=True)
class BatchRequest:
    studies: tuple[ResearchRequest, ...]
    output_dir: Path


@dataclass
class ResearchDataset:
    universe: pd.DataFrame
    history: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    key_columns: tuple[str, ...] = ("code", "trade_date")


@dataclass
class MetricResult:
    frame: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    output_kind: str = "summary"


@dataclass
class StudyResult:
    output_dir: Path
    summary: dict[str, Any]
    metric_frames: dict[str, MetricResult]
    analysis_panel: pd.DataFrame | None
    timings: dict[str, float]
    cache_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    output_dir: Path
    studies: list[StudyResult]
    batch_summary: pd.DataFrame
    batch_group_summary: pd.DataFrame
