"""CLI for generic-runner small-cap factor research."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.data import DuckDBDataPortal
from backtest.db import DuckDBConfig
from backtest.utils.config_loader import parse_bool
from research.cache import ResearchFrameCache
from research.config import GroupFilterSpec, ResearchSpec, ValueFilterSpec
from research.metrics import CapBucketMetric, HoldingAttributionMetric, StandardMetricSuite
from research.models import BatchRequest, ResearchRequest
from research.runner import ResearchRunner
from research.smallcap_factor_research.config import SmallCapDatasetConfig
from research.smallcap_factor_research.dataset import (
    CapBucketTransformer,
    SmallCapEligibilitySelector,
    SmallCapResearchDatasetBuilder,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_BATCH_OUTPUT_DIR = Path(__file__).resolve().parent / "output_batch"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行小市值因子研究")
    parser.add_argument("--jobs-csv", default="")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--candidate-pool-size", type=int, default=150)
    parser.add_argument("--group-count", type=int, default=5)
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument("--features", default="amount_expand")
    parser.add_argument("--feature-directions", default="")
    parser.add_argument("--research-mode", choices=["single_factor", "double_sort"], default="single_factor")
    parser.add_argument("--primary-feature", default="")
    parser.add_argument("--secondary-feature", default="")
    parser.add_argument("--primary-direction", default="")
    parser.add_argument("--secondary-direction", default="")
    parser.add_argument("--filter-mode", choices=["none", "value", "group"], default="none")
    parser.add_argument("--filter-feature", default="")
    parser.add_argument("--filter-direction", default="")
    parser.add_argument("--filter-operator", default="")
    parser.add_argument("--filter-value", default="")
    parser.add_argument("--filter-group-count", type=int, default=0)
    parser.add_argument("--top-pct", type=float, default=0.2)
    parser.add_argument("--price-mode", choices=["raw", "qfq", "hfq"], default="hfq")
    parser.add_argument("--min-listing-trade-days", type=int, default=120)
    parser.add_argument("--include-st", action="store_true")
    parser.add_argument("--holding-source-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--render-charts", action="store_true")
    parser.add_argument("--export-panel", action="store_true")
    return parser.parse_args()


def _csv_values(value: object) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in str(value).split(",") if item.strip())


def _int_values(value: object) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in _csv_values(value))
    except ValueError as exc:
        raise ValueError(f"comma-separated integer list expected, got: {value!r}") from exc


def _optional_direction(value: object) -> int | None:
    if value is None or not str(value).strip():
        return None
    try:
        numeric = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"direction must be 1 or -1, got: {value!r}") from exc
    if not numeric.is_integer() or int(numeric) not in (-1, 1):
        raise ValueError(f"direction must be 1 or -1, got: {value!r}")
    return int(numeric)


def _directions(value: object) -> dict[str, int] | None:
    if value is None or not str(value).strip():
        return None
    result = {}
    for item in _csv_values(value):
        if ":" not in item:
            raise ValueError(f"feature direction must use feature:direction format, got: {item!r}")
        name, direction = item.split(":", 1)
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError(f"feature direction has empty feature name: {item!r}")
        parsed = _optional_direction(direction)
        if parsed is None:
            raise ValueError(f"feature direction is missing direction value: {item!r}")
        result[normalized] = parsed
    return result


def _filter(args) -> ValueFilterSpec | GroupFilterSpec | None:
    mode = str(args.filter_mode).strip().lower()
    if mode == "none":
        return None
    if not args.filter_feature:
        raise ValueError("filter-feature 不能为空")
    if mode == "value":
        if args.filter_value is None or not str(args.filter_value).strip():
            raise ValueError("value filter requires --filter-value")
        return ValueFilterSpec(args.filter_feature, args.filter_operator or "==", args.filter_value)
    if args.filter_value is None or not str(args.filter_value).strip():
        raise ValueError("group filter requires --filter-value as target group")
    if not int(args.filter_group_count):
        raise ValueError("group filter requires --filter-group-count")
    try:
        target_group = int(float(args.filter_value))
    except ValueError as exc:
        raise ValueError(f"group filter target group must be numeric, got: {args.filter_value!r}") from exc
    return GroupFilterSpec(
        args.filter_feature,
        _optional_direction(args.filter_direction) or 1,
        target_group,
        args.filter_group_count,
    )


def build_request(
    args,
    *,
    output_dir: Path,
    job_name: str = "single",
    job_index: int = 1,
    overrides: dict | None = None,
) -> ResearchRequest:
    values = overrides or {}
    get = lambda name, default: values.get(name, default)
    features = _csv_values(get("features", args.features))
    holding_dir = str(get("holding_source_dir", args.holding_source_dir) or "").strip() or None
    study = ResearchSpec(
        start_date=datetime.strptime(str(get("start_date", args.start_date)), "%Y-%m-%d"),
        end_date=datetime.strptime(str(get("end_date", args.end_date)), "%Y-%m-%d"),
        horizons=_int_values(get("horizons", args.horizons)),
        features=features,
        group_count=int(get("group_count", args.group_count)),
        top_pct=float(get("top_pct", args.top_pct)),
        feature_directions=_directions(get("feature_directions", args.feature_directions)),
        research_mode=str(get("research_mode", args.research_mode)).strip().lower(),
        primary_feature=str(get("primary_feature", args.primary_feature) or "").strip() or None,
        secondary_feature=str(get("secondary_feature", args.secondary_feature) or "").strip() or None,
        primary_direction=_optional_direction(get("primary_direction", args.primary_direction)),
        secondary_direction=_optional_direction(get("secondary_direction", args.secondary_direction)),
        sample_filter=_filter(_RowArgs(args, values)),
        holding_source_dir=holding_dir,
        job_name=job_name,
        job_index=job_index,
    )
    dataset = SmallCapDatasetConfig(
        candidate_pool_size=int(get("candidate_pool_size", args.candidate_pool_size)),
        price_mode=str(get("price_mode", args.price_mode)),
        min_listing_trade_days=int(get("min_listing_trade_days", args.min_listing_trade_days)),
        exclude_st=not parse_bool(get("include_st", args.include_st)),
    )
    metrics = list(StandardMetricSuite().metrics)
    metrics.append(CapBucketMetric())
    if holding_dir:
        metrics.append(HoldingAttributionMetric())
    return ResearchRequest(
        study=study,
        dataset=dataset,
        output_dir=Path(output_dir),
        selectors=(SmallCapEligibilitySelector(),),
        transformers=(CapBucketTransformer(),),
        metric_suite=StandardMetricSuite(metrics),
        render_charts=bool(args.render_charts),
        export_panel=bool(args.export_panel),
    )


class _RowArgs:
    def __init__(self, defaults, values):
        self._defaults, self._values = defaults, values

    def __getattr__(self, name):
        return self._values.get(name, getattr(self._defaults, name))


def load_batch(args) -> BatchRequest:
    frame = pd.read_csv(args.jobs_csv)
    if frame.empty:
        raise ValueError("jobs csv 为空")
    root = Path(args.output_dir or DEFAULT_BATCH_OUTPUT_DIR)
    studies = []
    for index, row in frame.iterrows():
        values = {name: value for name, value in row.items() if pd.notna(value)}
        name = str(values.get("job_name", f"job_{index + 1:02d}")).strip()
        studies.append(
            build_request(
                args,
                output_dir=root / f"{index + 1:02d}_{name}",
                job_name=name,
                job_index=index + 1,
                overrides=values,
            )
        )
    return BatchRequest(tuple(studies), root)


def main() -> None:
    args = parse_args()
    db_client = DuckDBConfig()
    try:
        portal = DuckDBDataPortal(db_client)
        cache = ResearchFrameCache(None if args.disable_cache else args.cache_dir)
        runner = ResearchRunner()

        def builder_factory(_request):
            return SmallCapResearchDatasetBuilder(portal, cache)

        if args.jobs_csv:
            result = runner.run_batch(load_batch(args), dataset_builder_factory=builder_factory)
        else:
            request = build_request(
                args,
                output_dir=Path(args.output_dir or DEFAULT_OUTPUT_DIR),
            )
            result = runner.run(request, dataset_builder=builder_factory(request))
        print(f"output_dir={result.output_dir}")
    finally:
        db_client.close()


if __name__ == "__main__":
    main()
