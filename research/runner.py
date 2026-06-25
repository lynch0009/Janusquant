"""Research execution and shared batch orchestration."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from backtest.utils.dataframe_cache import json_ready

from .config import ResearchSpec, normalize_spec, required_research_features
from .errors import ResearchConfigError
from .factor_registry import DEFAULT_FACTOR_REGISTRY, FactorEngine, FactorRegistry
from .grouping import prepare_job_frame
from .labels import NextCloseForwardReturnLabelBuilder
from .metrics import StandardMetricSuite
from .models import BatchRequest, BatchResult, DataRequirements, MetricResult, ResearchRequest, StudyResult
from .output import StagedOutput
from .panel import PanelAssembler
from .reporting import ResearchReporter


class ResearchRunner:
    def __init__(
        self,
        *,
        factor_registry: FactorRegistry = DEFAULT_FACTOR_REGISTRY,
        label_builder=None,
        reporter=None,
    ):
        self.factor_registry = factor_registry
        self.factor_engine = FactorEngine(factor_registry)
        self.label_builder = label_builder or NextCloseForwardReturnLabelBuilder()
        self.reporter = reporter or ResearchReporter()
        self.panel_assembler = PanelAssembler()

    def normalize_request(self, request: ResearchRequest) -> tuple[ResearchRequest, dict[str, int]]:
        spec = normalize_spec(request.study)
        required_features = required_research_features(spec)
        self.factor_registry.validate_names(required_features)
        directions = self.factor_registry.default_directions(spec.features)
        if spec.feature_directions:
            unknown = set(spec.feature_directions) - set(spec.features)
            if unknown:
                raise ResearchConfigError(f"feature directions contain unknown features: {sorted(unknown)}")
            directions.update(spec.feature_directions)
        if spec.research_mode == "double_sort":
            primary_direction = spec.primary_direction or directions[str(spec.primary_feature)]
            secondary_direction = spec.secondary_direction or directions[str(spec.secondary_feature)]
            spec = replace(spec, primary_direction=primary_direction, secondary_direction=secondary_direction)
        return replace(request, study=spec), directions

    def requirements(self, request: ResearchRequest) -> DataRequirements:
        features = required_research_features(request.study)
        fields = set(self.factor_registry.required_fields(features))
        fields.update(self.label_builder.required_fields())
        fields.update(self._pipeline_input_fields(request))
        suite = request.metric_suite or StandardMetricSuite()
        fields.update(getattr(suite, "data_fields", lambda _request: ())(request))
        return DataRequirements(
            fields=tuple(sorted(fields)),
            features=features,
            horizons=request.study.horizons,
            warmup_window=self.factor_registry.max_warmup_window(features),
            future_window=self.label_builder.required_future_window(request.study.horizons),
            factor_version=self.factor_registry.version_signature(features),
            label_version=self.label_builder.version,
        )

    def prepare_panel(
        self,
        request: ResearchRequest,
        dataset_builder,
    ) -> tuple[pd.DataFrame, dict[str, float], dict[str, dict[str, int]]]:
        requirements = self.requirements(request)
        assembly_fields = set(self.factor_registry.required_fields(requirements.features))
        assembly_fields.update(self.label_builder.required_fields())
        assembly_fields.update(self._pipeline_input_fields(request))
        timings: dict[str, float] = {}
        stage_rows: dict[str, dict[str, int]] = {}
        started = perf_counter()
        dataset = dataset_builder.build(request, requirements)
        timings["build_dataset"] = perf_counter() - started
        stage_rows["build_dataset"] = {
            "input_rows": 0,
            "output_rows": len(dataset.universe),
            "history_rows": len(dataset.history),
        }
        if dataset.universe.empty or dataset.history.empty:
            columns = [*dataset.key_columns, *requirements.fields, *requirements.features,
                       *(f"fwd_ret_{value}d" for value in requirements.horizons)]
            empty = pd.DataFrame(columns=list(dict.fromkeys(columns)))
            return empty, timings, stage_rows
        started = perf_counter()
        factors = self.factor_engine.compute(dataset.history, requirements.features)
        timings["compute_factors"] = perf_counter() - started
        stage_rows["compute_factors"] = {"input_rows": len(dataset.history), "output_rows": len(factors)}
        started = perf_counter()
        labels = self.label_builder.build(dataset.history, requirements.horizons, key_columns=dataset.key_columns)
        timings["build_labels"] = perf_counter() - started
        stage_rows["build_labels"] = {"input_rows": len(dataset.history), "output_rows": len(labels)}
        started = perf_counter()
        panel = self.panel_assembler.assemble(
            dataset, factors, labels,
            required_columns=tuple(sorted(assembly_fields)),
            features=requirements.features,
            horizons=requirements.horizons,
            start_date=request.study.start_date,
            end_date=request.study.end_date,
        )
        timings["assemble_panel"] = perf_counter() - started
        stage_rows["assemble_panel"] = {
            "input_rows": len(dataset.universe),
            "output_rows": len(panel),
        }
        return panel, timings, stage_rows

    @staticmethod
    def _pipeline_input_fields(request: ResearchRequest) -> set[str]:
        fields: set[str] = set()
        produced: set[str] = {
            *required_research_features(request.study),
            *(f"fwd_ret_{value}d" for value in request.study.horizons),
            "code",
            "trade_date",
        }
        for transformer in request.transformers:
            fields.update(set(transformer.required_fields(request)) - produced)
            produced.update(getattr(transformer, "produced_fields", ()))
        for selector in request.selectors:
            fields.update(set(selector.required_fields(request)) - produced)
        return fields

    def run(self, request: ResearchRequest, *, dataset_builder, prepared_panel: pd.DataFrame | None = None) -> StudyResult:
        request, directions = self.normalize_request(request)
        timings: dict[str, float] = {}
        stage_rows: dict[str, dict[str, int]] = {}
        if prepared_panel is None:
            panel, timings, stage_rows = self.prepare_panel(request, dataset_builder)
        else:
            panel = prepared_panel.copy()
            timings["reuse_prepared_panel"] = 0.0
            stage_rows["reuse_prepared_panel"] = {"input_rows": len(panel), "output_rows": len(panel)}
        for index, transformer in enumerate(request.transformers):
            started, before = perf_counter(), len(panel)
            panel = transformer.transform(panel, request)
            timings[f"transform_{index}"] = perf_counter() - started
            stage_rows[f"transform_{index}"] = {"input_rows": before, "output_rows": len(panel)}
        for index, selector in enumerate(request.selectors):
            started, before = perf_counter(), len(panel)
            panel = selector.select(panel, request)
            timings[f"select_{index}"] = perf_counter() - started
            stage_rows[f"select_{index}"] = {"input_rows": before, "output_rows": len(panel)}
        started = perf_counter()
        job_panel = prepare_job_frame(panel, request.study, feature_directions=directions)
        timings["prepare_job_panel"] = perf_counter() - started
        stage_rows["prepare_job_panel"] = {"input_rows": len(panel), "output_rows": len(job_panel)}
        suite = request.metric_suite or StandardMetricSuite()
        started = perf_counter()
        metric_frames, summary = suite.compute(job_panel, request)
        timings["build_metrics"] = perf_counter() - started
        stage_rows["build_metrics"] = {"input_rows": len(job_panel), "output_rows": sum(
            len(value.frame) for value in metric_frames.values()
        )}
        cache = getattr(dataset_builder, "cache", None)
        result = StudyResult(
            output_dir=Path(request.output_dir),
            summary=summary,
            metric_frames=metric_frames,
            analysis_panel=job_panel,
            timings=timings,
            cache_stats=cache.summary() if cache is not None and hasattr(cache, "summary") else {},
            metadata={
                "study": asdict(request.study),
                "dataset": json_ready(request.dataset),
                "factor_version": self.factor_registry.version_signature(required_research_features(request.study)),
                "label_version": self.label_builder.version,
                "stage_rows": stage_rows,
                "selectors": [_stable_component(value) for value in request.selectors],
                "transformers": [_stable_component(value) for value in request.transformers],
            },
        )
        result.output_dir = self.reporter.export(result, request)
        return result

    def run_batch(self, batch: BatchRequest, *, dataset_builder_factory) -> BatchResult:
        if not batch.studies:
            raise ResearchConfigError("batch requires at least one study")
        normalized: list[ResearchRequest] = []
        identifiers = set()
        for request in batch.studies:
            resolved, _ = self.normalize_request(request)
            identifier = (resolved.study.job_index, resolved.study.job_name)
            if identifier in identifiers:
                raise ResearchConfigError(f"duplicate batch job identifier: {identifier}")
            identifiers.add(identifier)
            normalized.append(resolved)

        publication = StagedOutput(batch.output_dir)
        with publication as batch_staging:
            grouped: dict[str, list[ResearchRequest]] = {}
            for request in normalized:
                builder = dataset_builder_factory(request)
                key_payload = {
                    "builder": f"{type(builder).__module__}.{type(builder).__qualname__}",
                    "builder_version": builder.cache_identity,
                    "dataset": json_ready(builder.stable_config(request.dataset)),
                    "start": request.study.start_date.isoformat(),
                    "end": request.study.end_date.isoformat(),
                    "label": {
                        "type": f"{type(self.label_builder).__module__}.{type(self.label_builder).__qualname__}",
                        "version": self.label_builder.version,
                    },
                    "selectors": [_stable_component(value, strict=True) for value in request.selectors],
                    "transformers": [_stable_component(value, strict=True) for value in request.transformers],
                }
                try:
                    key = json.dumps(key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                except TypeError as exc:
                    raise ResearchConfigError("batch sharing configuration must be stably JSON serializable") from exc
                grouped.setdefault(key, []).append(request)

            studies: list[StudyResult] = []
            group_rows = []
            for group_index, requests in enumerate(grouped.values(), 1):
                builder = dataset_builder_factory(requests[0])
                shared = self._shared_request(requests, batch_staging, group_index)
                shared_panel, shared_timings, shared_rows = self.prepare_panel(shared, builder)
                group_rows.append({"shared_group_index": group_index, "job_count": len(requests),
                                   "prepared_rows": len(shared_panel), "stage_rows": json.dumps(shared_rows),
                                   **shared_timings})
                for request in requests:
                    child = replace(
                        request,
                        output_dir=batch_staging / f"{request.study.job_index:02d}_{_slug(request.study.job_name)}",
                    )
                    studies.append(self.run(child, dataset_builder=builder, prepared_panel=shared_panel))
            batch_summary = pd.DataFrame([
                {"job_index": item.metadata["study"]["job_index"], "job_name": item.metadata["study"]["job_name"],
                 "output_dir": str(publication.output_dir / item.output_dir.relative_to(batch_staging)),
                 **item.summary}
                for item in studies
            ])
            groups = pd.DataFrame(group_rows)
            batch_summary.to_csv(batch_staging / "batch_summary.csv", index=False, encoding="utf-8-sig")
            groups.to_csv(batch_staging / "batch_group_summary.csv", index=False, encoding="utf-8-sig")
            metadata_path = batch_staging / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "study_count": len(studies),
                        "shared_group_count": len(grouped),
                        "jobs": [
                            {
                                "job_index": item.metadata["study"]["job_index"],
                                "job_name": item.metadata["study"]["job_name"],
                                "output_dir": str(
                                    publication.output_dir / item.output_dir.relative_to(batch_staging)
                                ),
                            }
                            for item in studies
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (batch_staging / "summary.json").write_text(
                json.dumps(
                    {
                        "study_count": len(studies),
                        "shared_group_count": len(grouped),
                        "output_dir": str(publication.output_dir),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (batch_staging / "report.md").write_text(
                "\n".join([
                    "# 批量研究报告",
                    "",
                    f"- 任务数：{len(studies)}",
                    f"- 共享计算组数：{len(grouped)}",
                    f"- 输出目录：{publication.output_dir}",
                ]),
                encoding="utf-8",
            )
            _write_batch_manifest(batch_staging)
        final = publication.output_dir
        for item in studies:
            relative = item.output_dir.relative_to(batch_staging)
            item.output_dir = final / relative
        return BatchResult(final, studies, batch_summary, groups)

    def _shared_request(self, requests: list[ResearchRequest], output_dir: Path, index: int) -> ResearchRequest:
        first = requests[0]
        features = tuple(sorted({name for request in requests for name in required_research_features(request.study)}))
        horizons = tuple(sorted({value for request in requests for value in request.study.horizons}))
        fields_selectors = tuple(component for request in requests for component in request.selectors)
        fields_transformers = tuple(component for request in requests for component in request.transformers)
        metrics = _UnionMetricRequirements(tuple(request.metric_suite or StandardMetricSuite() for request in requests))
        return ResearchRequest(
            study=ResearchSpec(first.study.start_date, first.study.end_date, horizons=horizons, features=features,
                               job_name=f"shared_{index}", job_index=index),
            dataset=first.dataset, output_dir=output_dir,
            selectors=fields_selectors, transformers=fields_transformers, metric_suite=metrics,
        )


class _UnionMetricRequirements:
    def __init__(self, suites):
        self.suites = suites

    def required_fields(self, request):
        return tuple(sorted({field for suite in self.suites for field in suite.required_fields(request)}))

    def data_fields(self, request):
        return tuple(sorted({
            field
            for suite in self.suites
            for field in getattr(suite, "data_fields", lambda _request: ())(request)
        }))

    def compute(self, panel, request):
        return {}, {}


def _slug(value: str) -> str:
    slug = "".join(char if char.isalnum() or char in "_-" else "_" for char in value.strip())
    return slug.strip("_") or "job"


def _stable_component(component: object, *, strict: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": f"{type(component).__module__}.{type(component).__qualname__}",
        "version": getattr(component, "version", "v1"),
    }
    stable_config = getattr(component, "stable_config", None)
    if stable_config is not None:
        payload["config"] = json_ready(stable_config())
    elif strict and getattr(component, "__dict__", {}):
        raise ResearchConfigError(
            f"stateful batch component must implement stable_config(): {payload['type']}"
        )
    elif getattr(component, "__dict__", {}):
        payload["config"] = {
            key: json_ready(value)
            for key, value in component.__dict__.items()
            if not key.startswith("_") and isinstance(value, (str, int, float, bool, Path, tuple, list, dict, type(None)))
        }
    return payload


def _write_batch_manifest(root: Path) -> None:
    from datetime import datetime
    from hashlib import sha256

    files = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name == "manifest.json":
            continue
        files.append({
            "path": path.relative_to(root).as_posix(),
            "type": path.suffix.lstrip("."),
            "size": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        })
    (root / "manifest.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(), "files": files}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
