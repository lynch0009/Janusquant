"""Transactional unified research output."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from backtest.utils.dataframe_cache import json_ready

from .models import ResearchRequest, StudyResult
from .output import StagedOutput


class ResearchReporter:
    def export(self, result: StudyResult, request: ResearchRequest) -> Path:
        started = perf_counter()
        publication = StagedOutput(request.output_dir)
        with publication as staging:
            metrics_dir = staging / "metrics"
            metrics_dir.mkdir()
            manifest_entries = []
            for name, metric in result.metric_frames.items():
                if metric.output_kind not in {"summary", "detail"}:
                    raise ValueError(f"unsupported metric output_kind: {metric.output_kind}")
                if metric.output_kind == "detail":
                    path = metrics_dir / f"{name}.parquet"
                    metric.frame.to_parquet(path, index=False)
                else:
                    path = metrics_dir / f"{name}.csv"
                    metric.frame.to_csv(path, index=False, encoding="utf-8-sig")
                manifest_entries.append(self._entry(path, staging, metric.frame))

            if request.export_panel and result.analysis_panel is not None:
                panel_path = staging / "analysis_panel.parquet"
                result.analysis_panel.to_parquet(panel_path, index=False)
                manifest_entries.append(self._entry(panel_path, staging, result.analysis_panel))

            chart_refs = self._render_charts(staging, result) if request.render_charts else []
            for relative_path in chart_refs:
                manifest_entries.append(self._entry(staging / relative_path, staging))
            # The final JSON/Markdown/manifest writes are included by timing the complete
            # preparation work and then rewriting the small summary artifacts once.
            result.timings["export_reports"] = perf_counter() - started
            result.summary["stage_timings_seconds"] = {
                key: round(value, 4) for key, value in result.timings.items()
            }
            result.summary["total_runtime_seconds"] = round(sum(result.timings.values()), 4)
            result.summary["cache_stats"] = result.cache_stats

            summary_path = staging / "summary.json"
            summary_path.write_text(json.dumps(json_ready(result.summary), ensure_ascii=False, indent=2), encoding="utf-8")
            metadata_path = staging / "metadata.json"
            metadata_path.write_text(json.dumps(json_ready(result.metadata), ensure_ascii=False, indent=2), encoding="utf-8")
            report_path = staging / "report.md"
            report_path.write_text(self._report(result, chart_refs), encoding="utf-8")
            for path in (summary_path, metadata_path, report_path):
                manifest_entries.append(self._entry(path, staging))

            manifest_path = staging / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {"generated_at": datetime.now().isoformat(), "files": manifest_entries},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
            result.timings["export_reports"] = perf_counter() - started
            result.summary["stage_timings_seconds"]["export_reports"] = round(
                result.timings["export_reports"], 4
            )
            result.summary["total_runtime_seconds"] = round(sum(result.timings.values()), 4)
            summary_path.write_text(
                json.dumps(json_ready(result.summary), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            report_path.write_text(self._report(result, chart_refs), encoding="utf-8")
            # Refresh entries changed by the final timing update.
            changed = {"summary.json", "report.md"}
            manifest_entries = [
                self._entry(staging / entry["path"], staging)
                if entry["path"] in changed else entry
                for entry in manifest_entries
            ]
            manifest_path.write_text(
                json.dumps(
                    {"generated_at": datetime.now().isoformat(), "files": manifest_entries},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
        result.output_dir = publication.output_dir
        return publication.output_dir

    @staticmethod
    def _entry(path: Path, root: Path, frame: pd.DataFrame | None = None) -> dict[str, Any]:
        payload = {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "type": path.suffix.lstrip("."),
            "generated_at": datetime.now().isoformat(),
        }
        if frame is not None:
            payload["rows"] = len(frame)
            payload["columns"] = list(frame.columns)
        return payload

    @staticmethod
    def _report(result: StudyResult, chart_refs: list[str]) -> str:
        lines = ["# 研究报告", "", f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}", "", "## 摘要", ""]
        lines.extend(f"- {key}: {value}" for key, value in result.summary.items())
        if chart_refs:
            lines.extend(["", "## 图表", ""])
            lines.extend(f"![{Path(path).stem}]({path})" for path in chart_refs)
        return "\n".join(lines)

    @staticmethod
    def _render_charts(staging: Path, result: StudyResult) -> list[str]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return []
        coverage = result.metric_frames.get("feature_coverage")
        if coverage is None or coverage.frame.empty:
            return []
        charts = staging / "charts"
        charts.mkdir()
        path = charts / "feature_coverage.png"
        frame = coverage.frame
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(frame["feature_name"], frame["coverage_ratio"])
        ax.set_ylim(0, 1)
        ax.set_title("因子覆盖率")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return [path.relative_to(staging).as_posix()]
