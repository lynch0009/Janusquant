"""Metric context and reusable research metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .errors import ResearchDataContractError
from .grouping import prepare_true_double_sort_frame, research_column
from .models import MetricResult, ResearchRequest
from .validation import require_columns


class MetricContext:
    def __init__(self, panel: pd.DataFrame):
        self.panel = panel
        self._numeric: dict[str, pd.Series] = {}
        self._rank: dict[str, pd.Series] = {}
        self._sorted: dict[tuple[str, str | None], pd.DataFrame] = {}

    def numeric(self, column: str) -> pd.Series:
        if column not in self._numeric:
            require_columns(self.panel, (column,), context="metric panel")
            self._numeric[column] = pd.to_numeric(self.panel[column], errors="coerce")
        return self._numeric[column]

    def daily_rank(self, column: str) -> pd.Series:
        if column not in self._rank:
            self._rank[column] = self.numeric(column).groupby(
                self.panel["trade_date"], sort=False, observed=True
            ).rank(method="average")
        return self._rank[column]

    def sorted_panel(self, score_column: str, condition_column: str | None = None) -> pd.DataFrame:
        key = (score_column, condition_column)
        if key not in self._sorted:
            require_columns(self.panel, ("trade_date", "code", score_column), context="ranking panel")
            frame = self.panel
            if condition_column:
                require_columns(frame, (condition_column,), context="ranking panel")
                frame = frame[frame[condition_column].fillna(False)]
            ranked = frame.dropna(subset=[score_column]).sort_values(
                ["trade_date", score_column, "code"],
                ascending=[True, False, True],
                kind="mergesort",
            ).copy()
            ranked["rank_desc"] = ranked.groupby("trade_date", sort=False, observed=True).cumcount() + 1
            ranked["sample_count"] = ranked.groupby("trade_date", sort=False, observed=True)[score_column].transform("size")
            self._sorted[key] = ranked
        return self._sorted[key]


class BaseMetric:
    name = ""
    required_columns: tuple[str, ...] = ()
    output_kind = "summary"


class FeatureCoverageMetric(BaseMetric):
    name = "feature_coverage"

    def compute(self, context: MetricContext, request: ResearchRequest) -> MetricResult:
        rows = []
        for feature in request.study.features:
            values = context.numeric(feature).dropna()
            rows.append(
                {
                    "feature_name": feature,
                    "sample_rows": len(context.panel),
                    "non_null_rows": len(values),
                    "coverage_ratio": len(values) / len(context.panel) if len(context.panel) else np.nan,
                    "min_value": values.min() if not values.empty else np.nan,
                    "p25": values.quantile(.25) if not values.empty else np.nan,
                    "p50": values.quantile(.50) if not values.empty else np.nan,
                    "p75": values.quantile(.75) if not values.empty else np.nan,
                    "max_value": values.max() if not values.empty else np.nan,
                }
            )
        return MetricResult(pd.DataFrame(rows))


class QuantileReturnsMetric(BaseMetric):
    name = "quantile_returns"

    def compute(self, context: MetricContext, request: ResearchRequest) -> MetricResult:
        rows = []
        for feature in request.study.features:
            group_col = f"feature_group_{feature}"
            for horizon in request.study.horizons:
                label = f"fwd_ret_{horizon}d"
                require_columns(context.panel, ("trade_date", group_col, label), context=self.name)
                working = context.panel[["trade_date", group_col, label]].dropna()
                if working.empty:
                    continue
                daily = working.groupby(["trade_date", group_col], observed=True)[label].agg(["mean", "median", "size"]).reset_index()
                summary = daily.groupby(group_col, observed=True).agg(
                    avg_forward_return=("mean", "mean"),
                    avg_daily_median=("median", "mean"),
                    avg_group_size=("size", "mean"),
                ).reset_index().rename(columns={group_col: "group_id"}).sort_values("group_id")
                spread = summary["avg_forward_return"].iloc[-1] - summary["avg_forward_return"].iloc[0]
                monotonicity = np.diff(summary["avg_forward_return"]).astype(float)
                for row in summary.itertuples(index=False):
                    rows.append({
                        "feature_name": feature, "horizon": horizon, "group_id": int(row.group_id),
                        "avg_forward_return": float(row.avg_forward_return),
                        "avg_daily_median": float(row.avg_daily_median),
                        "avg_group_size": float(row.avg_group_size),
                        "top_minus_bottom": float(spread),
                        "monotonicity_score": float((monotonicity >= 0).mean()) if monotonicity.size else np.nan,
                    })
        return MetricResult(pd.DataFrame(rows))


def _rank_ic_values(context: MetricContext, score: str, label: str, min_sample: int) -> pd.DataFrame:
    frame = context.panel[["trade_date"]].copy()
    frame["x"] = context.daily_rank(score)
    frame["y"] = context.daily_rank(label)
    frame = frame.dropna()
    if frame.empty:
        return pd.DataFrame()
    frame["x2"], frame["y2"], frame["xy"] = frame["x"] ** 2, frame["y"] ** 2, frame["x"] * frame["y"]
    stats = frame.groupby("trade_date", observed=True).agg(
        n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"),
        sx2=("x2", "sum"), sy2=("y2", "sum"), sxy=("xy", "sum"),
    ).reset_index()
    stats = stats[stats["n"] >= min_sample].copy()
    numerator = stats["sxy"] - stats["sx"] * stats["sy"] / stats["n"]
    denominator = np.sqrt(
        (stats["sx2"] - stats["sx"] ** 2 / stats["n"]) *
        (stats["sy2"] - stats["sy"] ** 2 / stats["n"])
    )
    stats["rank_ic"] = (numerator / denominator).replace([np.inf, -np.inf], np.nan)
    return stats.dropna(subset=["rank_ic"])


class RankICMetric(BaseMetric):
    name = "rank_ic"

    def compute(self, context: MetricContext, request: ResearchRequest) -> MetricResult:
        rows = []
        for feature in request.study.features:
            score = research_column(feature)
            for horizon in request.study.horizons:
                values = _rank_ic_values(context, score, f"fwd_ret_{horizon}d", max(8, request.study.group_count))
                if values.empty:
                    continue
                series = values["rank_ic"]
                std = series.std(ddof=0)
                rows.append({
                    "feature_name": feature, "horizon": horizon,
                    "rank_ic_mean": series.mean(), "rank_ic_median": series.median(),
                    "rank_ic_std": std, "icir": series.mean() / std if std > 0 else np.nan,
                    "positive_ic_ratio": (series > 0).mean(), "sample_dates": len(series),
                    "avg_sample_size": values["n"].mean(),
                })
        return MetricResult(pd.DataFrame(rows))


class FactorPersistenceMetric(BaseMetric):
    name = "factor_persistence"

    def compute(self, context: MetricContext, request: ResearchRequest) -> MetricResult:
        rows = []
        dates = pd.Index(sorted(pd.to_datetime(context.panel["trade_date"].dropna().unique())))
        date_index = {pd.Timestamp(value): index for index, value in enumerate(dates)}
        for feature in request.study.features:
            score = research_column(feature)
            ranked = context.sorted_panel(score)
            ranked["top_count"] = np.maximum(1, np.floor(ranked["sample_count"] * request.study.top_pct)).astype(int)
            selected = ranked[ranked["rank_desc"] <= ranked["top_count"]]
            sets = {pd.Timestamp(date): set(section["code"].astype(str)) for date, section in selected.groupby("trade_date", observed=True)}
            for horizon in request.study.horizons:
                ratios = []
                for date, codes in sets.items():
                    target = date_index.get(date, -1) + horizon
                    if 0 <= target < len(dates) and codes:
                        future = sets.get(pd.Timestamp(dates[target]), set())
                        ratios.append(len(codes & future) / len(codes))
                if ratios:
                    rows.append({"feature_name": feature, "horizon": horizon, "top_pct": request.study.top_pct,
                                 "persistence_ratio": np.mean(ratios), "sample_dates": len(ratios)})
        return MetricResult(pd.DataFrame(rows))


class CapBucketMetric(BaseMetric):
    name = "bucket_returns"
    required_columns = ("cap_bucket",)

    def compute(self, context: MetricContext, request: ResearchRequest) -> MetricResult:
        require_columns(context.panel, self.required_columns, context=self.name)
        rows = []
        for feature in request.study.features:
            group_col = f"feature_group_{feature}"
            for horizon in request.study.horizons:
                label = f"fwd_ret_{horizon}d"
                working = context.panel[["trade_date", "cap_bucket", group_col, label]].dropna()
                daily = working.groupby(["trade_date", "cap_bucket", group_col], observed=True)[label].mean().reset_index()
                summary = daily.groupby(["cap_bucket", group_col], observed=True)[label].mean().reset_index()
                for bucket, section in summary.groupby("cap_bucket", observed=True):
                    section = section.sort_values(group_col)
                    spread = section[label].iloc[-1] - section[label].iloc[0]
                    for row in section.itertuples(index=False):
                        rows.append({"feature_name": feature, "horizon": horizon, "cap_bucket": bucket,
                                     "group_id": int(getattr(row, group_col)), "avg_forward_return": float(getattr(row, label)),
                                     "top_minus_bottom": float(spread)})
        return MetricResult(pd.DataFrame(rows))


class DoubleSortMetric(BaseMetric):
    name = "double_sort_returns"

    def compute(self, context: MetricContext, request: ResearchRequest) -> MetricResult:
        if request.study.research_mode != "double_sort":
            return MetricResult(pd.DataFrame())
        prepared = prepare_true_double_sort_frame(context.panel, request.study)
        rows = []
        for horizon in request.study.horizons:
            label = f"fwd_ret_{horizon}d"
            working = prepared.dropna(subset=["primary_group", "secondary_group", label])
            if working.empty:
                continue
            daily = working.groupby(
                ["trade_date", "primary_group", "secondary_group"], observed=True
            )[label].mean().reset_index()
            summary = daily.groupby(
                ["primary_group", "secondary_group"], observed=True
            )[label].agg(["mean", "median", "count"]).reset_index()
            for row in summary.itertuples(index=False):
                rows.append({
                    "primary_feature": request.study.primary_feature,
                    "secondary_feature": request.study.secondary_feature,
                    "horizon": horizon,
                    "primary_group": int(row.primary_group),
                    "secondary_group": int(row.secondary_group),
                    "avg_forward_return": float(row.mean),
                    "median_forward_return": float(row.median),
                    "sample_dates": int(row.count),
                })
        return MetricResult(pd.DataFrame(rows))


class HoldingAttributionMetric(BaseMetric):
    name = "holding_attribution"

    def compute(self, context: MetricContext, request: ResearchRequest) -> MetricResult:
        source = request.study.holding_source_dir
        if not source:
            return MetricResult(pd.DataFrame())
        path = Path(source) / "closed_positions.csv"
        if not path.exists():
            raise ResearchDataContractError(f"holding file not found: {path}")
        closed = pd.read_csv(path)
        require_columns(closed, ("entry_trade_date", "code", "realized_return"), context="holding file")
        closed["entry_trade_date"] = pd.to_datetime(closed["entry_trade_date"]).dt.normalize()
        closed["realized_return"] = pd.to_numeric(closed["realized_return"], errors="coerce")
        if "holding_trade_days" in closed:
            closed["holding_trade_days"] = pd.to_numeric(closed["holding_trade_days"], errors="coerce")
        else:
            closed["holding_trade_days"] = pd.Series(np.nan, index=closed.index, dtype=float)
        rows = []
        for feature in request.study.features:
            group_col = f"feature_group_{feature}"
            merged = closed.merge(
                context.panel[["trade_date", "code", group_col]].rename(columns={"trade_date": "entry_trade_date"}),
                on=["entry_trade_date", "code"], how="left", validate="many_to_one",
            )
            unmatched = int(merged[group_col].isna().sum())
            for group_id, section in merged.dropna(subset=[group_col]).groupby(group_col, observed=True):
                returns = section["realized_return"].dropna()
                days = section["holding_trade_days"].dropna()
                long_hold = section.dropna(subset=["holding_trade_days", "realized_return"])
                rows.append({
                    "feature_name": feature, "group_id": int(group_id),
                    "return_sample_count": len(returns), "holding_days_sample_count": len(days),
                    "long_hold_sample_count": len(long_hold), "unmatched_position_count": unmatched,
                    "avg_holding_return": returns.mean(), "median_holding_return": returns.median(),
                    "win_rate": (returns > 0).mean(), "large_loss_ratio": (returns <= -.10).mean(),
                    "avg_holding_days": days.mean(),
                    "long_hold_negative_ratio": (
                        (long_hold["holding_trade_days"] >= 40) & (long_hold["realized_return"] < 0)
                    ).mean() if not long_hold.empty else np.nan,
                })
        return MetricResult(pd.DataFrame(rows))


class StandardMetricSuite:
    def __init__(self, metrics: Iterable[Any] | None = None):
        self.metrics = tuple(metrics or (
            FeatureCoverageMetric(),
            QuantileReturnsMetric(),
            RankICMetric(),
            FactorPersistenceMetric(),
            DoubleSortMetric(),
        ))

    def required_fields(self, request: ResearchRequest) -> tuple[str, ...]:
        fields = set()
        for metric in self.metrics:
            fields.update(metric.required_columns)
        return tuple(sorted(fields))

    def data_fields(self, request: ResearchRequest) -> tuple[str, ...]:
        return ()

    def compute(self, panel: pd.DataFrame, request: ResearchRequest):
        context = MetricContext(panel)
        outputs: dict[str, MetricResult] = {}
        summary: dict[str, Any] = {
            "sample_rows": len(panel),
            "trade_date_count": panel["trade_date"].nunique() if not panel.empty else 0,
            "code_count": panel["code"].nunique() if not panel.empty else 0,
            "features": list(request.study.features),
            "horizons": list(request.study.horizons),
        }
        for metric in self.metrics:
            require_columns(panel, metric.required_columns, context=metric.name)
            result = metric.compute(context, request)
            if not result.frame.empty:
                outputs[metric.name] = result
            summary.update(result.summary)
        return outputs, summary


def topn_returns(context: MetricContext, score: str, condition: str | None, horizons, top_ns) -> pd.DataFrame:
    ranked = context.sorted_panel(score, condition)
    rows = []
    for horizon in horizons:
        label = f"fwd_ret_{horizon}d"
        require_columns(ranked, (label,), context="topn_returns")
        for top_n in top_ns:
            selected = ranked[ranked["rank_desc"] <= top_n].dropna(subset=[label])
            daily = selected.groupby("trade_date", observed=True)[label].agg(["mean", "size"]).reset_index()
            rows.append({"horizon": horizon, "top_n": top_n, "date_count": len(daily),
                         "selected_count": len(selected), "avg_selected_per_date": daily["size"].mean(),
                         "date_weighted_mean": daily["mean"].mean(), "date_weighted_median": daily["mean"].median(),
                         "date_weighted_win_rate": (daily["mean"] > 0).mean(),
                         "trade_weighted_mean": selected[label].mean()})
    return pd.DataFrame(rows)


def bucket_returns(context: MetricContext, score: str, condition: str | None, horizons, bucket_count: int) -> pd.DataFrame:
    ranked = context.sorted_panel(score, condition)
    ranked = ranked[ranked["sample_count"] >= bucket_count].copy()
    ranked["bucket_id"] = np.floor((ranked["rank_desc"] - 1) * bucket_count / ranked["sample_count"]).astype(int) + 1
    rows = []
    for horizon in horizons:
        label = f"fwd_ret_{horizon}d"
        daily = ranked.dropna(subset=[label]).groupby(["trade_date", "bucket_id"], observed=True)[label].mean().reset_index()
        summary = daily.groupby("bucket_id", observed=True)[label].agg(["mean", "median", "count"]).reset_index()
        top = summary.loc[summary["bucket_id"] == 1, "mean"]
        bottom = summary.loc[summary["bucket_id"] == bucket_count, "mean"]
        spread = top.iloc[0] - bottom.iloc[0] if not top.empty and not bottom.empty else np.nan
        for row in summary.itertuples(index=False):
            rows.append({"horizon": horizon, "bucket_id": int(row.bucket_id), "bucket_count": bucket_count,
                         "date_count": int(row.count), "date_weighted_mean": float(row.mean),
                         "date_weighted_median": float(row.median), "top_minus_bottom": spread})
    return pd.DataFrame(rows)
