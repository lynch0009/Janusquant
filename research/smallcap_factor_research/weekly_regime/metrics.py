"""Metrics for weekly regime research, computed from one shared ranked panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.models import MetricResult, ResearchRequest
from research.validation import require_columns

from .config import CONDITION_NAMES, FACTOR_NAMES, FOCUS_PERIODS


class WeeklyMetricSuite:
    """Rank once per weekday/condition/factor and reuse it for every horizon."""

    def __init__(
        self,
        *,
        top_ns: tuple[int, ...],
        bucket_count: int,
        regime_run_dir: Path,
        factor_names: tuple[str, ...] = FACTOR_NAMES,
        condition_names: tuple[str, ...] = CONDITION_NAMES,
    ):
        self.top_ns = tuple(sorted({int(value) for value in top_ns if int(value) > 0}))
        self.bucket_count = int(bucket_count)
        self.regime_run_dir = Path(regime_run_dir)
        self.factor_names = tuple(factor_names)
        self.condition_names = tuple(condition_names)
        if not self.top_ns:
            raise ValueError("top_ns 不能为空")
        if self.bucket_count < 2:
            raise ValueError("bucket_count must be >= 2")

    def required_fields(self, request: ResearchRequest) -> tuple[str, ...]:
        return (
            *self.factor_names,
            *(f"condition_{name}" for name in self.condition_names),
            *(f"fwd_ret_{horizon}d" for horizon in request.study.horizons),
        )

    def data_fields(self, request: ResearchRequest) -> tuple[str, ...]:
        return ()

    def compute(self, panel: pd.DataFrame, request: ResearchRequest):
        require_columns(panel, self.required_fields(request), context="weekly metrics")
        keys = ["trade_date"]
        if "weekday" in panel:
            keys.insert(0, "weekday")
        daily_top_rows: list[dict] = []
        bucket_rows: list[dict] = []
        ic_rows: list[dict] = []
        overlap_rows: list[dict] = []
        benchmark = self._benchmark_sets()

        for condition in self.condition_names:
            condition_col = f"condition_{condition}"
            selected = panel[panel[condition_col].fillna(False)].copy()
            for factor in self.factor_names:
                ranked = self._rank_factor_panel(selected, keys, factor)
                if ranked.empty:
                    continue

                for horizon in request.study.horizons:
                    label = f"fwd_ret_{horizon}d"
                    valid = self._valid_horizon_frame(ranked, keys, factor, label)
                    ic_rows.extend(_rank_ic_rows(valid, keys, factor=factor, condition=condition, horizon=horizon))
                    daily_top_rows.extend(_topn_rows(valid, keys, factor=factor, condition=condition,
                                                     horizon=horizon, label=label, top_ns=self.top_ns))
                    bucket_rows.extend(_bucket_rows(valid, keys, factor=factor, condition=condition,
                                                    horizon=horizon, label=label, bucket_count=self.bucket_count))

                overlap_rows.extend(
                    self._overlap(ranked, keys, factor=factor, condition=condition, benchmark=benchmark)
                )

        daily_top = pd.DataFrame(daily_top_rows)
        daily_bucket = pd.DataFrame(bucket_rows)
        daily_ic = pd.DataFrame(ic_rows)
        outputs = {
            "condition_coverage": MetricResult(self._condition_coverage(panel)),
            "topn_returns": MetricResult(_aggregate_topn(daily_top)),
            "bucket_returns": MetricResult(_aggregate_bucket(daily_bucket, self.bucket_count)),
            "rank_ic": MetricResult(_aggregate_ic(daily_ic)),
            "period_returns": MetricResult(_period_returns(daily_top)),
            "focus_period_returns": MetricResult(_focus_returns(daily_top)),
            "strategy_overlap": MetricResult(pd.DataFrame(overlap_rows)),
        }
        if "weekday" in panel:
            outputs["weekday_comparison"] = MetricResult(_weekday_comparison(daily_top))
        outputs = {name: result for name, result in outputs.items() if not result.frame.empty}
        summary = {
            "sample_rows": len(panel),
            "trade_date_count": panel["trade_date"].nunique() if not panel.empty else 0,
            "code_count": panel["code"].nunique() if not panel.empty else 0,
            "horizons": list(request.study.horizons),
            "weekly_factor_count": len(self.factor_names),
            "weekly_condition_count": len(self.condition_names),
        }
        return outputs, summary

    @staticmethod
    def _rank_factor_panel(frame: pd.DataFrame, keys: list[str], factor: str) -> pd.DataFrame:
        ranked = frame.copy()
        ranked[factor] = pd.to_numeric(ranked[factor], errors="coerce")
        ranked = ranked.dropna(subset=[factor]).sort_values(
            [*keys, factor, "code"],
            ascending=[*[True] * len(keys), False, True],
            kind="mergesort",
        ).copy()
        if ranked.empty:
            return ranked
        grouped = ranked.groupby(keys, sort=False, observed=True)
        ranked["rank_desc"] = grouped.cumcount() + 1
        ranked["sample_count"] = grouped[factor].transform("size")
        return ranked

    @staticmethod
    def _valid_horizon_frame(ranked: pd.DataFrame, keys: list[str], factor: str, label: str) -> pd.DataFrame:
        if ranked.empty:
            return ranked
        valid = ranked.copy()
        valid[label] = pd.to_numeric(valid[label], errors="coerce")
        valid = valid.dropna(subset=[label]).copy()
        if valid.empty:
            return valid
        valid["_score_rank"] = valid.groupby(keys, observed=True)[factor].rank(method="average")
        valid["_label_rank"] = valid.groupby(keys, observed=True)[label].rank(method="average")
        return valid

    def _condition_coverage(self, panel: pd.DataFrame) -> pd.DataFrame:
        rows = []
        denominator = len(panel)
        for condition in self.condition_names:
            selected = panel[panel[f"condition_{condition}"].fillna(False)]
            rows.append({
                "condition_name": condition,
                "date_count": selected["trade_date"].nunique(),
                "candidate_count": len(selected),
                "avg_candidates_per_date": (
                    selected.groupby("trade_date", observed=True).size().mean()
                    if not selected.empty else 0.0
                ),
                "coverage_vs_none": len(selected) / denominator if denominator else np.nan,
            })
        return pd.DataFrame(rows)

    def _benchmark_sets(self) -> dict[int, dict[pd.Timestamp, set[str]]]:
        path = self.regime_run_dir / "weekly_fill_signal_table.csv"
        if not path.exists():
            return {}
        frame = pd.read_csv(path)
        if frame.empty or not {"trade_date", "code"}.issubset(frame):
            return {}
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"].astype(str).str.slice(0, 10),
            errors="coerce",
            format="%Y-%m-%d",
        ).dt.normalize()
        frame = frame.dropna(subset=["trade_date", "code"]).copy()
        if frame.empty:
            return {}
        if "fill_rank" not in frame:
            frame = frame.sort_values(["trade_date", "code"], kind="mergesort")
            frame["fill_rank"] = frame.groupby("trade_date", observed=True).cumcount() + 1
        frame["fill_rank"] = pd.to_numeric(frame["fill_rank"], errors="coerce")
        frame = frame.dropna(subset=["fill_rank"]).copy()
        if frame.empty:
            return {}
        return {
            top_n: {
                pd.Timestamp(date): set(section["code"].astype(str))
                for date, section in frame[frame["fill_rank"] <= top_n].groupby("trade_date", observed=True)
            }
            for top_n in self.top_ns
        }

    def _overlap(self, ranked, keys, *, factor, condition, benchmark):
        if not benchmark:
            return []
        rows = []
        partitions = (
            ranked.groupby("weekday", observed=True)
            if "weekday" in keys
            else [(None, ranked)]
        )
        for weekday, partition in partitions:
            for top_n in self.top_ns:
                counts, ratios = [], []
                selected = partition[partition["rank_desc"] <= top_n]
                expected_by_date = benchmark.get(top_n, {})
                for trade_date, section in selected.groupby("trade_date", observed=True):
                    expected = expected_by_date.get(pd.Timestamp(trade_date), set())
                    if not expected:
                        continue
                    codes = set(section["code"].astype(str))
                    overlap = len(codes & expected)
                    counts.append(overlap)
                    ratios.append(overlap / max(1, min(len(codes), top_n)))
                row = {
                    "factor_name": factor,
                    "condition_name": condition,
                    "top_n": top_n,
                    "date_count": len(counts),
                    "avg_overlap_count": np.mean(counts) if counts else np.nan,
                    "avg_overlap_ratio": np.mean(ratios) if ratios else np.nan,
                }
                if weekday is not None:
                    row["weekday"] = int(weekday)
                rows.append(row)
        return rows


def _rank_ic_rows(frame, keys, *, factor, condition, horizon):
    if frame.empty:
        return []
    rows = []
    for group_key, section in frame.groupby(keys, observed=True):
        if len(section) < 8:
            continue
        row = _key_values(keys, group_key)
        row.update({
            "factor_name": factor,
            "condition_name": condition,
            "horizon": horizon,
            "rank_ic": section["_score_rank"].corr(section["_label_rank"]),
            "sample_count": len(section),
        })
        rows.append(row)
    return rows


def _topn_rows(frame, keys, *, factor, condition, horizon, label, top_ns):
    if frame.empty:
        return []
    rows = []
    for top_n in top_ns:
        top = frame[frame["rank_desc"] <= top_n]
        for group_key, section in top.groupby(keys, observed=True):
            row = _key_values(keys, group_key)
            row.update({
                "factor_name": factor,
                "condition_name": condition,
                "horizon": horizon,
                "top_n": top_n,
                "forward_return": section[label].mean(),
                "selected_count": len(section),
            })
            rows.append(row)
    return rows


def _bucket_rows(frame, keys, *, factor, condition, horizon, label, bucket_count):
    if frame.empty:
        return []
    bucketable = frame[frame["sample_count"] >= bucket_count].copy()
    if bucketable.empty:
        return []
    bucketable["bucket_id"] = (
        np.floor((bucketable["rank_desc"] - 1) * bucket_count / bucketable["sample_count"]).astype(int) + 1
    )
    daily_bucket = bucketable.groupby([*keys, "bucket_id"], observed=True)[label].mean().reset_index()
    rows = []
    for row in daily_bucket.itertuples(index=False):
        values = {name: getattr(row, name) for name in keys}
        values.update({
            "factor_name": factor,
            "condition_name": condition,
            "horizon": horizon,
            "bucket_id": int(row.bucket_id),
            "forward_return": getattr(row, label),
        })
        rows.append(values)
    return rows


def _key_values(keys, value):
    values = value if isinstance(value, tuple) else (value,)
    return dict(zip(keys, values))


def _aggregate_topn(frame):
    if frame.empty:
        return frame
    groups = [name for name in ("weekday", "factor_name", "condition_name", "horizon", "top_n") if name in frame]
    result = frame.assign(
        _weighted_return=frame["forward_return"] * frame["selected_count"]
    ).groupby(groups, observed=True).agg(
        date_count=("trade_date", "nunique"),
        selected_count=("selected_count", "sum"),
        avg_selected_per_date=("selected_count", "mean"),
        date_weighted_mean=("forward_return", "mean"),
        date_weighted_median=("forward_return", "median"),
        date_weighted_win_rate=("forward_return", lambda value: (value > 0).mean()),
        _weighted_return=("_weighted_return", "sum"),
    ).reset_index()
    result["trade_weighted_mean"] = result["_weighted_return"] / result["selected_count"].replace(0, np.nan)
    return result.drop(columns="_weighted_return")


def _aggregate_bucket(frame, bucket_count):
    if frame.empty:
        return frame
    groups = [name for name in ("weekday", "factor_name", "condition_name", "horizon", "bucket_id") if name in frame]
    result = frame.groupby(groups, observed=True)["forward_return"].agg(["mean", "median", "count"]).reset_index()
    result = result.rename(columns={"mean": "date_weighted_mean", "median": "date_weighted_median", "count": "date_count"})
    spread_keys = [name for name in groups if name != "bucket_id"]
    top = result[result["bucket_id"] == 1][[*spread_keys, "date_weighted_mean"]].rename(
        columns={"date_weighted_mean": "_top"}
    )
    bottom = result[result["bucket_id"] == bucket_count][[*spread_keys, "date_weighted_mean"]].rename(
        columns={"date_weighted_mean": "_bottom"}
    )
    result = result.merge(top, on=spread_keys, how="left").merge(bottom, on=spread_keys, how="left")
    result["top_minus_bottom"] = result["_top"] - result["_bottom"]
    result["bucket_count"] = bucket_count
    return result.drop(columns=["_top", "_bottom"])


def _aggregate_ic(frame):
    if frame.empty:
        return frame
    groups = [name for name in ("weekday", "factor_name", "condition_name", "horizon") if name in frame]
    return frame.groupby(groups, observed=True).agg(
        rank_ic_mean=("rank_ic", "mean"),
        rank_ic_median=("rank_ic", "median"),
        rank_ic_std=("rank_ic", "std"),
        positive_ic_ratio=("rank_ic", lambda value: (value > 0).mean()),
        sample_dates=("trade_date", "nunique"),
        avg_sample_size=("sample_count", "mean"),
    ).reset_index()


def _period_returns(frame):
    if frame.empty:
        return frame
    working = frame.copy()
    working["trade_date"] = pd.to_datetime(working["trade_date"])
    pieces = []
    for kind, values in (
        ("year", working["trade_date"].dt.year.astype(str)),
        ("month", working["trade_date"].dt.strftime("%Y-%m")),
    ):
        section = working.assign(period_type=kind, period=values)
        groups = [name for name in ("weekday", "period_type", "period", "factor_name", "condition_name", "horizon", "top_n") if name in section]
        pieces.append(section.groupby(groups, observed=True)["forward_return"].agg(["mean", "median", "count"]).reset_index())
    return pd.concat(pieces, ignore_index=True).rename(
        columns={"mean": "date_weighted_mean", "median": "date_weighted_median", "count": "date_count"}
    )


def _focus_returns(frame):
    if frame.empty:
        return frame
    pieces = []
    dates = pd.to_datetime(frame["trade_date"])
    for name, start, end in FOCUS_PERIODS:
        section = frame[dates.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
        if section.empty:
            continue
        section["period_name"], section["period_start"], section["period_end"] = name, start, end
        groups = [column for column in (
            "weekday", "period_name", "period_start", "period_end",
            "factor_name", "condition_name", "horizon", "top_n",
        ) if column in section]
        pieces.append(section.groupby(groups, observed=True)["forward_return"].agg(["mean", "median", "count"]).reset_index())
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True).rename(
        columns={"mean": "date_weighted_mean", "median": "date_weighted_median", "count": "date_count"}
    )


def _weekday_comparison(frame):
    if frame.empty or "weekday" not in frame:
        return pd.DataFrame()
    groups = ["weekday", "factor_name", "condition_name", "horizon", "top_n"]
    return frame.groupby(groups, observed=True)["forward_return"].agg(["mean", "median", "count"]).reset_index().rename(
        columns={"mean": "date_weighted_mean", "median": "date_weighted_median", "count": "date_count"}
    )
