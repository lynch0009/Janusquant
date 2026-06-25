"""Weekly regime workflow built entirely on the generic ResearchRunner."""

from __future__ import annotations

from dataclasses import replace

from research.cache import ResearchFrameCache
from research.config import ResearchSpec
from research.models import ResearchRequest, StudyResult
from research.runner import ResearchRunner
from research.smallcap_factor_research.config import SmallCapDatasetConfig
from research.smallcap_factor_research.dataset import (
    CapBucketTransformer,
    SmallCapEligibilitySelector,
    SmallCapResearchDatasetBuilder,
)

from .config import RegimeWeeklyFillResearchConfig, features_for_feature_set, validate_config
from .metrics import WeeklyMetricSuite
from .panel import IndexRelativeStrengthTransformer
from .scores import ConditionFlagTransformer, WeeklyScoreTransformer
from .selection import WeeklyRegimeSelector


def build_weekly_request(
    data_portal,
    config: RegimeWeeklyFillResearchConfig,
    *,
    weekdays: tuple[int, ...],
) -> ResearchRequest:
    validate_config(config)
    features = features_for_feature_set(config.feature_set)
    study = ResearchSpec(
        start_date=config.start_date,
        end_date=config.end_date,
        horizons=tuple(config.horizons),
        features=features,
        group_count=config.bucket_count,
        job_name="weekly_regime",
        job_index=1,
    )
    dataset = SmallCapDatasetConfig(
        candidate_pool_size=config.candidate_pool_size,
        price_mode=config.price_mode,
        min_listing_trade_days=config.min_listing_trade_days,
        exclude_st=True,
    )
    return ResearchRequest(
        study=study,
        dataset=dataset,
        output_dir=config.output_dir,
        transformers=(
            CapBucketTransformer(config.bucket_count),
            IndexRelativeStrengthTransformer(data_portal, index_code=config.index_code),
            WeeklyScoreTransformer(),
            ConditionFlagTransformer(),
        ),
        selectors=(
            SmallCapEligibilitySelector(),
            WeeklyRegimeSelector(
                config.regime_run_dir,
                weekdays=weekdays,
                min_close_price=config.min_close_price,
            ),
        ),
        metric_suite=WeeklyMetricSuite(
            top_ns=config.top_ns,
            bucket_count=config.bucket_count,
            regime_run_dir=config.regime_run_dir,
        ),
        export_panel=True,
    )


def run_regime_weekly_fill_research(
    data_portal,
    config: RegimeWeeklyFillResearchConfig,
    *,
    runner: ResearchRunner | None = None,
) -> StudyResult:
    request = build_weekly_request(
        data_portal,
        config,
        weekdays=(config.weekly_fill_weekday,),
    )
    cache = ResearchFrameCache(config.cache_dir)
    builder = SmallCapResearchDatasetBuilder(data_portal, cache)
    return (runner or ResearchRunner()).run(request, dataset_builder=builder)


def run_regime_weekly_fill_weekday_batch(
    data_portal,
    config: RegimeWeeklyFillResearchConfig,
    *,
    weekdays: tuple[int, ...],
    runner: ResearchRunner | None = None,
) -> StudyResult:
    """Run all weekdays from one dataset/factor/label/transformer pipeline."""

    request = build_weekly_request(data_portal, config, weekdays=weekdays)
    request = replace(
        request,
        study=replace(request.study, job_name="weekly_regime_weekday_batch"),
    )
    cache = ResearchFrameCache(config.cache_dir)
    builder = SmallCapResearchDatasetBuilder(data_portal, cache)
    return (runner or ResearchRunner()).run(request, dataset_builder=builder)


__all__ = [
    "build_weekly_request",
    "run_regime_weekly_fill_research",
    "run_regime_weekly_fill_weekday_batch",
]
