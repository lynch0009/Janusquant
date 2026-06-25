"""Small-cap dataset construction and small-cap-only panel components."""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from hashlib import sha1
from typing import Iterable

import numpy as np
import pandas as pd

from research.cache import ResearchFrameCache
from research.errors import ResearchDataContractError
from research.models import DataRequirements, ResearchDataset, ResearchRequest
from research.validation import require_columns

from .config import SmallCapDatasetConfig


class SmallCapResearchDatasetBuilder:
    """Build the daily small-cap universe and its buffered raw history."""

    cache_identity = "smallcap_dataset_v3"

    def __init__(self, data_portal, cache: ResearchFrameCache):
        self.data_portal = data_portal
        self.cache = cache

    @staticmethod
    def stable_config(dataset_config: object) -> object:
        if not isinstance(dataset_config, SmallCapDatasetConfig):
            raise TypeError("SmallCapResearchDatasetBuilder requires SmallCapDatasetConfig")
        return asdict(dataset_config)

    def build(self, request: ResearchRequest, requirements: DataRequirements) -> ResearchDataset:
        config = request.dataset
        if not isinstance(config, SmallCapDatasetConfig):
            raise TypeError("SmallCapResearchDatasetBuilder requires SmallCapDatasetConfig")
        config.validate()
        universe = self._build_smallcap_universe(request, config)
        history = self._load_daily_history(
            request,
            config,
            universe["code"].unique() if not universe.empty else (),
            requirements,
        )
        return ResearchDataset(
            universe=universe,
            history=history,
            metadata={"builder": self.cache_identity, **asdict(config)},
        )

    def _build_smallcap_universe(
        self,
        request: ResearchRequest,
        config: SmallCapDatasetConfig,
    ) -> pd.DataFrame:
        payload = {
            "start_date": request.study.start_date.strftime("%Y-%m-%d"),
            "end_date": request.study.end_date.strftime("%Y-%m-%d"),
            "candidate_pool_size": config.candidate_pool_size,
        }

        def builder() -> pd.DataFrame:
            history = self.data_portal.get_feature_history(
                request.study.start_date,
                request.study.end_date,
                fields=["code", "date", "liqaMV"],
            )
            if history.empty:
                return pd.DataFrame(columns=["code", "trade_date", "liqaMV", "cap_rank"])
            frame = history.rename(columns={"date": "trade_date"}).copy()
            require_columns(frame, ("code", "trade_date", "liqaMV"), context="small-cap universe")
            frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
            frame["liqaMV"] = pd.to_numeric(frame["liqaMV"], errors="coerce")
            frame = frame.dropna(subset=["code", "trade_date", "liqaMV"])
            frame = frame[frame["liqaMV"] > 0].sort_values(
                ["trade_date", "liqaMV", "code"], kind="mergesort"
            )
            frame["cap_rank"] = frame.groupby("trade_date", sort=False, observed=True).cumcount() + 1
            return frame.loc[
                frame["cap_rank"] <= config.candidate_pool_size,
                ["code", "trade_date", "liqaMV", "cap_rank"],
            ].reset_index(drop=True)

        return self.cache.load_or_build("smallcap_universe", payload, builder)

    def _load_daily_history(
        self,
        request: ResearchRequest,
        config: SmallCapDatasetConfig,
        codes: Iterable[str],
        requirements: DataRequirements,
    ) -> pd.DataFrame:
        normalized_codes = sorted({str(code) for code in codes if pd.notna(code)})
        if not normalized_codes:
            return pd.DataFrame(columns=["code", "trade_date", *requirements.fields])
        buffer_start = request.study.start_date - timedelta(
            days=max(420, (requirements.warmup_window + config.min_listing_trade_days) * 3)
        )
        buffer_end = request.study.end_date + timedelta(
            days=max(90, requirements.future_window * 3 + 20)
        )
        # These columns are produced by this builder rather than fetched from daily history.
        local_fields = {"listing_trade_days", "liqaMV", "cap_rank", "cap_bucket"}
        requested_fields = tuple(
            dict.fromkeys(("code", "trade_date", *(field for field in requirements.fields if field not in local_fields)))
        )
        payload = {
            "buffer_start": buffer_start.strftime("%Y-%m-%d"),
            "buffer_end": buffer_end.strftime("%Y-%m-%d"),
            "codes_signature": _codes_signature(normalized_codes),
            "requested_fields": requested_fields,
            "price_mode": config.price_mode,
            "factor_version": requirements.factor_version,
            "label_version": requirements.label_version,
        }

        def builder() -> pd.DataFrame:
            frame = self.data_portal.get_daily_history(
                buffer_start,
                buffer_end,
                codes=normalized_codes,
                fields=list(requested_fields),
                include_stopped=False,
                batch_size=1000,
                price_mode=config.price_mode,
            )
            if frame.empty:
                return frame
            require_columns(frame, requested_fields, context="small-cap daily history")
            frame = frame.copy()
            frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
            if frame["trade_date"].isna().any():
                raise ResearchDataContractError("small-cap daily history contains invalid trade_date")
            frame = frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)
            frame["listing_trade_days"] = frame.groupby("code", sort=False, observed=True).cumcount() + 1
            return frame

        return self.cache.load_or_build("daily_history", payload, builder)


class SmallCapEligibilitySelector:
    """Apply listing-age and ST eligibility after panel assembly."""

    def required_fields(self, request: ResearchRequest) -> tuple[str, ...]:
        config = _config(request)
        return ("listing_trade_days", "isST") if config.exclude_st else ("listing_trade_days",)

    def stable_config(self) -> dict:
        return {}

    def select(self, panel: pd.DataFrame, request: ResearchRequest) -> pd.DataFrame:
        config = _config(request)
        require_columns(panel, self.required_fields(request), context="small-cap eligibility")
        mask = pd.to_numeric(panel["listing_trade_days"], errors="coerce") >= config.min_listing_trade_days
        if config.exclude_st:
            is_st = panel["isST"]
            if is_st.dtype != bool:
                is_st = is_st.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
            mask &= ~is_st.fillna(False)
        return panel.loc[mask.fillna(False)].copy()


class CapBucketTransformer:
    """Add a daily market-cap bucket from cap_rank."""

    def __init__(self, bucket_count: int = 5):
        if int(bucket_count) < 2:
            raise ValueError("bucket_count must be >= 2")
        self.bucket_count = int(bucket_count)
        self.produced_fields = ("cap_bucket",)

    def required_fields(self, request: ResearchRequest) -> tuple[str, ...]:
        return ("cap_rank",)

    def stable_config(self) -> dict:
        return {"bucket_count": self.bucket_count}

    def transform(self, panel: pd.DataFrame, request: ResearchRequest) -> pd.DataFrame:
        require_columns(panel, ("cap_rank",), context="cap bucket transformer")
        result = panel.copy()
        rank = pd.to_numeric(result["cap_rank"], errors="coerce")
        count = rank.groupby(result["trade_date"], sort=False, observed=True).transform("count")
        bucket = np.floor((rank - 1) * self.bucket_count / count) + 1
        result["cap_bucket"] = pd.array(
            pd.Series(bucket, index=result.index).where(count >= self.bucket_count), dtype="Int16"
        )
        return result


def _config(request: ResearchRequest) -> SmallCapDatasetConfig:
    if not isinstance(request.dataset, SmallCapDatasetConfig):
        raise TypeError("small-cap component requires SmallCapDatasetConfig")
    return request.dataset


def _codes_signature(codes: Iterable[str]) -> str:
    joined = "|".join(sorted({str(code) for code in codes if pd.notna(code)}))
    return sha1(joined.encode("utf-8")).hexdigest()[:20] if joined else "empty"
