from __future__ import annotations

from dataclasses import dataclass

from research.errors import ResearchConfigError


@dataclass(frozen=True)
class SmallCapDatasetConfig:
    candidate_pool_size: int = 150
    price_mode: str = "hfq"
    min_listing_trade_days: int = 120
    exclude_st: bool = True

    def validate(self) -> None:
        if self.candidate_pool_size < 1:
            raise ResearchConfigError("candidate_pool_size must be >= 1")
        if self.price_mode not in {"raw", "qfq", "hfq"}:
            raise ResearchConfigError(f"unsupported price_mode: {self.price_mode}")
        if self.min_listing_trade_days < 0:
            raise ResearchConfigError("min_listing_trade_days must be >= 0")


__all__ = ["SmallCapDatasetConfig"]
