from __future__ import annotations

from pathlib import Path

from backtest.utils.dataframe_cache import DataFrameCache


class ResearchFrameCache(DataFrameCache):
    """Research-compatible name for the shared DataFrame cache."""

    def __init__(self, root: str | Path | None, *, version: str = "research-v2"):
        super().__init__(Path(root) if root else None, version=version)
