"""数据访问层导出。

对外统一暴露 DuckDB 数据门面，避免上层直接依赖底层存储实现。
"""

from .duckdb_portal import CachedDuckDBDataPortal, DuckDBDataPortal
from .frame_cache import FrameCache
from .research_store import ResearchDailyHistoryStore

DataPortal = DuckDBDataPortal
CachedDataPortal = CachedDuckDBDataPortal

__all__ = [
    "CachedDataPortal",
    "CachedDuckDBDataPortal",
    "DataPortal",
    "DuckDBDataPortal",
    "FrameCache",
    "ResearchDailyHistoryStore",
]
