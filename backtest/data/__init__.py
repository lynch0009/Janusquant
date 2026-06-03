"""数据访问层导出。

对外统一暴露数据门面，避免上层直接依赖底层 Mongo 仓储实现。
"""

from .frame_cache import CachedMongoDataPortal, FrameCache
from .portal import MongoDataPortal
from .research_store import ResearchDailyHistoryStore

__all__ = ["CachedMongoDataPortal", "FrameCache", "MongoDataPortal", "ResearchDailyHistoryStore"]
