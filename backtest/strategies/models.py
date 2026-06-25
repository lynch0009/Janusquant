"""策略层数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DailyCandidate:
    """描述某个交易日生成的一只候选股票。"""

    signal_date: datetime
    code: str
    score: float | None = None
    hold_days: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
