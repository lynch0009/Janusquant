"""执行引擎内部共享类型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class PendingRiskExit:
    """收盘确认后，等待下一交易日执行的风险退出任务。"""

    code: str
    signal_date: datetime
    scheduled_trade_date: datetime
    reason: str
    score: float | None = None
    metadata: dict[str, Any] | None = None
    risk_rule_name: str | None = None
