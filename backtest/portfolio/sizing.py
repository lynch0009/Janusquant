"""仓位分配器定义。

仓位分配器决定一笔候选信号在当前资金和剩余仓位约束下能拿到多少预算。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BasePositionSizer(ABC):
    """仓位分配器抽象基类。"""

    @abstractmethod
    def allocate(
        self,
        *,
        cash: float,
        remaining_slots: int,
        max_position_pct: float,
    ) -> float:
        """根据当前组合状态返回单笔可用预算。"""

        raise NotImplementedError


class FixedFractionSizer(BasePositionSizer):
    """固定比例仓位分配器。

    适合单票有明确上限比例的策略。每一笔预算受两层约束：
    1. 不超过当前现金的一定比例
    2. 不超过平均分配到剩余槽位后的预算
    """

    def allocate(
        self,
        *,
        cash: float,
        remaining_slots: int,
        max_position_pct: float,
    ) -> float:
        if cash <= 0 or remaining_slots <= 0:
            return 0.0
        return min(cash * max_position_pct, cash / remaining_slots)


class EqualSlotSizer(BasePositionSizer):
    """按剩余槽位等权分配预算。

    这是轮动类策略更自然的仓位口径：
    当前剩余现金尽量平均分给剩余待买股票。
    """

    def allocate(
        self,
        *,
        cash: float,
        remaining_slots: int,
        max_position_pct: float,
    ) -> float:
        if cash <= 0 or remaining_slots <= 0:
            return 0.0
        return cash / remaining_slots
