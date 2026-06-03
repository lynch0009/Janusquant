"""DataFrame 相关的共享工具。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

import pandas as pd


def first_sorted_row(frame: pd.DataFrame | None, *, sort_field: str = "dt") -> pd.Series | None:
    """返回按指定字段排序后的第一行，优先避免整表排序。"""

    if frame is None or frame.empty:
        return None
    if sort_field not in frame.columns or len(frame) == 1:
        return frame.iloc[0]
    try:
        return frame.nsmallest(1, sort_field).iloc[0]
    except (TypeError, ValueError):
        return frame.sort_values(sort_field).iloc[0]


def sort_frame(frame: pd.DataFrame | None, *, sort_field: str = "dt") -> pd.DataFrame:
    """按指定字段排序并重建连续索引。"""

    if frame is None:
        return pd.DataFrame()
    if frame.empty or sort_field not in frame.columns:
        return frame.reset_index(drop=True).copy()
    return frame.sort_values(sort_field).reset_index(drop=True)


def records_to_frame(records: Iterable[Any]) -> pd.DataFrame:
    """把 dataclass / 带 to_dict 的对象列表统一导出成 DataFrame。"""

    rows: list[dict[str, Any]] = []
    for record in records:
        if hasattr(record, "to_dict"):
            rows.append(record.to_dict())
        elif is_dataclass(record):
            rows.append(asdict(record))
        else:
            rows.append(dict(record))
    return pd.DataFrame(rows)
