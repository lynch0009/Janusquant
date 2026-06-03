"""日期时间相关的共享工具。"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Iterable


def to_pydatetime(value: Any) -> Any:
    """把 pandas / numpy 风格时间对象统一转换成原生 datetime。"""

    if value is None:
        return None
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value


def to_pydatetime_set(values: Iterable[Any]) -> set[Any]:
    """批量转换为可直接做 dict/set key 的原生时间对象。"""

    return {to_pydatetime(value) for value in values}


def combine_trade_date(trade_date: Any, clock_time: time) -> datetime:
    """把交易日和一个时刻拼成完整 datetime。"""

    normalized = to_pydatetime(trade_date)
    if isinstance(normalized, datetime):
        return datetime.combine(normalized.date(), clock_time)
    if isinstance(normalized, date):
        return datetime.combine(normalized, clock_time)
    raise TypeError(f"unsupported trade_date type: {type(trade_date)!r}")


def to_trade_datetime(value: Any) -> datetime:
    """把日期/时间值规范化成交易日零点 datetime。"""

    normalized = to_pydatetime(value)
    if isinstance(normalized, datetime):
        return datetime(normalized.year, normalized.month, normalized.day)
    if isinstance(normalized, date):
        return datetime(normalized.year, normalized.month, normalized.day)
    text = str(normalized).strip()
    if len(text) == 8 and text.isdigit():
        parsed = datetime.strptime(text, "%Y%m%d")
    else:
        parsed = datetime.fromisoformat(text[:10])
    return datetime(parsed.year, parsed.month, parsed.day)


def date_text(value: Any) -> str:
    """把日期/时间值格式化成 YYYY-MM-DD。"""

    return to_trade_datetime(value).strftime("%Y-%m-%d")


def trade_time_for_frequency(
    value: Any,
    trade_date: Any,
    *,
    data_frequency: str,
    daily_time: time = time(15, 0),
) -> datetime:
    """按数据频率把时间字段统一成可用于成交/风控判断的 datetime。"""

    if data_frequency == "daily" or value is None:
        return combine_trade_date(trade_date, daily_time)
    normalized = to_pydatetime(value)
    if isinstance(normalized, datetime):
        return normalized
    if isinstance(normalized, date):
        return datetime.combine(normalized, daily_time)
    raise TypeError(f"unsupported time value type: {type(value)!r}")
