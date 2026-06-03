"""A 股涨跌停价格、涨停统计与日线可成交性判断工具。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import pandas as pd


LIMIT_STATE_NORMAL = "normal"
LIMIT_STATE_UP_TOUCHED = "up_limit_touched"
LIMIT_STATE_UP_LOCKED = "up_limit_locked"
LIMIT_STATE_DOWN_TOUCHED = "down_limit_touched"
LIMIT_STATE_DOWN_LOCKED = "down_limit_locked"


def strict_round(value: float, ndigits: int = 2) -> float:
    """使用四舍五入到指定小数位，避免 Python 默认 round 的银行家舍入。"""

    quantize_pattern = "0." + ("0" * ndigits) if ndigits > 0 else "0"
    return float(Decimal(str(value)).quantize(Decimal(quantize_pattern), rounding=ROUND_HALF_UP))


def extract_numeric_code(code: Any) -> str:
    """提取不带交易所前缀的纯数字证券代码。"""

    text = str(code or "").strip()
    if "." in text:
        return text.split(".", 1)[1]
    return text


def get_a_share_limit_ratio(code: Any, *, is_st: bool = False) -> float:
    """根据证券代码和 ST 状态返回 A 股涨跌停比例。"""

    numeric_code = extract_numeric_code(code)
    if is_st:
        return 0.05
    if numeric_code.startswith(("4", "8")):
        return 0.30
    if numeric_code.startswith(("30", "68")):
        return 0.20
    return 0.10


def calculate_a_share_limit_prices(
    code: Any,
    preclose: Any,
    *,
    is_st: bool = False,
) -> tuple[float | None, float | None]:
    """计算 A 股当日涨停价和跌停价。"""

    if preclose is None or pd.isna(preclose):
        return None, None
    preclose_value = float(preclose)
    if preclose_value <= 0:
        return None, None

    limit_ratio = get_a_share_limit_ratio(code, is_st=is_st)
    up_limit = strict_round(preclose_value * (1 + limit_ratio), 2)
    down_limit = strict_round(preclose_value * (1 - limit_ratio), 2)
    return up_limit, down_limit


def calculate_limit_up_price(code: Any, preclose: Any, *, is_st: bool = False) -> float | None:
    """计算当日涨停价。"""

    up_limit, _ = calculate_a_share_limit_prices(code, preclose, is_st=is_st)
    return up_limit


def is_limit_up_close(
    code: Any,
    preclose: Any,
    close: Any,
    *,
    is_st: bool = False,
    atol: float = 1e-6,
) -> bool:
    """判断收盘价是否封在涨停价。"""

    if close is None or pd.isna(close):
        return False
    up_limit, _ = calculate_a_share_limit_prices(code, preclose, is_st=is_st)
    if up_limit is None:
        return False
    return abs(float(close) - up_limit) <= atol


def consecutive_true_counts(values: pd.Series) -> pd.Series:
    """计算布尔序列中连续 True 的长度。"""

    counts: list[int] = []
    streak = 0
    for value in values.fillna(False).astype(bool).tolist():
        if value:
            streak += 1
        else:
            streak = 0
        counts.append(streak)
    return pd.Series(counts, index=values.index, dtype="int64")


def add_limit_up_features(
    frame: pd.DataFrame,
    *,
    code_col: str = "code",
    date_col: str = "trade_date",
    preclose_col: str = "preclose",
    close_col: str = "close",
    is_st_col: str = "isST",
) -> pd.DataFrame:
    """给日线数据补充涨停价、是否涨停、连板数等字段。"""

    if frame.empty:
        return frame.copy()

    result = frame.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.sort_values([code_col, date_col]).reset_index(drop=True)

    result["limit_up_price"] = result.apply(
        lambda row: calculate_limit_up_price(
            row[code_col],
            row.get(preclose_col),
            is_st=bool(row.get(is_st_col, False)),
        ),
        axis=1,
    )
    result["is_limit_up"] = result.apply(
        lambda row: is_limit_up_close(
            row[code_col],
            row.get(preclose_col),
            row.get(close_col),
            is_st=bool(row.get(is_st_col, False)),
        ),
        axis=1,
    )
    result["limit_up_streak"] = (
        result.groupby(code_col, group_keys=False)["is_limit_up"].apply(consecutive_true_counts).astype(int)
    )
    return result


@dataclass(frozen=True)
class DailyLimitStatus:
    """描述单根日线上的涨跌停触板与封板状态。"""

    up_limit: float | None
    down_limit: float | None
    limit_state: str
    up_limit_touched: bool
    down_limit_touched: bool


@dataclass(frozen=True)
class DailyLimitFillDecision:
    """描述日线级别下某个方向是否可成交，以及建议成交价。"""

    fillable: bool
    execution_price: float | None
    reject_reason: str | None
    limit_state: str
    up_limit: float | None
    down_limit: float | None


def get_daily_limit_status(
    code: Any,
    *,
    preclose: Any,
    open_price: Any,
    high_price: Any,
    low_price: Any,
    close_price: Any,
    is_st: bool = False,
) -> DailyLimitStatus:
    """基于单根日线，判断当日涨跌停触板与封板状态。"""

    up_limit, down_limit = calculate_a_share_limit_prices(code, preclose, is_st=is_st)
    if up_limit is None or down_limit is None:
        return DailyLimitStatus(
            up_limit=None,
            down_limit=None,
            limit_state=LIMIT_STATE_NORMAL,
            up_limit_touched=False,
            down_limit_touched=False,
        )

    open_value = float(open_price)
    high_value = float(high_price)
    low_value = float(low_price)
    close_value = float(close_price)

    up_limit_touched = high_value >= up_limit
    down_limit_touched = low_value <= down_limit

    if up_limit_touched and open_value >= up_limit and low_value >= up_limit and close_value >= up_limit:
        limit_state = LIMIT_STATE_UP_LOCKED
    elif down_limit_touched and open_value <= down_limit and high_value <= down_limit and close_value <= down_limit:
        limit_state = LIMIT_STATE_DOWN_LOCKED
    elif up_limit_touched:
        limit_state = LIMIT_STATE_UP_TOUCHED
    elif down_limit_touched:
        limit_state = LIMIT_STATE_DOWN_TOUCHED
    else:
        limit_state = LIMIT_STATE_NORMAL

    return DailyLimitStatus(
        up_limit=up_limit,
        down_limit=down_limit,
        limit_state=limit_state,
        up_limit_touched=up_limit_touched,
        down_limit_touched=down_limit_touched,
    )


def decide_daily_buy_fill(
    code: Any,
    *,
    preclose: Any,
    open_price: Any,
    high_price: Any,
    low_price: Any,
    close_price: Any,
    fallback_price: float,
    is_st: bool = False,
) -> DailyLimitFillDecision:
    """判断日线买入是否可成交。

    规则：
    1. 全天封死涨停则买不到。
    2. 盘中触及涨停但打开过，按涨停价成交。
    3. 其余情况按 fallback_price 成交。
    """

    status = get_daily_limit_status(
        code,
        preclose=preclose,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        is_st=is_st,
    )
    if status.limit_state == LIMIT_STATE_UP_LOCKED:
        return DailyLimitFillDecision(
            fillable=False,
            execution_price=None,
            reject_reason="limit_up_locked",
            limit_state=status.limit_state,
            up_limit=status.up_limit,
            down_limit=status.down_limit,
        )
    if status.limit_state == LIMIT_STATE_UP_TOUCHED:
        return DailyLimitFillDecision(
            fillable=True,
            execution_price=status.up_limit,
            reject_reason=None,
            limit_state=status.limit_state,
            up_limit=status.up_limit,
            down_limit=status.down_limit,
        )
    return DailyLimitFillDecision(
        fillable=True,
        execution_price=fallback_price,
        reject_reason=None,
        limit_state=status.limit_state,
        up_limit=status.up_limit,
        down_limit=status.down_limit,
    )


def decide_daily_sell_fill(
    code: Any,
    *,
    preclose: Any,
    open_price: Any,
    high_price: Any,
    low_price: Any,
    close_price: Any,
    fallback_price: float,
    is_st: bool = False,
) -> DailyLimitFillDecision:
    """判断日线卖出是否可成交。

    规则：
    1. 全天封死跌停则卖不掉。
    2. 盘中触及跌停但打开过，按跌停价成交。
    3. 其余情况按 fallback_price 成交。
    """

    status = get_daily_limit_status(
        code,
        preclose=preclose,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        is_st=is_st,
    )
    if status.limit_state == LIMIT_STATE_DOWN_LOCKED:
        return DailyLimitFillDecision(
            fillable=False,
            execution_price=None,
            reject_reason="limit_down_locked",
            limit_state=status.limit_state,
            up_limit=status.up_limit,
            down_limit=status.down_limit,
        )
    if status.limit_state == LIMIT_STATE_DOWN_TOUCHED:
        return DailyLimitFillDecision(
            fillable=True,
            execution_price=status.down_limit,
            reject_reason=None,
            limit_state=status.limit_state,
            up_limit=status.up_limit,
            down_limit=status.down_limit,
        )
    return DailyLimitFillDecision(
        fillable=True,
        execution_price=fallback_price,
        reject_reason=None,
        limit_state=status.limit_state,
        up_limit=status.up_limit,
        down_limit=status.down_limit,
    )


__all__ = [
    "DailyLimitFillDecision",
    "DailyLimitStatus",
    "LIMIT_STATE_DOWN_LOCKED",
    "LIMIT_STATE_DOWN_TOUCHED",
    "LIMIT_STATE_NORMAL",
    "LIMIT_STATE_UP_LOCKED",
    "LIMIT_STATE_UP_TOUCHED",
    "add_limit_up_features",
    "calculate_a_share_limit_prices",
    "calculate_limit_up_price",
    "consecutive_true_counts",
    "decide_daily_buy_fill",
    "decide_daily_sell_fill",
    "extract_numeric_code",
    "get_a_share_limit_ratio",
    "get_daily_limit_status",
    "is_limit_up_close",
    "strict_round",
]
