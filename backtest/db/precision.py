from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any

import pandas as pd


PRICE_QUANT = Decimal("0.01")
INTEGER_QUANT = Decimal("1")


def _to_decimal(value: Any) -> Decimal | None:
    if value is None or pd.isna(value):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def normalize_price(value: Any) -> float | None:
    decimal_value = _to_decimal(value)
    if decimal_value is None:
        return None
    return float(decimal_value.quantize(PRICE_QUANT, rounding=ROUND_HALF_UP))


def normalize_volume(value: Any) -> int | None:
    decimal_value = _to_decimal(value)
    if decimal_value is None:
        return None
    return int(decimal_value.quantize(INTEGER_QUANT, rounding=ROUND_HALF_UP))


def normalize_amount(value: Any) -> int | None:
    decimal_value = _to_decimal(value)
    if decimal_value is None:
        return None
    return int(decimal_value.quantize(INTEGER_QUANT, rounding=ROUND_HALF_UP))


def normalize_price_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_price, na_action="ignore").astype(float)


def normalize_volume_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_volume, na_action="ignore").astype("Int64")


def normalize_amount_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_amount, na_action="ignore").astype("Int64")
