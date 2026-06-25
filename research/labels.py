"""Research label builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import normalize_trade_dates, require_columns, validate_unique_keys


class NextCloseForwardReturnLabelBuilder:
    """Enter at next close and exit after N additional trading rows."""

    version = "next_close_forward_return_v1"

    def required_fields(self) -> tuple[str, ...]:
        return ("code", "trade_date", "close")

    def required_future_window(self, horizons: tuple[int, ...]) -> int:
        return max(horizons, default=0) + 1

    def build(
        self,
        history: pd.DataFrame,
        horizons: tuple[int, ...],
        *,
        key_columns: tuple[str, ...] = ("code", "trade_date"),
    ) -> pd.DataFrame:
        if key_columns != ("code", "trade_date"):
            raise ValueError("NextCloseForwardReturnLabelBuilder requires code/trade_date keys")
        require_columns(history, (*key_columns, *self.required_fields()), context="label history")
        working = normalize_trade_dates(history, context="label history")
        validate_unique_keys(working, context="label history")
        working = working.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)
        close = pd.to_numeric(working["close"], errors="coerce")
        grouped = close.groupby(working["code"], sort=False, observed=True)
        entry = grouped.shift(-1).replace(0, np.nan)
        result = working[["code", "trade_date"]].copy()
        for horizon in sorted({int(value) for value in horizons if int(value) > 0}):
            result[f"fwd_ret_{horizon}d"] = grouped.shift(-(horizon + 1)) / entry - 1.0
        return result
