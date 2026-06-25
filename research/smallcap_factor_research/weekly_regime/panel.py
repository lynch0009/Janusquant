"""Weekly-regime panel transformers."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from research.models import ResearchRequest
from research.validation import require_columns


class IndexRelativeStrengthTransformer:
    version = "index_relative_strength_v2"

    def __init__(self, data_portal, *, index_code: str, window: int = 20):
        self.data_portal = data_portal
        self.index_code = str(index_code)
        self.window = int(window)
        self._loaded: pd.DataFrame | None = None
        self.produced_fields = (
            "index_ret_1d",
            f"index_ret_{self.window}d",
            f"rs{self.window}_vs_index",
            f"rs_positive_days_{self.window}d",
        )

    def required_fields(self, request: ResearchRequest) -> tuple[str, ...]:
        return ("ret_1d", f"ret_{self.window}d")

    def stable_config(self) -> dict:
        return {"index_code": self.index_code, "window": self.window}

    def transform(self, panel: pd.DataFrame, request: ResearchRequest) -> pd.DataFrame:
        require_columns(panel, ("code", "trade_date", *self.required_fields(request)), context="index relative strength")
        result = panel.merge(self._index_returns(request), on="trade_date", how="left", validate="many_to_one")
        result[f"rs{self.window}_vs_index"] = (
            pd.to_numeric(result[f"ret_{self.window}d"], errors="coerce")
            - pd.to_numeric(result[f"index_ret_{self.window}d"], errors="coerce")
        )
        stock_ret = pd.to_numeric(result["ret_1d"], errors="coerce")
        index_ret = pd.to_numeric(result["index_ret_1d"], errors="coerce")
        positive = (stock_ret > index_ret).astype(float).where(stock_ret.notna() & index_ret.notna())
        ordered = result.sort_values(["code", "trade_date"], kind="mergesort").copy()
        positive = positive.reindex(ordered.index)
        ordered[f"rs_positive_days_{self.window}d"] = (
            positive.groupby(ordered["code"], sort=False, observed=True)
            .rolling(self.window, min_periods=self.window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        return ordered.sort_values(["trade_date", "code"], kind="mergesort").reset_index(drop=True)

    def _index_returns(self, request: ResearchRequest) -> pd.DataFrame:
        if self._loaded is not None:
            return self._loaded
        start = request.study.start_date - timedelta(days=max(180, self.window * 4))
        end = request.study.end_date + timedelta(days=1)
        history = self.data_portal.get_daily_history(
            start,
            end,
            codes=[self.index_code],
            fields=["code", "trade_date", "close"],
            include_stopped=False,
            batch_size=1000,
            price_mode="raw",
        )
        if history.empty:
            raise ValueError(f"未找到指数日线: {self.index_code}")
        frame = history.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        frame = frame.sort_values("trade_date", kind="mergesort")
        close = pd.to_numeric(frame["close"], errors="coerce")
        frame["index_ret_1d"] = close / close.shift(1).replace(0, np.nan) - 1.0
        frame[f"index_ret_{self.window}d"] = close / close.shift(self.window).replace(0, np.nan) - 1.0
        self._loaded = frame[["trade_date", "index_ret_1d", f"index_ret_{self.window}d"]].copy()
        return self._loaded
