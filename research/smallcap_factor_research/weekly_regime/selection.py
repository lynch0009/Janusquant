"""Regime-state loading and weekly date selection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest.utils.config_loader import parse_bool
from research.models import ResearchRequest
from research.validation import require_columns

from .config import validate_weekday


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.map(lambda value: parse_bool(value, default=False)).astype(bool)


def load_regime_state(regime_run_dir: Path) -> pd.DataFrame:
    path = regime_run_dir / "regime_state.csv"
    if not path.exists():
        raise FileNotFoundError(f"regime_state.csv not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"regime_state.csv is empty: {path}")
    required = {"trade_date", "regime_active", "event_trigger"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"regime_state.csv missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["regime_active"] = to_bool(frame["regime_active"])
    frame["event_trigger"] = to_bool(frame["event_trigger"])
    return frame.sort_values("trade_date").reset_index(drop=True)


def select_weekly_research_dates(
    regime_state: pd.DataFrame,
    *,
    start_date: datetime,
    end_date: datetime,
    weekly_fill_weekday: int,
) -> pd.DatetimeIndex:
    state = regime_state.copy()
    state = state[
        (state["trade_date"] >= pd.Timestamp(start_date))
        & (state["trade_date"] <= pd.Timestamp(end_date))
    ].copy()
    if state.empty:
        return pd.DatetimeIndex([])
    state["week_key"] = state["trade_date"].dt.to_period("W-SUN")
    target_weekday = validate_weekday(weekly_fill_weekday)
    selected_dates = []
    for _, week_frame in state.groupby("week_key", sort=True, observed=True):
        week_frame = week_frame.sort_values("trade_date").copy()
        after_target = week_frame[week_frame["trade_date"].dt.weekday >= target_weekday].copy()
        selected_row = after_target.iloc[0] if not after_target.empty else week_frame.iloc[-1]
        if bool(selected_row["regime_active"]) and not bool(selected_row["event_trigger"]):
            selected_dates.append(pd.Timestamp(selected_row["trade_date"]).normalize())
    return pd.DatetimeIndex(sorted(set(selected_dates)))


class WeeklyRegimeSelector:
    """Select one eligible regime date for each requested weekday and week."""

    version = "weekly_regime_selector_v2"

    def __init__(
        self,
        regime_run_dir: Path,
        *,
        weekdays: tuple[int, ...],
        min_close_price: float = 0.0,
    ):
        self.regime_run_dir = Path(regime_run_dir)
        self.weekdays = tuple(dict.fromkeys(validate_weekday(value) for value in weekdays))
        if not self.weekdays:
            raise ValueError("weekdays 不能为空")
        self.min_close_price = float(min_close_price)
        self._state: pd.DataFrame | None = None

    def required_fields(self, request: ResearchRequest) -> tuple[str, ...]:
        return ("close",)

    def stable_config(self) -> dict:
        return {
            "regime_run_dir": str(self.regime_run_dir.resolve()),
            "weekdays": self.weekdays,
            "min_close_price": self.min_close_price,
        }

    def select(self, panel: pd.DataFrame, request: ResearchRequest) -> pd.DataFrame:
        require_columns(panel, ("trade_date", "close"), context="weekly regime selector")
        if self._state is None:
            self._state = load_regime_state(self.regime_run_dir)
        date_to_weekdays: dict[pd.Timestamp, list[int]] = {}
        for weekday in self.weekdays:
            dates = select_weekly_research_dates(
                self._state,
                start_date=request.study.start_date,
                end_date=request.study.end_date,
                weekly_fill_weekday=weekday,
            )
            for date in dates:
                date_to_weekdays.setdefault(pd.Timestamp(date), []).append(weekday)
        selected = panel[pd.to_datetime(panel["trade_date"]).isin(date_to_weekdays)].copy()
        selected = selected[
            pd.to_numeric(selected["close"], errors="coerce") > self.min_close_price
        ].copy()
        if len(self.weekdays) == 1:
            selected["weekday"] = self.weekdays[0]
            return selected
        selected["_weekdays"] = selected["trade_date"].map(date_to_weekdays)
        selected = selected.explode("_weekdays", ignore_index=True)
        selected["weekday"] = pd.to_numeric(selected.pop("_weekdays"), errors="raise").astype("int8")
        return selected.sort_values(["weekday", "trade_date", "code"], kind="mergesort").reset_index(drop=True)


__all__ = ["WeeklyRegimeSelector", "load_regime_state", "select_weekly_research_dates", "to_bool"]
