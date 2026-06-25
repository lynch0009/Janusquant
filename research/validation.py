"""Central validation for research configuration and DataFrame contracts."""

from __future__ import annotations

from collections.abc import Iterable
import pandas as pd

from .errors import ResearchDataContractError


def require_columns(frame: pd.DataFrame, columns: Iterable[str], *, context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ResearchDataContractError(f"{context} missing required columns: {missing}")


def validate_unique_keys(
    frame: pd.DataFrame,
    keys: tuple[str, ...] = ("code", "trade_date"),
    *,
    context: str,
) -> None:
    require_columns(frame, keys, context=context)
    duplicates = frame.duplicated(list(keys), keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, list(keys)].head(5).to_dict("records")
        raise ResearchDataContractError(f"{context} contains duplicate keys {keys}: {examples}")


def normalize_trade_dates(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    require_columns(frame, ("trade_date",), context=context)
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
    if result["trade_date"].isna().any():
        raise ResearchDataContractError(f"{context} contains invalid trade_date values")
    return result


def validate_panel_date_range(frame: pd.DataFrame, *, start_date, end_date) -> None:
    if frame.empty:
        return
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    values = pd.to_datetime(frame["trade_date"]).dt.normalize()
    if (values < start).any() or (values > end).any():
        actual_min = values.min().strftime("%Y-%m-%d")
        actual_max = values.max().strftime("%Y-%m-%d")
        raise ResearchDataContractError(
            f"analysis panel date range [{actual_min}, {actual_max}] exceeds "
            f"configured range [{start:%Y-%m-%d}, {end:%Y-%m-%d}]"
        )
