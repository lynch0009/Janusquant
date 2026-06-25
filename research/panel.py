"""Generic assembly of universe, factor and label frames."""

from __future__ import annotations

import pandas as pd

from .errors import ResearchDataContractError
from .models import ResearchDataset
from .validation import normalize_trade_dates, require_columns, validate_unique_keys


class PanelAssembler:
    def assemble(
        self,
        dataset: ResearchDataset,
        factors: pd.DataFrame,
        labels: pd.DataFrame,
        *,
        required_columns: tuple[str, ...],
        features: tuple[str, ...],
        horizons: tuple[int, ...],
        start_date,
        end_date,
    ) -> pd.DataFrame:
        keys = dataset.key_columns
        if keys != ("code", "trade_date"):
            raise ResearchDataContractError("current assembler requires ('code', 'trade_date') keys")
        universe = normalize_trade_dates(dataset.universe, context="research universe")
        factors = normalize_trade_dates(factors, context="factor frame")
        labels = normalize_trade_dates(labels, context="label frame")
        for name, frame in (("research universe", universe), ("factor frame", factors), ("label frame", labels)):
            validate_unique_keys(frame, keys, context=name)

        label_columns = tuple(f"fwd_ret_{value}d" for value in horizons)
        require_columns(factors, (*keys, *features), context="factor frame")
        require_columns(labels, (*keys, *label_columns), context="label frame")

        factor_payload = [column for column in factors.columns if column not in keys and column not in universe.columns]
        merged = universe.merge(factors[[*keys, *factor_payload]], on=list(keys), how="inner", validate="one_to_one")
        merged = merged.merge(labels[[*keys, *label_columns]], on=list(keys), how="left", validate="one_to_one")
        merged = merged[
            pd.to_datetime(merged["trade_date"]).between(
                pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
            )
        ].copy()
        require_columns(merged, (*keys, *required_columns, *features, *label_columns), context="analysis panel")
        merged = merged.sort_values(list(keys), kind="mergesort").reset_index(drop=True)
        validate_unique_keys(merged, keys, context="analysis panel")
        return merged
