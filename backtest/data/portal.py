"""Compatibility module for the DuckDB data portal.

The legacy Mongo-backed portal implementation has been removed from the main
code path. Import DuckDBDataPortal or DataPortal from this module.
"""

from __future__ import annotations

from backtest.data.duckdb_portal import CachedDuckDBDataPortal, DuckDBDataPortal

DataPortal = DuckDBDataPortal
CachedDataPortal = CachedDuckDBDataPortal

__all__ = [
    "CachedDataPortal",
    "CachedDuckDBDataPortal",
    "DataPortal",
    "DuckDBDataPortal",
]
