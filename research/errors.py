"""Research framework exception hierarchy."""

from __future__ import annotations


class ResearchError(Exception):
    """Base class for research framework failures."""


class ResearchConfigError(ResearchError, ValueError):
    """Raised when a research request is internally inconsistent."""


class ResearchDataContractError(ResearchError, ValueError):
    """Raised when input data violates the declared research contract."""


class ResearchFactorError(ResearchError, ValueError):
    """Raised when factor registration or dependency resolution fails."""


class ResearchCacheError(ResearchError, RuntimeError):
    """Raised when a cache entry cannot be read or rebuilt."""
