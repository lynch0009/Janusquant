"""Weekly report metadata helpers.

File publication is intentionally delegated to the generic ResearchReporter.
"""

from __future__ import annotations

from dataclasses import asdict

from .config import RegimeWeeklyFillResearchConfig


def weekly_report_metadata(config: RegimeWeeklyFillResearchConfig) -> dict:
    return {"weekly_regime": asdict(config)}


__all__ = ["weekly_report_metadata"]
