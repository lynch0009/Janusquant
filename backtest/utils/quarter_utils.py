"""季度日期工具。"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable


def quarter_from_date(value: datetime) -> tuple[int, int]:
    return value.year, ((value.month - 1) // 3) + 1


def quarter_end(year: int, quarter: int) -> datetime:
    month = quarter * 3
    day = 30 if month in (6, 9) else 31
    return datetime(year, month, day)


def next_quarter(year: int, quarter: int) -> tuple[int, int]:
    if quarter == 4:
        return year + 1, 1
    return year, quarter + 1


def iter_quarters(start_date: datetime, end_date: datetime) -> list[tuple[int, int]]:
    year, quarter = quarter_from_date(start_date)
    end_year, end_quarter = quarter_from_date(end_date)
    return list(iter_quarter_pairs(year, quarter, end_year, end_quarter))


def iter_quarter_pairs(
    start_year: int,
    start_quarter: int,
    end_year: int,
    end_quarter: int,
) -> Iterable[tuple[int, int]]:
    year, quarter = start_year, start_quarter
    while (year, quarter) <= (end_year, end_quarter):
        yield year, quarter
        year, quarter = next_quarter(year, quarter)


def format_quarter(year: int, quarter: int) -> str:
    return f"{year}Q{quarter}"


def resolve_incremental_target_quarter(today: datetime) -> tuple[int, int]:
    month = today.month
    if month <= 3:
        return today.year - 1, 3
    if month <= 6:
        return today.year, 1
    if month <= 9:
        return today.year, 2
    return today.year, 3
