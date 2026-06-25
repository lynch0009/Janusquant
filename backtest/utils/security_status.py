"""证券基础信息状态判断工具。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from .datetime_utils import to_pydatetime


def parse_basic_date(value: Any) -> datetime | None:
    """把 basic_info 中的日期字段统一成零点 datetime。"""

    normalized = to_pydatetime(value)
    if normalized in (None, "", 0, "0", "19700101", "19000101", "99999999"):
        return None
    try:
        if pd.isna(normalized):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(normalized, datetime):
        return datetime(normalized.year, normalized.month, normalized.day)
    if isinstance(normalized, date):
        return datetime(normalized.year, normalized.month, normalized.day)

    text = str(normalized).strip()
    if not text:
        return None
    if text in {"0", "19700101", "19000101", "99999999"}:
        return None
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d")
    try:
        parsed = datetime.fromisoformat(text[:10])
    except ValueError:
        return None
    return datetime(parsed.year, parsed.month, parsed.day)


def is_st_name(name: Any) -> bool:
    """按证券名称判断是否带 ST 风险警示。"""

    return "ST" in str(name or "").strip().upper()


def is_delisted_basic_doc(doc: dict[str, Any], today: datetime) -> bool:
    """按 basic_info 字段判断股票是否已经退市或不再 active。"""

    if doc.get("status") is False:
        return True
    if str(doc.get("listing_status") or "").strip().lower() == "delisted":
        return True
    out_date = parse_basic_date(doc.get("outDate"))
    today_date = parse_basic_date(today)
    return out_date is not None and today_date is not None and out_date <= today_date
