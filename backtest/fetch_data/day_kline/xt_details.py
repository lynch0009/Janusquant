from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from backtest.utils import normalize_internal_code, safe_float

from .constants import FIXED_DAY_KLINE_INDEX_CODES
from .models import StockMeta


def fetch_xt_detail_map(xtdata_client: Any, xt_codes: Sequence[str]) -> dict[str, dict[str, Any]]:
    requested_codes = [str(code).strip() for code in xt_codes if str(code).strip()]
    if not requested_codes:
        return {}

    if hasattr(xtdata_client, "get_instrument_detail_list"):
        try:
            details = xtdata_client.get_instrument_detail_list(requested_codes, True)
        except Exception:
            details = None
        if isinstance(details, dict) and details:
            return {str(code): detail for code, detail in details.items() if isinstance(detail, dict)}

    details_by_xt_code: dict[str, dict[str, Any]] = {}
    for xt_code in requested_codes:
        try:
            detail = xtdata_client.get_instrument_detail(xt_code, iscomplete=True) or {}
        except TypeError:
            detail = xtdata_client.get_instrument_detail(xt_code) or {}
        except Exception:
            detail = {}
        if isinstance(detail, dict) and detail:
            details_by_xt_code[xt_code] = detail
    return details_by_xt_code


def enrich_stock_metas_with_xt_details(
    metas: Sequence[StockMeta],
    details_by_xt_code: dict[str, dict[str, Any]],
    *,
    active_codes: set[str] | None = None,
) -> tuple[list[StockMeta], dict[str, int]]:
    active_codes = {normalize_internal_code(code) for code in active_codes} if active_codes is not None else None
    enriched: list[StockMeta] = []
    present_count = 0
    missing_count = 0
    zero_count = 0

    for meta in metas:
        is_active_stock = (active_codes is None or meta.code in active_codes) and meta.code not in FIXED_DAY_KLINE_INDEX_CODES
        if not is_active_stock:
            enriched.append(meta)
            continue

        detail = details_by_xt_code.get(meta.xt_code)
        float_volume = safe_float((detail or {}).get("FloatVolume"))
        if float_volume is None:
            missing_count += 1
            enriched.append(meta)
            continue
        if float_volume == 0:
            zero_count += 1
            enriched.append(replace(meta, float_volume=None))
            continue
        present_count += 1
        enriched.append(replace(meta, float_volume=float_volume))

    stats = {
        "day_kline_float_volume_present_count": present_count,
        "day_kline_float_volume_missing_count": missing_count,
        "day_kline_float_volume_zero_count": zero_count,
    }
    return enriched, stats
