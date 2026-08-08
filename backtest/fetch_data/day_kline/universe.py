from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from backtest.db import DuckDBConfig
from backtest.utils import (
    is_bj_code,
    is_delisted_basic_doc,
    is_hs_a_share_code,
    is_index_code,
    normalize_internal_code,
    safe_float,
    to_trade_datetime,
    to_xt_code,
)

from .basic_info import stock_meta_from_basic_doc
from .constants import (
    BASIC_INFO_COLLECTION,
    DAY_COLLECTION,
    FIXED_DAY_KLINE_INDEX_CODES,
    FIXED_DAY_KLINE_INDEX_NAME,
    MIN_DAY_KLINE_START_DATE,
)
from .models import SecurityMeta, StockMeta


BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")
DAY_KLINE_CLOSE_TIME = time(15, 45)


def latest_available_day_kline_date(now: datetime) -> datetime:
    """Return the latest calendar date whose daily bar may be fetched."""

    local_now = now.replace(tzinfo=BEIJING_TIMEZONE) if now.tzinfo is None else now.astimezone(BEIJING_TIMEZONE)
    result = datetime(local_now.year, local_now.month, local_now.day)
    if local_now.time().replace(tzinfo=None) < DAY_KLINE_CLOSE_TIME:
        result -= timedelta(days=1)
    return result


def fixed_day_kline_index_metas() -> list[StockMeta]:
    """固定维护的指数/指数类标的，随每日 xtquant 日 K 一起增量更新。"""

    metas: list[StockMeta] = []
    for code in FIXED_DAY_KLINE_INDEX_CODES:
        normalized = normalize_internal_code(code)
        metas.append(
            StockMeta(
                code=normalized,
                xt_code=to_xt_code(normalized),
                code_name=FIXED_DAY_KLINE_INDEX_NAME,
                ipo_date=None,
                out_date=None,
                float_volume=None,
            )
        )
    return metas


def merge_fixed_day_kline_indexes(stock_metas: Sequence[StockMeta]) -> tuple[list[StockMeta], int]:
    """把固定指数加入更新范围；若代码已由股票 universe 覆盖，则保留股票元信息。"""

    metas_by_code = {meta.code: meta for meta in stock_metas}
    fixed_metas = fixed_day_kline_index_metas()
    for meta in fixed_metas:
        if meta.code in metas_by_code:
            continue
        metas_by_code[meta.code] = meta
    return sorted(metas_by_code.values(), key=lambda item: item.code), len({meta.code for meta in fixed_metas})


def load_latest_state_map(cfg: DuckDBConfig, codes: Sequence[str], end_date: datetime) -> dict[str, dict[str, Any]]:
    if not codes:
        return {}
    scope = pd.DataFrame({"code": sorted(set(codes))})
    with cfg.registered_frame("_latest_state_scope", scope):
        frame = cfg.fetch_df(
            f"""
            select
                code,
                state.date as date,
                state.close as close,
                state.is_st as isST
            from (
                select
                    scope.code,
                    arg_max(
                        struct_pack(date := history.date, close := history.c, is_st := history.isST),
                        history.date
                    ) as state
                from _latest_state_scope as scope
                join "{DAY_COLLECTION}" as history on history.code = scope.code
                where history.date <= ?
                group by scope.code
            )
            order by code
            """,
            [end_date],
        )
    result: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        code = str(row.get("code", "")).strip().lower()
        if code:
            result[code] = {
                "date": to_trade_datetime(row["date"]),
                "close": safe_float(row.get("close")),
                "isST": bool(row.get("isST", False)),
            }
    return result


def load_previous_state_before_start_map(
    cfg: DuckDBConfig,
    code_start_map: dict[str, datetime],
) -> dict[str, dict[str, Any]]:
    """Load one internally consistent row strictly before each code's backfill start."""

    if not code_start_map:
        return {}
    scope = pd.DataFrame(
        [
            {"code": code, "start_date": to_trade_datetime(start_date)}
            for code, start_date in sorted(code_start_map.items())
        ]
    )
    with cfg.registered_frame("_previous_state_scope", scope):
        frame = cfg.fetch_df(
            f"""
            select
                code,
                state.date as date,
                state.close as close,
                state.is_st as isST
            from (
                select
                    scope.code,
                    arg_max(
                        struct_pack(date := history.date, close := history.c, is_st := history.isST),
                        history.date
                    ) as state
                from _previous_state_scope as scope
                join "{DAY_COLLECTION}" as history
                  on history.code = scope.code
                 and history.date < scope.start_date
                group by scope.code
            )
            order by code
            """
        )
    return {
        str(row["code"]).strip().lower(): {
            "date": to_trade_datetime(row["date"]),
            "close": safe_float(row.get("close")),
            "isST": bool(row.get("isST", False)),
        }
        for row in frame.to_dict("records")
        if str(row.get("code", "")).strip()
    }


def load_previous_trade_date_stock_count(
    cfg: DuckDBConfig,
    latest_trade_date: datetime,
) -> tuple[datetime | None, int]:
    frame = cfg.fetch_df(
        f"""
        with previous as (
            select max(history.date) as trade_date
            from "{DAY_COLLECTION}" as history
            inner join "{BASIC_INFO_COLLECTION}" as basic on basic.code = history.code
            where history.date < ?
              and (lower(basic.code) like 'sh.%' or lower(basic.code) like 'sz.%')
        )
        select
            previous.trade_date,
            count(distinct history.code) as stock_count
        from previous
        left join "{DAY_COLLECTION}" as history on history.date = previous.trade_date
        left join "{BASIC_INFO_COLLECTION}" as basic on basic.code = history.code
        where basic.code is not null
          and (lower(history.code) like 'sh.%' or lower(history.code) like 'sz.%')
        group by previous.trade_date
        """,
        [to_trade_datetime(latest_trade_date)],
    )
    if frame.empty or pd.isna(frame.iloc[0].get("trade_date")):
        return None, 0
    return (
        to_trade_datetime(frame.iloc[0]["trade_date"]),
        int(frame.iloc[0]["stock_count"]),
    )


def build_code_start_map(
    metas: Sequence[StockMeta],
    latest_state_map: dict[str, dict[str, Any]],
    end_date: datetime,
    *,
    min_start_date: datetime = MIN_DAY_KLINE_START_DATE,
) -> dict[str, datetime]:
    result: dict[str, datetime] = {}
    min_start_date = to_trade_datetime(min_start_date)
    for meta in metas:
        # 日常任务只维护 2020 年以来的增量；更早的历史缺口不再由 QMT 定时任务修复。
        latest = latest_state_map.get(meta.code)
        if latest and latest.get("date") is not None:
            start_date = latest["date"] + timedelta(days=1)
        elif meta.ipo_date is not None:
            start_date = to_trade_datetime(meta.ipo_date)
        else:
            start_date = end_date

        if meta.ipo_date is not None and start_date < to_trade_datetime(meta.ipo_date):
            start_date = to_trade_datetime(meta.ipo_date)
        if start_date < min_start_date:
            start_date = min_start_date
        if meta.out_date is not None and start_date > to_trade_datetime(meta.out_date):
            continue
        if start_date <= end_date:
            result[meta.code] = start_date
    return result


def expected_dates_for_meta(meta: StockMeta, start_date: datetime, end_date: datetime, trade_dates: Sequence[datetime]) -> list[datetime]:
    result: list[datetime] = []
    for trade_date in trade_dates:
        if trade_date < start_date or trade_date > end_date:
            continue
        if meta.ipo_date is not None and trade_date < to_trade_datetime(meta.ipo_date):
            continue
        if meta.out_date is not None and trade_date > to_trade_datetime(meta.out_date):
            continue
        result.append(trade_date)
    return result


def build_expected_dates_by_code(
    metas: Sequence[StockMeta],
    code_start_map: dict[str, datetime],
    trade_dates: Sequence[datetime],
    end_date: datetime,
) -> dict[str, set[datetime]]:
    return {
        meta.code: set(expected_dates_for_meta(meta, code_start_map[meta.code], end_date, trade_dates))
        for meta in metas
        if meta.code in code_start_map
    }


def build_day_kline_update_universe(
    basic_docs: Sequence[dict[str, Any]],
    stock_pool_diff_rows: Sequence[dict[str, Any]],
    today: datetime,
    *,
    excluded_db_only_codes: Sequence[str] | None = None,
) -> tuple[list[StockMeta], dict[str, int]]:
    if excluded_db_only_codes is None:
        excluded_db_only_codes = [
            str(row["code"])
            for row in stock_pool_diff_rows
            if row.get("diff_type") == "db_only"
        ]
    db_only_codes = {
        normalize_internal_code(str(code))
        for code in (excluded_db_only_codes or ())
    }
    metas: list[StockMeta] = []
    stats = {
        "day_kline_excluded_db_only_count": 0,
        "day_kline_excluded_inactive_count": 0,
    }
    for doc in basic_docs:
        try:
            code = normalize_internal_code(str(doc.get("code", "")))
        except ValueError:
            continue
        if is_bj_code(code) or not is_hs_a_share_code(code):
            continue
        if code in db_only_codes:
            stats["day_kline_excluded_db_only_count"] += 1
            continue
        if is_delisted_basic_doc(doc, today):
            stats["day_kline_excluded_inactive_count"] += 1
            continue
        metas.append(stock_meta_from_basic_doc(doc))
    return sorted(metas, key=lambda item: item.code), stats


def normalize_symbol(value: Any) -> str:
    return str(value).strip().zfill(6)


def load_existing_index_codes(cfg: DuckDBConfig) -> list[str]:
    frame = cfg.fetch_df(
        f"""
        select distinct code
        from "{DAY_COLLECTION}"
        where code is not null
        except
        select distinct code
        from "{BASIC_INFO_COLLECTION}"
        where code is not null
        """
    )
    if frame.empty or "code" not in frame.columns:
        return []
    return sorted(code for code in frame["code"].astype(str).str.strip().str.lower().tolist() if is_index_code(code))


def load_manual_universe(cfg: DuckDBConfig, end_date: datetime) -> dict[str, SecurityMeta]:
    frame = cfg.fetch_df(
        f"""
        select code, code_name, ipoDate, outDate
        from "{BASIC_INFO_COLLECTION}"
        where ipoDate <= ?
        order by code
        """,
        [end_date],
    )

    universe: dict[str, SecurityMeta] = {}
    for doc in frame.to_dict("records"):
        code = str(doc.get("code", "")).strip()
        if not code:
            continue
        try:
            lowered = normalize_internal_code(code)
        except ValueError:
            continue
        universe[lowered] = SecurityMeta(
            code=lowered,
            code_name=str(doc.get("code_name", "")).strip(),
            ipo_date=to_trade_datetime(doc["ipoDate"]) if doc.get("ipoDate") is not None else None,
            out_date=to_trade_datetime(doc["outDate"]) if doc.get("outDate") is not None else None,
            is_index=False,
        )

    for lowered in load_existing_index_codes(cfg):
        universe.setdefault(
            lowered,
            SecurityMeta(code=lowered, code_name=lowered, ipo_date=None, out_date=None, is_index=True),
        )
    return universe


def build_symbol_code_map(universe: dict[str, SecurityMeta]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for code, meta in universe.items():
        if meta.is_index:
            continue
        mapping[code.split(".", 1)[1]] = code
    return mapping


def resolve_requested_codes(selector_text: str, universe: dict[str, SecurityMeta]) -> list[str]:
    if not selector_text.strip():
        return sorted(universe)

    symbol_map = build_symbol_code_map(universe)
    resolved: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()

    for raw_selector in selector_text.split(","):
        selector = raw_selector.strip()
        if not selector:
            continue

        candidate = ""
        lowered = selector.lower()
        if lowered in universe:
            candidate = lowered
        elif "." in selector:
            try:
                candidate = normalize_internal_code(selector)
            except ValueError:
                candidate = ""
        else:
            candidate = symbol_map.get(normalize_symbol(selector), "")

        if candidate and candidate in universe:
            if candidate not in seen:
                seen.add(candidate)
                resolved.append(candidate)
            continue
        missing.append(selector)

    if missing:
        raise ValueError(f"unknown stock selectors: {', '.join(missing)}")
    return sorted(resolved)


def build_manual_code_start_map(
    codes: Sequence[str],
    universe: dict[str, SecurityMeta],
    explicit_start_date: datetime,
    end_date: datetime,
) -> dict[str, datetime]:
    code_start_map: dict[str, datetime] = {}
    for code in codes:
        meta = universe[code]
        start_date = explicit_start_date
        if meta.ipo_date is not None and start_date < meta.ipo_date:
            start_date = meta.ipo_date
        if meta.out_date is not None and start_date > meta.out_date:
            continue
        if start_date > end_date:
            continue
        code_start_map[code] = start_date
    return code_start_map
