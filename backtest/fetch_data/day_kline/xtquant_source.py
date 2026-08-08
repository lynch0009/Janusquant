from __future__ import annotations

from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta
from collections.abc import Mapping
from typing import Any, Iterable, Sequence

import pandas as pd

from backtest.db.precision import normalize_amount, normalize_price, normalize_volume
from backtest.utils import is_blank, safe_float, safe_int, to_trade_datetime

from .constants import FIXED_DAY_KLINE_INDEX_CODES, XT_DAY_FIELDS, XT_TIMEZONE_OFFSET_HOURS
from .models import StockMeta, XtDayRow


def extract_xt_stock_frame(market_data: dict[str, pd.DataFrame], xt_code: str) -> pd.DataFrame:
    """把 xtquant field->DataFrame 的横向结构，摊平成单只股票按日期排列的行。"""

    columns: set[Any] = set()
    series_by_field: dict[str, pd.Series] = {}
    for field in XT_DAY_FIELDS:
        frame = market_data.get(field)
        if not isinstance(frame, pd.DataFrame) or xt_code not in frame.index:
            continue
        selected = frame.loc[xt_code]
        if isinstance(selected, pd.DataFrame):
            selected = selected.iloc[-1]
        series_by_field[field] = selected
        columns.update(selected.index.tolist())
    if not columns:
        return pd.DataFrame()

    ordered_columns = sorted(columns)
    result = pd.DataFrame({"column_time": ordered_columns})
    for field in XT_DAY_FIELDS:
        series = series_by_field.get(field)
        result[field] = None if series is None else series.reindex(ordered_columns).tolist()
    return result


def resolve_xt_trade_date(row: Mapping[str, Any] | pd.Series) -> datetime | None:
    parsed_date = row.get("_trade_date")
    if isinstance(parsed_date, datetime):
        return parsed_date
    raw_time = row.get("time")
    if is_blank(raw_time):
        raw_time = row.get("column_time")
    numeric = pd.to_numeric(raw_time, errors="coerce")
    if not pd.isna(numeric):
        parsed = pd.to_datetime(float(numeric), unit="ms", errors="coerce") + pd.Timedelta(
            hours=XT_TIMEZONE_OFFSET_HOURS
        )
    else:
        parsed = pd.to_datetime(raw_time, errors="coerce")
    if pd.isna(parsed):
        return None
    return datetime(parsed.year, parsed.month, parsed.day)


def is_suspended_xt_row(row: Mapping[str, Any] | pd.Series) -> bool:
    flag = safe_int(row.get("suspendFlag"))
    return flag == 1


def normalize_xt_volume(value: Any) -> int | None:
    """Convert xtquant daily volume from lots to the table's share unit."""

    if value is None or pd.isna(value):
        return None
    try:
        # Index aggregates may contain fractional lots.  Convert first so the
        # final integer keeps share precision instead of rounding lots early.
        share_volume = Decimal(str(value)) * Decimal("100")
    except (InvalidOperation, TypeError, ValueError):
        return None
    return normalize_volume(share_volume)


def build_xt_day_doc(
    row: Mapping[str, Any] | pd.Series,
    meta: StockMeta,
    *,
    previous_close: float | None = None,
    is_st: bool = False,
) -> XtDayRow:
    trade_date = resolve_xt_trade_date(row)
    if trade_date is None:
        return XtDayRow(doc=None, date=None, missing_reason="missing_date")

    suspended = is_suspended_xt_row(row)
    open_price = normalize_price(row.get("open"))
    high_price = normalize_price(row.get("high"))
    low_price = normalize_price(row.get("low"))
    close_price = normalize_price(row.get("close"))
    is_index = meta.code in FIXED_DAY_KLINE_INDEX_CODES
    volume = normalize_xt_volume(row.get("volume"))
    amount = normalize_amount(row.get("amount"))
    preclose = normalize_price(row.get("preClose"))
    if preclose is None:
        preclose = normalize_price(row.get("preclose"))
    if preclose is None:
        preclose = normalize_price(previous_close)

    # 非停牌日如果缺价格、成交量/额字段，认为是 xtquant 数据层面的疑似缺失，交给 Baostock fallback。
    price_values = [open_price, high_price, low_price, close_price]
    has_valid_prices = all(value is not None and value > 0 for value in price_values)
    has_valid_amount = volume is not None and amount is not None and volume >= 0 and amount >= 0
    zero_trade_without_suspend = volume == 0 and amount == 0 and not suspended
    if (not has_valid_prices or not has_valid_amount or zero_trade_without_suspend) and not suspended:
        return XtDayRow(
            doc=None,
            date=trade_date,
            is_suspended=False,
            missing_reason="invalid_non_suspend_day",
        )

    pct_chg = None
    if preclose not in (None, 0) and close_price is not None:
        pct_chg = (float(close_price) - float(preclose)) / float(preclose) * 100.0

    turn = None
    if meta.float_volume not in (None, 0) and volume not in (None, 0):
        turn = float(volume) / float(meta.float_volume) * 100.0

    doc = {
        "code": meta.code,
        "date": trade_date,
        "o": open_price,
        "h": high_price,
        "l": low_price,
        "c": close_price,
        "prec": preclose,
        "v": volume,
        "a": amount,
        "turn": None if turn is None else round(float(turn), 5),
        "pctChg": None if pct_chg is None else float(pct_chg),
        "tradestatus": not suspended,
        "isST": bool(is_st),
    }
    return XtDayRow(doc=doc, date=trade_date, is_suspended=suspended)


def xt_market_data_to_docs(
    market_data: dict[str, pd.DataFrame],
    metas: Sequence[StockMeta],
    latest_state_map: dict[str, dict[str, Any]] | None = None,
    *,
    st_status_by_code: dict[str, dict[datetime, bool]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[datetime]]]:
    latest_state_map = latest_state_map or {}
    st_status_by_code = st_status_by_code or {}
    docs: list[dict[str, Any]] = []
    invalid_dates: dict[str, list[datetime]] = {}
    previous_close_by_code = {
        code: safe_float(state.get("close"))
        for code, state in latest_state_map.items()
        if state.get("close") is not None
    }

    for meta in metas:
        frame = extract_xt_stock_frame(market_data, meta.xt_code)
        if frame.empty:
            continue
        rows = frame.to_dict("records")
        for row in rows:
            trade_date = resolve_xt_trade_date(row)
            row["_trade_date"] = trade_date
        rows.sort(key=lambda row: row["_trade_date"] or datetime.min)
        for row in rows:
            trade_date = row["_trade_date"]
            result = build_xt_day_doc(
                row,
                meta,
                previous_close=previous_close_by_code.get(meta.code),
                is_st=bool(st_status_by_code.get(meta.code, {}).get(trade_date, False)),
            )
            if result.doc is not None:
                docs.append(result.doc)
                if result.doc.get("c") is not None:
                    # 同一批内顺序推进前收盘价，给 xtquant 不返回 preClose 的行兜底。
                    previous_close_by_code[meta.code] = safe_float(result.doc.get("c"))
            elif result.date is not None and result.missing_reason:
                invalid_dates.setdefault(meta.code, []).append(result.date)
    return docs, invalid_dates


def fetch_xt_market_data(xtdata_client, xt_codes: Sequence[str], start_date: datetime, end_date: datetime) -> dict[str, pd.DataFrame]:
    if not xt_codes:
        return {}
    start_text = start_date.strftime("%Y%m%d")
    end_text = end_date.strftime("%Y%m%d")
    # 先增量下载到 QMT 本地缓存，再从本地/客户端读取，和 xtquant 推荐用法保持一致。
    xtdata_client.download_history_data2(
        list(xt_codes),
        period="1d",
        start_time=start_text,
        end_time=end_text,
        incrementally=True,
    )
    return xtdata_client.get_market_data(
        field_list=list(XT_DAY_FIELDS),
        stock_list=list(xt_codes),
        period="1d",
        start_time=start_text,
        end_time=end_text,
        count=-1,
        dividend_type="none",
        fill_data=True,
    )


def iter_xt_sync_batches(
    metas: Sequence[StockMeta],
    code_start_map: dict[str, datetime],
    *,
    xt_batch_size: int,
) -> Iterable[tuple[datetime, list[StockMeta]]]:
    """按增量起点分组后再分批，避免少数长缺口股票拖累整批回拉历史。"""

    metas_by_start: dict[datetime, list[StockMeta]] = {}
    for meta in metas:
        start_date = code_start_map.get(meta.code)
        if start_date is None:
            continue
        metas_by_start.setdefault(start_date, []).append(meta)

    batch_size = max(1, int(xt_batch_size))
    for start_date in sorted(metas_by_start):
        group = sorted(metas_by_start[start_date], key=lambda item: item.code)
        for index in range(0, len(group), batch_size):
            yield start_date, group[index : index + batch_size]
