from __future__ import annotations

from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.db import DuckDBConfig
from backtest.db.precision import normalize_amount, normalize_price, normalize_volume
from backtest.fetch_data.day_kline_common import normalize_day_kline_frame
from backtest.utils import to_trade_datetime

from .constants import FINANCE_COLLECTION, LOCAL_RENAME_MAP
from .universe import normalize_symbol


def round_turn(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 5)


def read_local_csv(path: Path, symbol_map: dict[str, str]) -> tuple[pd.DataFrame, int]:
    """读取单日本地 CSV，并只保留可映射到内部股票代码的行。"""

    if not path.exists():
        return pd.DataFrame(), 0

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            frame = pd.read_csv(path, encoding=encoding, dtype={"股票代码": "string"})
            break
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
    else:  # pragma: no cover - fallback path
        raise RuntimeError(f"failed to read local csv {path}: {last_error}")

    required = set(LOCAL_RENAME_MAP)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    working = frame[list(LOCAL_RENAME_MAP)].rename(columns=LOCAL_RENAME_MAP).copy()
    working["symbol"] = working["symbol"].map(normalize_symbol)
    working["code"] = working["symbol"].map(symbol_map)
    unmatched_count = int(working["code"].isna().sum())
    # 本地文件里会有少量当前库里无法识别的代码，这里直接跳过并记统计。
    working = working.dropna(subset=["code"]).copy()

    if working.empty:
        return pd.DataFrame(), unmatched_count

    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"]).copy()
    working = working[
        ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]
    ].rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "preclose": "preclose",
            "volume": "volume",
            "amount": "amount",
            "pctChg": "pctChg",
        }
    )
    working = working.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)
    return normalize_day_kline_frame(working), unmatched_count


def load_finance_index(cfg: DuckDBConfig, codes: list[str], end_date: datetime) -> dict[str, tuple[list[datetime], list[float | None]]]:
    """预加载流通股本时间轴，给本地日线补 turn 使用。"""

    if not codes:
        return {}
    placeholders = ", ".join(["?"] * len(codes))
    frame = cfg.fetch_df(
        f"""
        select code, pubDate, statDate, liqaShare
        from "{FINANCE_COLLECTION}"
        where code in ({placeholders}) and pubDate <= ?
        order by code, pubDate, statDate
        """,
        [*codes, end_date],
    )

    result: dict[str, tuple[list[datetime], list[float | None]]] = {}
    for row in frame.to_dict("records"):
        code = str(row.get("code", "")).strip().lower()
        pub_date = row.get("pubDate")
        if not code or pub_date is None:
            continue
        liqa_share = pd.to_numeric(row.get("liqaShare"), errors="coerce")
        dates, values = result.setdefault(code, ([], []))
        dates.append(to_trade_datetime(pub_date))
        values.append(None if pd.isna(liqa_share) else float(liqa_share))
    return result


def lookup_liqa_share(
    finance_index: dict[str, tuple[list[datetime], list[float | None]]],
    code: str,
    trade_date: datetime,
) -> float | None:
    """取某交易日可见的最新流通股本。"""

    item = finance_index.get(code)
    if item is None:
        return None
    dates, values = item
    index = bisect_right(dates, trade_date) - 1
    if index < 0:
        return None
    value = values[index]
    if value in (None, 0):
        return None
    return float(value)


def build_local_doc(
    local_row: pd.Series,
    code: str,
    trade_date: datetime,
    current_isst: bool,
    finance_index: dict[str, tuple[list[datetime], list[float | None]]],
) -> dict[str, Any]:
    """把本地单行日线转成最终日线记录。"""

    liqa_share = lookup_liqa_share(finance_index, code, trade_date)
    volume = normalize_volume(local_row.get("v"))
    turn = None
    if liqa_share not in (None, 0) and volume not in (None, 0):
        turn = round_turn(float(volume) / float(liqa_share) * 100.0)

    pct_value = pd.to_numeric(local_row.get("pctChg"), errors="coerce")
    return {
        "code": code,
        "date": trade_date,
        "o": normalize_price(local_row.get("o")),
        "h": normalize_price(local_row.get("h")),
        "l": normalize_price(local_row.get("l")),
        "c": normalize_price(local_row.get("c")),
        "prec": normalize_price(local_row.get("prec")),
        "v": volume,
        "a": normalize_amount(local_row.get("a")),
        "turn": turn,
        "pctChg": None if pd.isna(pct_value) else float(pct_value),
        "tradestatus": True,
        "isST": bool(current_isst),
    }
