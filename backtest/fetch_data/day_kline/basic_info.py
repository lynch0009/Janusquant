from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from backtest.db import DuckDBConfig
from backtest.db.duckdb_write import upsert_frame
from backtest.utils import (
    is_delisted_basic_doc,
    is_hs_a_share_code,
    is_supported_a_stock_xt_code,
    normalize_internal_code,
    parse_basic_date,
    safe_float,
    to_xt_code,
)
from backtest.utils.log import log_event

from .constants import BASIC_INFO_COLLECTION, HS_A_SHARE_SECTOR_NAME
from .models import BasicInfoSyncResult, StockMeta
from .xt_details import fetch_xt_detail_map


def infer_exchange(xt_code: str, detail: dict[str, Any] | None = None) -> str:
    if "." in str(xt_code):
        return str(xt_code).split(".", 1)[1].lower()
    if detail:
        exchange = str(detail.get("ExchangeID") or detail.get("ExchangeCode") or "").strip().lower()
        if exchange in {"sh", "sz", "bj"}:
            return exchange
    return ""


def infer_market_from_code(code: str) -> str:
    normalized = normalize_internal_code(code)
    exchange, symbol = normalized.split(".", 1)
    if exchange == "sh" and symbol.startswith(("68", "69")):
        return "科创板"
    if exchange == "sz" and symbol.startswith(("30", "31")):
        return "创业板"
    return "主板"


def build_basic_info_doc(xt_code: str, detail: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """把 xtquant 合约详情转为 A_stock_market_basic_info 的最小可用文档。"""

    now = now or datetime.now()
    code = normalize_internal_code(xt_code)
    ipo_date = parse_basic_date(detail.get("OpenDate") or detail.get("CreateDate"))
    out_date = parse_basic_date(detail.get("ExpireDate"))
    # IsTrading 表示当前是否可交易，实测正常上市股票也可能为 False，不能用来判断上市状态。
    status = out_date is None or out_date > now

    doc = {
        "code": code,
        "code_name": str(detail.get("InstrumentName") or detail.get("ProductName") or "").strip(),
        "type": 1,
        "status": status,
        "market": infer_market_from_code(code),
        "listing_status": "listed" if status else "delisted",
        "basic_info_source": "xtquant",
        "xt_code": to_xt_code(code),
        "updated_at": now,
        "created_at": now,
    }
    if ipo_date is not None:
        doc["ipoDate"] = ipo_date
    if out_date is not None:
        doc["outDate"] = out_date
    return doc


def build_basic_info_operations(docs: Sequence[dict[str, Any]], now: datetime | None = None) -> list[dict[str, Any]]:
    return list(docs)


def stock_meta_from_basic_doc(doc: dict[str, Any], detail: dict[str, Any] | None = None) -> StockMeta:
    code = normalize_internal_code(str(doc.get("code", "")))
    float_volume = safe_float((detail or {}).get("FloatVolume"))
    return StockMeta(
        code=code,
        xt_code=to_xt_code(code),
        code_name=str(doc.get("code_name", "")).strip(),
        ipo_date=parse_basic_date(doc.get("ipoDate")),
        out_date=parse_basic_date(doc.get("outDate")),
        float_volume=float_volume,
    )


def collect_xt_hs_stock_codes(xtdata_client) -> list[str]:
    """只读取沪深 A 股板块成分，不调用 download_sector_data，避免定时任务卡在板块下载。"""

    try:
        sector_codes = xtdata_client.get_stock_list_in_sector(HS_A_SHARE_SECTOR_NAME) or []
    except Exception as exc:
        raise RuntimeError(f"failed to get xtquant sector stock list: {exc}") from exc
    xt_codes = sorted(
        {
            to_xt_code(normalize_internal_code(str(code)))
            for code in sector_codes
            if is_supported_a_stock_xt_code(str(code)) and is_hs_a_share_code(str(code))
        }
    )
    if not xt_codes:
        raise RuntimeError(f"xtquant sector {HS_A_SHARE_SECTOR_NAME} returned no 沪深 A-share codes")
    return xt_codes


def fetch_xt_basic_docs(xtdata_client, xt_codes: Sequence[str], now: datetime | None = None) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    docs: list[dict[str, Any]] = []
    details_by_code: dict[str, dict[str, Any]] = {}
    failed_codes: list[str] = []
    details_by_xt_code = fetch_xt_detail_map(xtdata_client, xt_codes)
    for xt_code in xt_codes:
        detail = details_by_xt_code.get(xt_code, {})
        if not detail:
            log_event("warning", "empty_instrument_detail", xt_code=xt_code)
            failed_codes.append(normalize_internal_code(xt_code))
            continue
        doc = build_basic_info_doc(xt_code, detail, now=now)
        docs.append(doc)
        details_by_code[doc["code"]] = detail
    return docs, details_by_code, failed_codes


def load_basic_docs(cfg: DuckDBConfig, *, hs_only: bool = False) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    frame = cfg.fetch_df(
        f"""
        select code, code_name, ipoDate, outDate, market, status, listing_status
        from "{BASIC_INFO_COLLECTION}"
        order by code
        """
    )
    for doc in frame.to_dict("records"):
        try:
            code = normalize_internal_code(str(doc.get("code", "")))
        except ValueError:
            continue
        if hs_only and not is_hs_a_share_code(code):
            continue
        normalized = dict(doc)
        normalized["code"] = code
        docs.append(normalized)
    return docs


def build_stock_pool_diff_rows(
    xt_codes: Sequence[str],
    basic_docs: Sequence[dict[str, Any]],
    *,
    today: datetime | None = None,
) -> list[dict[str, Any]]:
    today = today or datetime.now()
    xt_code_set = {normalize_internal_code(code) for code in xt_codes}
    db_by_code = {normalize_internal_code(str(doc["code"])): doc for doc in basic_docs}
    rows: list[dict[str, Any]] = []
    for code in sorted(xt_code_set - set(db_by_code)):
        rows.append(
            {
                "diff_type": "xt_only",
                "code": code,
                "xt_code": to_xt_code(code),
                "code_name": "",
                "reason": "xtquant_has_code_but_duckdb_missing",
            }
        )
    for code in sorted(set(db_by_code) - xt_code_set):
        if is_delisted_basic_doc(db_by_code[code], today):
            continue
        rows.append(
            {
                "diff_type": "db_only",
                "code": code,
                "xt_code": to_xt_code(code),
                "code_name": str(db_by_code[code].get("code_name", "")),
                "reason": "duckdb_has_code_but_xtquant_sector_missing",
            }
        )
    return rows


def sync_incremental_basic_info(
    cfg: DuckDBConfig,
    xtdata_client,
    *,
    now: datetime,
    batch_size: int,
) -> BasicInfoSyncResult:
    xt_codes = collect_xt_hs_stock_codes(xtdata_client)
    basic_docs = load_basic_docs(cfg, hs_only=True)
    diff_rows = build_stock_pool_diff_rows(xt_codes, basic_docs, today=now)
    new_xt_codes = [row["xt_code"] for row in diff_rows if row["diff_type"] == "xt_only"]
    if not new_xt_codes:
        return BasicInfoSyncResult((), (), tuple(diff_rows), {"matched": 0, "modified": 0, "upserted": 0})

    new_docs, _details_by_code, failed_codes = fetch_xt_basic_docs(xtdata_client, new_xt_codes, now=now)
    names_by_code = {doc["code"]: doc.get("code_name", "") for doc in new_docs}
    failed_set = set(failed_codes)
    enriched_rows: list[dict[str, Any]] = []
    for row in diff_rows:
        enriched = dict(row)
        if row["diff_type"] == "xt_only":
            enriched["code_name"] = names_by_code.get(row["code"], "")
            if row["code"] in failed_set:
                enriched["reason"] = "xtquant_detail_missing"
        enriched_rows.append(enriched)

    write_summary = upsert_frame(
        cfg,
        BASIC_INFO_COLLECTION,
        build_basic_info_operations(new_docs, now=now),
        key_columns=("code",),
    )
    return BasicInfoSyncResult(
        tuple(new_docs),
        tuple(failed_codes),
        tuple(enriched_rows),
        {"matched": 0, "modified": 0, "upserted": int(write_summary.rows_written)},
    )
