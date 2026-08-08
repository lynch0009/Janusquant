from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from backtest.db import DuckDBConfig
from backtest.fetch_data.adjust_factor_baseline import (
    AdjustmentFactorBaselineWriteSummary,
    build_adjustment_factor_baseline,
    write_missing_adjustment_factor_baselines,
)
from backtest.fetch_data.core.writer import upsert_records
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


def _parse_xt_expire_date(value: Any) -> datetime | None:
    """Normalize QMT's no-expiry sentinel dates to a missing out-date."""

    parsed = parse_basic_date(value)
    # QMT may return values such as ``10001011`` for an active security with
    # no expiry date.  They parse as year 1000, but are not valid delisting
    # dates and cannot be represented by pandas' datetime64[ns].
    if parsed is not None and parsed.year < 1900:
        return None
    return parsed


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
    out_date = _parse_xt_expire_date(detail.get("ExpireDate"))
    # IsTrading 表示当前是否可交易，实测正常上市股票也可能为 False，不能用来判断上市状态。
    today = parse_basic_date(now)
    status = out_date is None or today is None or out_date >= today

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


def update_confirmed_delistings(
    cfg: DuckDBConfig,
    updates: Sequence[dict[str, Any]],
    *,
    dry_run: bool,
) -> int:
    if dry_run or not updates:
        return 0
    cfg.execute("begin transaction")
    try:
        for update in updates:
            cfg.execute(
                f"""
                update "{BASIC_INFO_COLLECTION}"
                set code_name = case
                        when ? is null or trim(?) = '' then code_name
                        else ?
                    end,
                    outDate = ?,
                    status = false,
                    listing_status = 'delisted',
                    outDate_source_field = 'ExpireDate',
                    xt_code = ?,
                    updated_at = ?
                where code = ?
                """,
                [
                    update.get("code_name"),
                    update.get("code_name"),
                    update.get("code_name"),
                    update["outDate"],
                    update["xt_code"],
                    update["updated_at"],
                    update["code"],
                ],
            )
        cfg.execute("commit")
    except Exception:
        cfg.execute("rollback")
        raise
    return len(updates)


def project_basic_docs(
    basic_docs: Sequence[dict[str, Any]],
    sync_result: BasicInfoSyncResult,
) -> list[dict[str, Any]]:
    """Apply the current basic-info changes in memory, including dry-run."""

    by_code = {
        normalize_internal_code(str(doc["code"])): dict(doc)
        for doc in basic_docs
        if str(doc.get("code", "")).strip()
    }
    for doc in sync_result.inserted_docs:
        by_code[doc["code"]] = dict(doc)
    for update in sync_result.confirmed_delisted_updates:
        current = dict(by_code.get(update["code"], {"code": update["code"]}))
        if str(update.get("code_name") or "").strip():
            current["code_name"] = str(update["code_name"]).strip()
        current.update(
            {
                "outDate": update["outDate"],
                "status": False,
                "listing_status": "delisted",
            }
        )
        by_code[update["code"]] = current
    return [by_code[code] for code in sorted(by_code)]


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
    dry_run: bool = False,
    initialize_adjust_factor_baselines: bool = False,
) -> BasicInfoSyncResult:
    xt_codes = collect_xt_hs_stock_codes(xtdata_client)
    basic_docs = load_basic_docs(cfg, hs_only=True)
    diff_rows = build_stock_pool_diff_rows(xt_codes, basic_docs, today=now)
    new_xt_codes = [row["xt_code"] for row in diff_rows if row["diff_type"] == "xt_only"]
    db_only_rows = [row for row in diff_rows if row["diff_type"] == "db_only"]
    basic_detail_xt_codes = sorted(
        set(new_xt_codes) | {str(row["xt_code"]) for row in db_only_rows}
    )
    requested_xt_codes = sorted(set(xt_codes) | {str(row["xt_code"]) for row in db_only_rows})
    details_by_xt_code = fetch_xt_detail_map(xtdata_client, requested_xt_codes)

    new_docs: list[dict[str, Any]] = []
    failed_codes: list[str] = []
    for xt_code in new_xt_codes:
        detail = details_by_xt_code.get(xt_code, {})
        if not detail:
            code = normalize_internal_code(xt_code)
            log_event("warning", "empty_instrument_detail", xt_code=xt_code)
            failed_codes.append(code)
            continue
        new_docs.append(build_basic_info_doc(xt_code, detail, now=now))

    confirmed_delisted_updates: list[dict[str, Any]] = []
    db_only_confirmed_delisted_codes: list[str] = []
    db_only_active_detail_codes: list[str] = []
    db_only_detail_missing_codes: list[str] = []
    for row in db_only_rows:
        code = normalize_internal_code(str(row["code"]))
        xt_code = str(row["xt_code"])
        detail = details_by_xt_code.get(xt_code, {})
        if not detail:
            db_only_detail_missing_codes.append(code)
            continue
        expire_date = _parse_xt_expire_date(detail.get("ExpireDate"))
        today = parse_basic_date(now)
        if expire_date is not None and today is not None and expire_date < today:
            confirmed_delisted_updates.append(
                {
                    "code": code,
                    "code_name": str(
                        detail.get("InstrumentName") or detail.get("ProductName") or ""
                    ).strip(),
                    "outDate": expire_date,
                    "xt_code": xt_code,
                    "updated_at": now,
                }
            )
            db_only_confirmed_delisted_codes.append(code)
        else:
            db_only_active_detail_codes.append(code)

    valid_new_docs = list(new_docs)
    missing_ipo_codes: list[str] = []
    baseline_docs: list[dict[str, Any]] = []
    if initialize_adjust_factor_baselines:
        valid_new_docs = []
        for doc in new_docs:
            ipo_date = parse_basic_date(doc.get("ipoDate"))
            if ipo_date is None:
                missing_ipo_codes.append(doc["code"])
                continue
            normalized = dict(doc)
            normalized["ipoDate"] = ipo_date
            valid_new_docs.append(normalized)
            baseline_docs.append(build_adjustment_factor_baseline(normalized["code"], ipo_date))

    names_by_code = {doc["code"]: doc.get("code_name", "") for doc in new_docs}
    failed_set = set(failed_codes)
    missing_ipo_set = set(missing_ipo_codes)
    confirmed_delisted_set = set(db_only_confirmed_delisted_codes)
    active_detail_set = set(db_only_active_detail_codes)
    missing_detail_set = set(db_only_detail_missing_codes)
    enriched_rows: list[dict[str, Any]] = []
    for row in diff_rows:
        enriched = dict(row)
        if row["diff_type"] == "xt_only":
            enriched["code_name"] = names_by_code.get(row["code"], "")
            if row["code"] in failed_set:
                enriched["reason"] = "xtquant_detail_missing"
            elif row["code"] in missing_ipo_set:
                enriched["reason"] = "xtquant_ipo_date_missing"
        elif row["code"] in confirmed_delisted_set:
            enriched["reason"] = "xtquant_expire_date_confirmed_delisted"
        elif row["code"] in active_detail_set:
            enriched["reason"] = "xtquant_detail_active_sector_missing"
        elif row["code"] in missing_detail_set:
            enriched["reason"] = "xtquant_detail_missing_preserved"
        enriched_rows.append(enriched)

    baseline_summary = AdjustmentFactorBaselineWriteSummary()
    if initialize_adjust_factor_baselines:
        baseline_summary = write_missing_adjustment_factor_baselines(
            cfg,
            baseline_docs,
            dry_run=dry_run,
        )
    if dry_run:
        rows_written = 0
    else:
        rows_written = upsert_records(
            cfg,
            BASIC_INFO_COLLECTION,
            valid_new_docs,
            key_columns=("code",),
            dry_run=False,
        )
    modified_rows = update_confirmed_delistings(
        cfg,
        confirmed_delisted_updates,
        dry_run=dry_run,
    )
    return BasicInfoSyncResult(
        tuple(valid_new_docs),
        tuple(failed_codes),
        tuple(enriched_rows),
        {
            "matched": len(confirmed_delisted_updates),
            "modified": modified_rows,
            "upserted": rows_written,
        },
        missing_ipo_codes=tuple(sorted(missing_ipo_codes)),
        adjust_factor_baseline_planned_count=baseline_summary.planned_count,
        adjust_factor_baseline_existing_count=baseline_summary.existing_count,
        adjust_factor_baseline_written_count=baseline_summary.written_count,
        detail_requested_codes=tuple(
            sorted(normalize_internal_code(code) for code in basic_detail_xt_codes)
        ),
        instrument_detail_requested_count=len(requested_xt_codes),
        db_only_confirmed_delisted_codes=tuple(sorted(db_only_confirmed_delisted_codes)),
        db_only_active_detail_codes=tuple(sorted(db_only_active_detail_codes)),
        db_only_detail_missing_codes=tuple(sorted(db_only_detail_missing_codes)),
        confirmed_delisted_updates=tuple(confirmed_delisted_updates),
        details_by_xt_code=details_by_xt_code,
    )
