from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .constants import REPORT_ROOT
from .models import BasicInfoSyncResult


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, Path):
        return str(value)
    return str(value)


def should_write_report(
    basic_result: BasicInfoSyncResult,
    missing_by_code: dict[str, list[datetime]],
    day_kline_missing_codes_today: Sequence[str],
    historical_missing_rows: Sequence[dict[str, str]] | None = None,
) -> bool:
    return bool(
        basic_result.stock_pool_diff_rows
        or basic_result.inserted_docs
        or basic_result.detail_failed_codes
        or missing_by_code
        or day_kline_missing_codes_today
        or historical_missing_rows
    )


def write_daily_sync_report(
    basic_result: BasicInfoSyncResult,
    summary: dict[str, Any],
    *,
    now: datetime,
    historical_missing_rows: Sequence[dict[str, str]] | None = None,
    report_root: Path = REPORT_ROOT,
) -> Path:
    report_dir = report_root / now.strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)

    diff_path = report_dir / "stock_pool_diff.csv"
    with diff_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["diff_type", "code", "xt_code", "code_name", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in basic_result.stock_pool_diff_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    if historical_missing_rows:
        historical_path = report_dir / "historical_missing_detail.csv"
        with historical_path.open("w", newline="", encoding="utf-8-sig") as handle:
            fieldnames = ["code", "date", "reason"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in historical_missing_rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
        summary["historical_missing_detail_path"] = str(historical_path)

    summary["report_dir"] = str(report_dir)
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=json_default)
    return report_dir
