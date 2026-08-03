from __future__ import annotations

r"""日 K 同步统一入口。

手工补数参数：

    --source       必填，xtquant / baostock / local 三选一。
    --stocks       必填，逗号分隔；支持 600000.SH、sh.600000、600000。
    --start-date   必填，支持 2026-05-01 或 20260501。
    --end-date     必填，支持 2026-05-31 或 20260531。
    --local-root   仅 source=local 使用，目录下按 YYYYMMDD.csv 找文件。
    --batch-size   DuckDB 批量写入大小，默认 2000。
    --xt-batch-size xtquant 每批股票数，默认 300，仅 source=xtquant 有意义。
    --dry-run      只规划不写库，适合先试命令。
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.db import DuckDBConfig
from backtest.fetch_data.day_kline.constants import (
    DEFAULT_LOCAL_ROOT,
    DEFAULT_MAX_FALLBACK_MISSING_DAYS,
    DEFAULT_MAX_FALLBACK_MISSING_STOCKS,
)
from backtest.fetch_data.day_kline.manual_sync import run_manual_day_kline_sync


def run_xtquant_daily_kline_sync(
    *,
    xtdata_client=None,
    cfg: DuckDBConfig | None = None,
    batch_size: int = 2000,
    xt_batch_size: int = 300,
    max_fallback_missing_stocks: int = DEFAULT_MAX_FALLBACK_MISSING_STOCKS,
    max_fallback_missing_days: int = DEFAULT_MAX_FALLBACK_MISSING_DAYS,
) -> dict:
    from backtest.fetch_data.day_kline.daily_sync import run_xtquant_daily_kline_sync as run_daily

    return run_daily(
        xtdata_client=xtdata_client,
        cfg=cfg,
        batch_size=batch_size,
        xt_batch_size=xt_batch_size,
        max_fallback_missing_stocks=max_fallback_missing_stocks,
        max_fallback_missing_days=max_fallback_missing_days,
    )


def parse_cli_date(value: str | None) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise argparse.ArgumentTypeError("date is required")
    try:
        dt = pd.to_datetime(text, errors="raise")
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"invalid date: {value}") from exc
    return datetime(dt.year, dt.month, dt.day)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync DuckDB daily K-line data.")
    subparsers = parser.add_subparsers(dest="command")

    daily = subparsers.add_parser("daily", help="Run scheduled xtquant daily sync")
    daily.add_argument("--batch-size", type=int, default=2000)
    daily.add_argument("--xt-batch-size", type=int, default=300)
    daily.add_argument("--max-fallback-missing-stocks", type=int, default=200)
    daily.add_argument("--max-fallback-missing-days", type=int, default=2000)

    fetch = subparsers.add_parser("fetch", help="Explicitly fetch selected daily K-line data")
    fetch.add_argument("--source", required=True, choices=("xtquant", "baostock", "local"))
    fetch.add_argument("--stocks", required=True, help="Comma separated stock codes")
    fetch.add_argument("--start-date", required=True, type=parse_cli_date, help="YYYYMMDD or YYYY-MM-DD")
    fetch.add_argument("--end-date", required=True, type=parse_cli_date, help="YYYYMMDD or YYYY-MM-DD")
    fetch.add_argument("--local-root", default=str(DEFAULT_LOCAL_ROOT), help="Local daily K CSV root directory")
    fetch.add_argument("--batch-size", type=int, default=2000)
    fetch.add_argument("--xt-batch-size", type=int, default=300)
    fetch.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict | None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command in (None, "daily"):
        return run_xtquant_daily_kline_sync(
            batch_size=getattr(args, "batch_size", 2000),
            xt_batch_size=getattr(args, "xt_batch_size", 300),
            max_fallback_missing_stocks=getattr(args, "max_fallback_missing_stocks", 200),
            max_fallback_missing_days=getattr(args, "max_fallback_missing_days", 2000),
        )
    if args.command == "fetch":
        return run_manual_day_kline_sync(
            source=args.source,
            stocks=args.stocks,
            start_date=args.start_date,
            end_date=args.end_date,
            local_root=args.local_root,
            batch_size=args.batch_size,
            xt_batch_size=args.xt_batch_size,
            dry_run=args.dry_run,
        )
    parser.error(f"unknown command: {args.command}")
    return None


if __name__ == "__main__":
    main()
