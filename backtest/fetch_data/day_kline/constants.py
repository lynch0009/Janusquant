from __future__ import annotations

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DAY_COLLECTION = "A_stock_market_day_kline"
BASIC_INFO_COLLECTION = "A_stock_market_basic_info"
HS_A_SHARE_SECTOR_NAME = "沪深A股"
XT_DAY_FIELDS = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "preClose",
    "preclose",
    "suspendFlag",
)
MAX_DAILY_SYNC_BATCH_SIZE = 50_000
DEFAULT_MAX_FALLBACK_MISSING_STOCKS = 200
DEFAULT_MAX_FALLBACK_MISSING_DAYS = 2_000
FALLBACK_WINDOW_TRADE_DAYS = 20
XT_TIMEZONE_OFFSET_HOURS = 8
MIN_DAY_KLINE_START_DATE = datetime(2020, 1, 1)
REPORT_ROOT = PROJECT_ROOT / "backtest" / "runs" / "output" / "xtquant_daily_sync"
FIXED_DAY_KLINE_INDEX_CODES = (
    "sh.000001",
    "sh.000016",
    "sh.000300",
    "sh.000852",
    "sh.000903",
    "sh.000905",
    "sz.399001",
    "sz.399102",
    "sz.399303",
    "sz.399905",
)
FIXED_DAY_KLINE_INDEX_NAME = "固定日K指数"
