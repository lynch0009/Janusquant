from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.db import DuckDBConfig, upsert_frame
from backtest.fetch_data.baostock_utils import BaostockQueryError, fetch_query_dataframe, login_with_retry, safe_logout
from backtest.fetch_data.stock_universe import load_stock_windows_duckdb
from backtest.utils import to_pydatetime
from backtest.utils.log import format_date, log


BASIC_INFO_COLLECTION = "A_stock_market_basic_info"
ADJUST_FACTOR_COLLECTION = "A_stock_market_adjust_factor"
def parse_fetch_date(value) -> datetime | None:
    value = to_pydatetime(value)
    if value in (None, "") or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return datetime(parsed.year, parsed.month, parsed.day)


class AdjustFactorFetch:
    def __init__(self, db_client=None):
        self.db_client = db_client if isinstance(db_client, DuckDBConfig) else DuckDBConfig()

    def update_adjust_factor(self, start_date, end_date=None):
        import baostock as bs
        start_dt = parse_fetch_date(start_date)
        end_dt = parse_fetch_date(end_date) or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if start_dt is None:
            raise ValueError("start_date is required")
        if start_dt > end_dt:
            raise ValueError("start_date cannot be greater than end_date")
        stocks = load_stock_windows_duckdb(self.db_client, start_date=start_dt, end_date=end_dt)
        login_with_retry()
        docs: list[dict] = []
        count = 0
        try:
            for index, stock in enumerate(stocks, start=1):
                code = stock.code
                stock_start_date = max(start_dt, stock.ipo_date) if stock.ipo_date is not None else start_dt
                query_start_date = format_date(stock_start_date)
                try:
                    result_factor, _retries_used = fetch_query_dataframe(
                        bs.query_adjust_factor,
                        code=code,
                        start_date=query_start_date,
                        context=f"{code} query_adjust_factor from {query_start_date}",
                    )
                except BaostockQueryError as exc:
                    log.error(f"{code} 复权因子读取失败: {exc}")
                    continue

                if len(result_factor) > 0:
                    result_factor = result_factor.astype({
                        "code": "str",  # 强制转为字符串
                        "foreAdjustFactor": "float64",
                        "backAdjustFactor": "float64",
                        "adjustFactor": "float64",
                    })
                    result_factor['dividOperateDate'] = pd.to_datetime(result_factor['dividOperateDate'], format='%Y-%m-%d')
                    result_factor = result_factor[
                        result_factor["dividOperateDate"] >= stock_start_date
                    ].copy()
                    result_factor.drop("adjustFactor", axis=1, inplace=True)
                    result_factor.rename(columns={
                        "foreAdjustFactor": "qfq_fac",
                        "backAdjustFactor": "hfq_fac",
                        "dividOperateDate": "date"
                    }, inplace=True)
                    if not result_factor.empty:
                        count += 1
                        for row in result_factor.to_dict("records"):
                            row["date"] = datetime(row["date"].year, row["date"].month, row["date"].day)
                            docs.append(row)
                        if count % 100 == 0:
                            log.info(f"已读取{count}只股票复权因子，当前进度 {index}/{len(stocks)}")
        finally:
            safe_logout()

        if not docs:
            log.info("没有读取到复权因子数据")
            return 0

        upsert_frame(self.db_client, ADJUST_FACTOR_COLLECTION, docs, key_columns=("code", "date"))
        affected = len(docs)
        log.info(f"已读取{count}只股票复权因子 成功写入/匹配 {affected} 条数据")
        return affected


if __name__ == '__main__':
    fetch = AdjustFactorFetch(db_client=DuckDBConfig())
    fetch.update_adjust_factor(start_date="2000-01-01")
