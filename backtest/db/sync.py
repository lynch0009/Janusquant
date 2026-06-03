from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
from pymongo.errors import BulkWriteError
from xtquant import xtdata

from backtest.utils import to_pydatetime

from .precision import (
    normalize_amount_series,
    normalize_price_series,
    normalize_volume_series,
)
from .repository import minute_collection_name, normalize_code

MAX_MONGO_BATCH_SIZE = 50_000


def to_xt_code(normalized: str) -> str:
    exchange, stock_code = normalized.split(".")
    return f"{stock_code}.{exchange.upper()}"


def load_stock_codes_from_file(file_path: str | Path) -> list[str]:
    path = Path(file_path)
    content: str | None = None
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            content = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if content is None:
        content = path.read_text(encoding="utf-8", errors="ignore")

    stock_codes: list[str] = []
    for line in content.splitlines():
        value = line.strip()
        if not value:
            continue
        stock_codes.append(normalize_code(value))
    return stock_codes


@dataclass(frozen=True)
class MinuteSyncResult:
    stock_code: str
    xt_code: str
    collection_name: str
    inserted_count: int
    fetched_rows: int
    skipped: bool = False
    reason: str = ""


class MinuteKlineSyncService:
    default_fields = ["time", "open", "high", "low", "close", "volume", "amount"]

    def __init__(
        self,
        db_client,
        *,
        timezone_offset_hours: int = 8,
        batch_size: int = 20_000,
    ):
        self.db_client = db_client
        self.db = db_client.db if hasattr(db_client, "db") else db_client
        self.timezone_offset_hours = timezone_offset_hours
        self.batch_size = max(1, min(int(batch_size), MAX_MONGO_BATCH_SIZE))

    def fetch_xt_local_data(
        self,
        xt_code: str,
        *,
        period: str,
        start_time: str,
        end_time: str,
        fill_data: bool = True,
        download: bool = True,
    ) -> pd.DataFrame:
        if download:
            xtdata.download_history_data(
                xt_code,
                period=period,
                start_time=start_time,
                end_time=end_time,
            )

        data_dict = xtdata.get_local_data(
            field_list=self.default_fields,
            stock_list=[xt_code],
            period=period,
            start_time=start_time,
            end_time=end_time,
            dividend_type="none",
            fill_data=fill_data,
        )
        frame = data_dict.get(xt_code)
        if not isinstance(frame, pd.DataFrame):
            return pd.DataFrame()
        return frame

    def _frame_to_documents(self, frame: pd.DataFrame) -> list[dict]:
        if frame.empty:
            return []

        working = frame.copy()
        working["time"] = pd.to_datetime(working["time"].astype(float), unit="ms") + pd.Timedelta(
            hours=self.timezone_offset_hours
        )
        working["o"] = normalize_price_series(working["open"])
        working["h"] = normalize_price_series(working["high"])
        working["l"] = normalize_price_series(working["low"])
        working["c"] = normalize_price_series(working["close"])
        working["v"] = normalize_volume_series(working["volume"])
        working["a"] = normalize_amount_series(working["amount"])
        working["dt"] = working["time"].map(to_pydatetime)
        working = working.dropna(subset=["o", "h", "l", "c", "v", "a", "dt"])
        if working.empty:
            return []
        documents = working[["o", "h", "l", "c", "v", "a", "dt"]].to_dict("records")
        for doc in documents:
            doc["v"] = int(doc["v"])
            doc["a"] = int(doc["a"])
        return documents

    def sync_stock(
        self,
        stock_code: str,
        *,
        start_time: str,
        end_time: str,
        period: str = "1m",
        fill_data: bool = True,
        download: bool = True,
    ) -> MinuteSyncResult:
        normalized_code = normalize_code(stock_code)
        xt_code = to_xt_code(normalized_code)
        collection_name = minute_collection_name(normalized_code)

        frame = self.fetch_xt_local_data(
            xt_code,
            period=period,
            start_time=start_time,
            end_time=end_time,
            fill_data=fill_data,
            download=download,
        )
        if frame.empty:
            return MinuteSyncResult(
                stock_code=normalized_code,
                xt_code=xt_code,
                collection_name=collection_name,
                inserted_count=0,
                fetched_rows=0,
                skipped=True,
                reason="no_data",
            )

        documents = self._frame_to_documents(frame)
        if not documents:
            return MinuteSyncResult(
                stock_code=normalized_code,
                xt_code=xt_code,
                collection_name=collection_name,
                inserted_count=0,
                fetched_rows=len(frame),
                skipped=True,
                reason="no_valid_documents",
            )

        collection = self.db[collection_name]
        collection.create_index([("dt", 1)], unique=True)

        inserted_count = 0
        for start_idx in range(0, len(documents), self.batch_size):
            batch = documents[start_idx : start_idx + self.batch_size]
            try:
                result = collection.insert_many(batch, ordered=False)
                inserted_count += len(result.inserted_ids)
            except BulkWriteError as exc:
                inserted_count += exc.details.get("nInserted", 0)

        return MinuteSyncResult(
            stock_code=normalized_code,
            xt_code=xt_code,
            collection_name=collection_name,
            inserted_count=inserted_count,
            fetched_rows=len(frame),
        )

    def sync_stocks(
        self,
        stock_codes: Sequence[str],
        *,
        start_time: str,
        end_time: str,
        period: str = "1m",
        fill_data: bool = True,
        download: bool = True,
    ) -> list[MinuteSyncResult]:
        results: list[MinuteSyncResult] = []
        for stock_code in stock_codes:
            results.append(
                self.sync_stock(
                    stock_code,
                    start_time=start_time,
                    end_time=end_time,
                    period=period,
                    fill_data=fill_data,
                    download=download,
                )
            )
        return results

    def get_latest_dt(self, stock_code: str) -> datetime | None:
        normalized_code = normalize_code(stock_code)
        collection = self.db[minute_collection_name(normalized_code)]
        doc = collection.find_one(sort=[("dt", -1)], projection={"_id": 0, "dt": 1})
        if not doc:
            return None
        return doc.get("dt")


@dataclass(frozen=True)
class MinuteCoverageGap:
    stock_code: str
    missing_dates: tuple[datetime, ...]


@dataclass(frozen=True)
class MinuteSyncPlanItem:
    stock_code: str
    start_time: str
    end_time: str
    missing_dates: tuple[datetime, ...]


class MinuteCoverageInspector:
    def __init__(self, db_client):
        self.db_client = db_client
        self.db = db_client.db if hasattr(db_client, "db") else db_client
        self.day_collection = self.db["A_stock_market_day_kline"]

    def get_expected_trade_dates(
        self,
        stock_code: str,
        *,
        start_date: datetime,
        end_date: datetime,
    ) -> list[datetime]:
        normalized_code = normalize_code(stock_code)
        dates = self.day_collection.distinct(
            "date",
            {
                "code": normalized_code,
                "date": {"$gte": start_date, "$lte": end_date},
                "tradestatus": True,
            },
        )
        return sorted(pd.to_datetime(dates).normalize().to_pydatetime().tolist())

    def get_existing_minute_dates(
        self,
        stock_code: str,
        *,
        start_date: datetime,
        end_date: datetime,
    ) -> list[datetime]:
        normalized_code = normalize_code(stock_code)
        collection = self.db[minute_collection_name(normalized_code)]
        pipeline = [
            {
                "$match": {
                    "dt": {
                        "$gte": start_date,
                        "$lt": end_date + pd.Timedelta(days=1),
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$dt",
                            "timezone": "Asia/Shanghai",
                        }
                    }
                }
            },
            {"$sort": {"_id": 1}},
        ]
        values = [item["_id"] for item in collection.aggregate(pipeline)]
        if not values:
            return []
        return sorted(pd.to_datetime(values).normalize().to_pydatetime().tolist())

    def find_missing_dates(
        self,
        stock_code: str,
        *,
        start_date: datetime,
        end_date: datetime,
    ) -> MinuteCoverageGap:
        expected_dates = self.get_expected_trade_dates(
            stock_code,
            start_date=start_date,
            end_date=end_date,
        )
        existing_dates = set(
            self.get_existing_minute_dates(
                stock_code,
                start_date=start_date,
                end_date=end_date,
            )
        )
        missing_dates = tuple(date for date in expected_dates if date not in existing_dates)
        return MinuteCoverageGap(stock_code=normalize_code(stock_code), missing_dates=missing_dates)

    def build_sync_plan(
        self,
        stock_code: str,
        *,
        start_date: datetime,
        end_date: datetime,
    ) -> list[MinuteSyncPlanItem]:
        gap = self.find_missing_dates(
            stock_code,
            start_date=start_date,
            end_date=end_date,
        )
        if not gap.missing_dates:
            return []

        missing_dates = list(gap.missing_dates)
        plan: list[MinuteSyncPlanItem] = []
        current_group: list[datetime] = [missing_dates[0]]

        for date in missing_dates[1:]:
            previous = current_group[-1]
            if (pd.Timestamp(date) - pd.Timestamp(previous)).days <= 3:
                current_group.append(date)
            else:
                plan.append(self._build_plan_item(gap.stock_code, current_group))
                current_group = [date]

        if current_group:
            plan.append(self._build_plan_item(gap.stock_code, current_group))
        return plan

    @staticmethod
    def _build_plan_item(stock_code: str, dates: list[datetime]) -> MinuteSyncPlanItem:
        return MinuteSyncPlanItem(
            stock_code=stock_code,
            start_time=pd.Timestamp(dates[0]).strftime("%Y%m%d"),
            end_time=pd.Timestamp(dates[-1]).strftime("%Y%m%d"),
            missing_dates=tuple(dates),
        )
