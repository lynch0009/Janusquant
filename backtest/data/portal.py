"""回测数据访问门面。

这一层负责把底层 Mongo 数据整理成回测引擎更容易消费的结构：
交易日列表、日线快照、分钟线批量结果、因子截面和历史序列等。
"""

from __future__ import annotations

from datetime import datetime, time
import re
from typing import Iterable, Sequence

import pandas as pd

from backtest.db import MongoRepository, normalize_code
from backtest.feature.feature import Feature


DAILY_PRICE_FIELDS = ("open", "high", "low", "close", "preclose")
DAILY_VALUE_FIELDS = (
    "liqaMV",
    "totalMV",
    "financePubDate",
)
DAY_KLINE_STANDARD_TO_RAW = {
    "code": "code",
    "trade_date": "date",
    "open": "o",
    "high": "h",
    "low": "l",
    "close": "c",
    "preclose": "prec",
    "volume": "v",
    "amount": "a",
    "turn": "turn",
    "pctChg": "pctChg",
    "tradestatus": "tradestatus",
    "isST": "isST",
}
DEFAULT_DAILY_HISTORY_FIELDS = (
    "code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "turn",
    "pctChg",
    "tradestatus",
    "isST",
    "liqaMV",
    "totalMV",
)
DIVIDEND_RAW_FIELDS = (
    "code",
    "dividCashStock",
    "dividOperateDate",
    "dividPayDate",
    "dividStocksPs",
    "dividReserveToStockPs",
    "dividCashPsBeforeTax",
)
STANDARDIZED_CORPORATE_ACTION_FIELDS = (
    "event_type",
    "code",
    "operate_date",
    "settle_date",
    "cash_dividend_per_share",
    "stock_dividend_ratio",
    "stock_dividend_share_ratio",
    "reserve_to_stock_ratio",
    "raw_text",
)


class MongoDataPortal:
    """
    面向回测引擎的只读数据门面。

    设计目标是把“集合名、字段压缩格式、分钟线分表”等存储细节
    都封装在这一层之下，让 execution/strategies 只关心业务语义。
    """

    def __init__(self, db_client, *, calendar_code: str = "sh.000001"):
        """初始化数据门面，并指定用于推导交易日历的基准指数代码。"""

        self.repo = MongoRepository(db_client)
        self.db = db_client.db if hasattr(db_client, "db") else db_client
        self.feature_service = Feature(db_client)
        self.calendar_code = normalize_code(calendar_code)
        self._feature_cache: dict[tuple[datetime, tuple[str, ...]], pd.DataFrame] = {}

    @staticmethod
    def _normalize_daily_history_fields(fields: Sequence[str] | None) -> list[str]:
        requested_fields = list(fields) if fields is not None else list(DEFAULT_DAILY_HISTORY_FIELDS)
        normalized_fields: list[str] = []
        for field in requested_fields:
            normalized_field = "trade_date" if field == "date" else field
            if normalized_field not in DAY_KLINE_STANDARD_TO_RAW and normalized_field not in DAILY_VALUE_FIELDS:
                raise ValueError(f"unsupported daily history field: {field}")
            if normalized_field not in normalized_fields:
                normalized_fields.append(normalized_field)
        if "code" not in normalized_fields:
            normalized_fields.insert(0, "code")
        if "trade_date" not in normalized_fields:
            normalized_fields.insert(1, "trade_date")
        return normalized_fields

    @staticmethod
    def _resolve_day_kline_fields(requested_fields: Sequence[str]) -> list[str]:
        raw_fields = {"code", "date"}
        if any(field in DAILY_PRICE_FIELDS for field in requested_fields):
            raw_fields.update(["o", "h", "l", "c", "prec"])
        for field in requested_fields:
            raw_field = DAY_KLINE_STANDARD_TO_RAW.get(field)
            if raw_field is not None:
                raw_fields.add(raw_field)
        ordered_fields = [
            "code",
            "date",
            "o",
            "h",
            "l",
            "c",
            "prec",
            "v",
            "a",
            "turn",
            "pctChg",
            "tradestatus",
            "isST",
        ]
        return [field for field in ordered_fields if field in raw_fields]

    def _load_feature_history_for_daily_fields(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        codes: Sequence[str] | None,
        requested_fields: Sequence[str],
    ) -> pd.DataFrame:
        value_fields = [field for field in requested_fields if field in DAILY_VALUE_FIELDS]
        if not value_fields:
            return pd.DataFrame()

        feature_fields = ["code", "date", *value_fields]
        feature_history = self.repo.feature.get_series_by_filters(
            start=start_date,
            end=end_date,
            codes=codes,
            fields=feature_fields,
        )
        if feature_history.empty:
            return feature_history
        return feature_history.rename(columns={"date": "trade_date"}).reset_index(drop=True)

    @staticmethod
    def _empty_corporate_action_frame() -> pd.DataFrame:
        """返回标准化公司行为事件的空表结构。"""

        return pd.DataFrame(columns=list(STANDARDIZED_CORPORATE_ACTION_FIELDS))

    def _standardize_dividend_events(self, dividend_frame: pd.DataFrame) -> pd.DataFrame:
        """把原始分红送转记录标准化成回测内部事件。

        一条原始记录最多拆成两条：
        1. 现金分红事件
        2. 送股/转增事件
        """

        if dividend_frame.empty:
            return self._empty_corporate_action_frame()

        frame = dividend_frame.copy()
        for field in ("dividOperateDate", "dividPayDate"):
            if field in frame.columns:
                frame[field] = pd.to_datetime(frame[field], errors="coerce")
        for field in ("dividStockMarketDate",):
            if field in frame.columns:
                frame[field] = pd.to_datetime(frame[field], errors="coerce")

        rows: list[dict[str, object]] = []
        for row in frame.itertuples(index=False):
            operate_date = getattr(row, "dividOperateDate", None)
            if pd.isna(operate_date) or operate_date is None:
                continue

            cash_dividend_per_share = float(getattr(row, "dividCashPsBeforeTax", 0.0) or 0.0)
            stock_dividend_share_ratio = float(getattr(row, "dividStocksPs", 0.0) or 0.0)
            reserve_to_stock_ratio = float(getattr(row, "dividReserveToStockPs", 0.0) or 0.0)
            stock_dividend_ratio = stock_dividend_share_ratio + reserve_to_stock_ratio
            raw_text = str(getattr(row, "dividCashStock", "") or "")
            pay_date = getattr(row, "dividPayDate", None)
            stock_market_date = getattr(row, "dividStockMarketDate", None)

            if cash_dividend_per_share > 0:
                # 只有存在现金分红时，才生成现金分红事件和对应到账日。
                rows.append(
                    {
                        "event_type": "cash_dividend",
                        "code": row.code,
                        "operate_date": pd.Timestamp(operate_date).to_pydatetime(),
                        "settle_date": pd.Timestamp(
                            pay_date if pay_date is not None and not pd.isna(pay_date) else operate_date
                        ).to_pydatetime(),
                        "cash_dividend_per_share": cash_dividend_per_share,
                        "stock_dividend_ratio": 0.0,
                        "stock_dividend_share_ratio": 0.0,
                        "reserve_to_stock_ratio": 0.0,
                        "raw_text": raw_text,
                    }
                )

            if stock_dividend_ratio > 0:
                # 送股和转增在账户层都表现为“新增股份”，这里统一成一个事件。
                rows.append(
                    {
                        "event_type": "stock_dividend",
                        "code": row.code,
                        "operate_date": pd.Timestamp(operate_date).to_pydatetime(),
                        "settle_date": pd.Timestamp(
                            stock_market_date
                            if stock_market_date is not None and not pd.isna(stock_market_date)
                            else operate_date
                        ).to_pydatetime(),
                        "cash_dividend_per_share": 0.0,
                        "stock_dividend_ratio": stock_dividend_ratio,
                        "stock_dividend_share_ratio": stock_dividend_share_ratio,
                        "reserve_to_stock_ratio": reserve_to_stock_ratio,
                        "raw_text": raw_text,
                    }
                )

        if not rows:
            return self._empty_corporate_action_frame()

        standardized = pd.DataFrame(rows)
        standardized["operate_date"] = pd.to_datetime(standardized["operate_date"])
        standardized["settle_date"] = pd.to_datetime(standardized["settle_date"])
        return standardized.sort_values(
            ["operate_date", "code", "event_type"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    def get_corporate_action_events(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        codes: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """读取区间内公司行为事件，并返回标准化结果。"""

        dividend_frame = self.repo.dividend.get_series(
            start=start_date,
            end=end_date,
            codes=codes,
            fields=list(DIVIDEND_RAW_FIELDS),
        )
        return self._standardize_dividend_events(dividend_frame)

    def get_corporate_action_event_slice(
        self,
        trade_date: datetime,
        *,
        codes: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """读取某个除权除息日对应的公司行为事件。"""

        dividend_frame = self.repo.dividend.get_operate_slice(
            trade_date=trade_date,
            codes=codes,
            fields=list(DIVIDEND_RAW_FIELDS),
        )
        return self._standardize_dividend_events(dividend_frame)

    def get_trade_calendar(self, start_date: datetime, end_date: datetime) -> list[datetime]:
        """
        读取交易日历。

        优先使用 `calendar_code` 对应指数的有效交易日；若指数数据不足，
        再退化为整个日线库里标记为可交易的日期集合。
        """

        dates = self.repo.day_kline.collection.distinct(
            "date",
            {
                "code": self.calendar_code,
                "date": {"$gte": start_date, "$lte": end_date},
                "tradestatus": True,
            },
        )
        if len(dates) < 2:
            dates = self.repo.day_kline.collection.distinct(
                "date",
                {
                    "date": {"$gte": start_date, "$lte": end_date},
                    "tradestatus": True,
                },
        )
        return sorted(pd.to_datetime(dates).to_pydatetime().tolist())

    def get_feature_slice(
        self,
        trade_date: datetime,
        *,
        fields: Sequence[str] | None = None,
        filters: dict | None = None,
    ) -> pd.DataFrame:
        """
        读取某个交易日的特征截面。

        未带过滤条件时会走简单内存缓存，避免回测主循环每天重复访问 Mongo。
        若传入 filters，则直接重新查询，保证筛选条件实时生效。
        """

        requested_fields = tuple(sorted(set((fields or []) + ["code", "date"])))
        cache_key = (pd.Timestamp(trade_date).to_pydatetime(), requested_fields)
        if cache_key not in self._feature_cache:
            self._feature_cache[cache_key] = self.repo.feature.get_cross_section(
                trade_date,
                fields=list(requested_fields),
                filters=filters,
            )
        frame = self._feature_cache[cache_key]
        if filters:
            # Cache stores the base frame for the requested fields; filters are applied on copy.
            query_frame = self.repo.feature.get_cross_section(
                trade_date,
                fields=list(requested_fields),
                filters=filters,
            )
            return query_frame.copy()
        return frame.copy()

    def get_feature_history(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        fields: Sequence[str] | None = None,
        filters: dict | None = None,
    ) -> pd.DataFrame:
        """按日期区间读取特征历史序列，常用于策略预处理。"""

        requested_fields = list(sorted(set((fields or []) + ["code", "date"])))
        return self.repo.feature.get_series_by_filters(
            start=start_date,
            end=end_date,
            filters=filters,
            fields=requested_fields,
        )

    def get_visible_finance_slice(
        self,
        trade_date: datetime,
        *,
        codes: Sequence[str],
        fields: Sequence[str] | None = None,
        start_pub_date: datetime | None = None,
    ) -> pd.DataFrame:
        """读取截至指定交易日可见的财报记录。"""

        requested_fields = list(sorted(set((fields or []) + ["code", "pubDate", "statDate"])))
        return self.repo.finance.get_visible_reports(
            codes=codes,
            as_of=trade_date,
            start_pub_date=start_pub_date,
            fields=requested_fields,
        )

    def get_finance_reports(
        self,
        *,
        codes: Sequence[str],
        start_pub_date: datetime | None = None,
        end_pub_date: datetime | None = None,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """按财报公告日区间读取财报记录。"""

        requested_fields = list(sorted(set((fields or []) + ["code", "pubDate", "statDate"])))
        return self.repo.finance.get_reports(
            codes=codes,
            start_pub_date=start_pub_date,
            end_pub_date=end_pub_date,
            fields=requested_fields,
        )

    def get_minervini_fundamental_features(
        self,
        *,
        codes: Sequence[str],
        start_pub_date: datetime | None = None,
        end_pub_date: datetime | None = None,
        fields: Sequence[str] | None = None,
        feature_version: str = "minervini_fundamental_v1",
    ) -> pd.DataFrame:
        """按公告日区间读取 Minervini 基本面特征时间线。"""

        requested_fields = list(sorted(set((fields or []) + ["code", "pubDate", "statDate", "featureVersion"])))
        return self.repo.minervini_fundamental_feature.get_features(
            codes=codes,
            start_pub_date=start_pub_date,
            end_pub_date=end_pub_date,
            fields=requested_fields,
            feature_version=feature_version,
        )

    def get_daily_close_map(self, codes: Sequence[str], trade_date: datetime) -> dict[str, float]:
        """读取一组股票截至指定交易日的最新收盘价映射。"""

        if not codes:
            return {}
        latest = self.repo.day_kline.get_latest_bars(codes, trade_date=trade_date, standardize=True)
        if latest.empty:
            return {}
        return latest.set_index("code")["close"].to_dict()

    def get_market_amount_history(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        code_prefixes: Sequence[str] | None = None,
        include_stopped: bool = False,
    ) -> pd.DataFrame:
        """按交易日聚合全市场成交额。

        默认只统计 A 股主板 / 创业板 / 科创板常见代码前缀，
        用于构建市场整体流动性开关。
        """

        normalized_prefixes = tuple(code_prefixes or ("sh.60", "sh.68", "sz.00", "sz.30"))
        if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
            return pd.DataFrame(columns=["trade_date", "market_amount", "security_count"])

        match_stage: dict[str, object] = {
            "date": {"$gte": start_date, "$lt": end_date},
            "$or": [{"code": {"$regex": f"^{re.escape(prefix)}"}} for prefix in normalized_prefixes],
        }
        if not include_stopped:
            match_stage["tradestatus"] = True

        pipeline = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": "$date",
                    "market_amount": {"$sum": "$a"},
                    "security_count": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        records = list(self.repo.day_kline.collection.aggregate(pipeline))
        if not records:
            return pd.DataFrame(columns=["trade_date", "market_amount", "security_count"])

        frame = pd.DataFrame.from_records(records).rename(columns={"_id": "trade_date"})
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        return frame[["trade_date", "market_amount", "security_count"]].reset_index(drop=True)

    def get_daily_history(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        codes: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        include_stopped: bool = False,
        batch_size: int | None = None,
        price_mode: str = "hfq",
    ) -> pd.DataFrame:
        """
        读取一段时间内的日线历史。

        当股票数较多时可按 batch_size 分批查询，避免单次 Mongo 查询过大。
        返回值会统一整理成标准化列名，便于策略直接做 pandas 计算。
        """

        normalized_price_mode = self.feature_service.normalize_price_mode(price_mode)
        requested_fields = self._normalize_daily_history_fields(fields)

        if codes and batch_size and len(codes) > batch_size:
            frames: list[pd.DataFrame] = []
            normalized_codes = [normalize_code(code) for code in codes]
            for offset in range(0, len(normalized_codes), batch_size):
                batch_codes = normalized_codes[offset : offset + batch_size]
                batch_frame = self.get_daily_history(
                    start_date,
                    end_date,
                    codes=batch_codes,
                    fields=fields,
                    include_stopped=include_stopped,
                    batch_size=None,
                    price_mode=normalized_price_mode,
                )
                if not batch_frame.empty:
                    frames.append(batch_frame)
            if not frames:
                return pd.DataFrame()
            return (
                pd.concat(frames, ignore_index=True)
                .sort_values(["code", "trade_date"])
                .reset_index(drop=True)
            )

        query: dict = {
            "date": {"$gte": start_date, "$lt": end_date},
        }
        if codes:
            query["code"] = {"$in": [normalize_code(code) for code in codes]}
        if not include_stopped:
            query["tradestatus"] = True

        if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
            return pd.DataFrame(columns=requested_fields)

        day_kline_fields = self._resolve_day_kline_fields(requested_fields)
        inclusive_end_date = (pd.Timestamp(end_date) - pd.Timedelta(days=1)).to_pydatetime()
        frame = self.repo.day_kline.get_bars(
            codes=[normalize_code(code) for code in codes] if codes else sorted(self.repo.day_kline.collection.distinct("code", query)),
            start=start_date,
            end=inclusive_end_date,
            fields=day_kline_fields,
            include_stopped=include_stopped,
            standardize=True,
        )
        if frame.empty:
            return frame

        frame = frame.rename(columns={"trade_time": "trade_date"}).reset_index(drop=True)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])

        if normalized_price_mode != "raw" and any(field in DAILY_PRICE_FIELDS for field in requested_fields):
            frame = self.feature_service.apply_price_mode(frame, price_mode=normalized_price_mode)

        feature_history = self._load_feature_history_for_daily_fields(
            start_date,
            end_date,
            codes=codes,
            requested_fields=requested_fields,
        )
        if not feature_history.empty:
            feature_history["trade_date"] = pd.to_datetime(feature_history["trade_date"])
            frame = frame.merge(feature_history, on=["code", "trade_date"], how="left")

        available_fields = [field for field in requested_fields if field in frame.columns]
        return frame[available_fields].reset_index(drop=True)

    def get_daily_bar_snapshot(
        self,
        codes: Sequence[str],
        trade_date: datetime,
    ) -> dict[str, pd.DataFrame]:
        """
        读取某个交易日的日线快照。

        返回值按 code 分组成字典，方便执行器直接按股票取值。
        即使某只股票没有数据，也会返回空 DataFrame，减少上层判空分支。
        """

        if not codes:
            return {}

        normalized_codes = [normalize_code(code) for code in codes]
        cursor = self.repo.day_kline.collection.find(
            {
                "code": {"$in": normalized_codes},
                "date": pd.Timestamp(trade_date).to_pydatetime(),
                "tradestatus": True,
            },
            {
                "_id": 0,
                "code": 1,
                "date": 1,
                "o": 1,
                "h": 1,
                "l": 1,
                "c": 1,
                "prec": 1,
                "v": 1,
                "a": 1,
                "turn": 1,
                "pctChg": 1,
                "isST": 1,
            },
        )
        frame = pd.DataFrame.from_records(list(cursor))
        if frame.empty:
            return {normalize_code(code): pd.DataFrame() for code in normalized_codes}

        frame = frame.rename(
            columns={
                "date": "dt",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "prec": "preclose",
                "v": "volume",
                "a": "amount",
            }
        )
        frame["dt"] = pd.to_datetime(frame["dt"])

        snapshot: dict[str, pd.DataFrame] = {normalize_code(code): pd.DataFrame() for code in normalized_codes}
        for code, group in frame.groupby("code"):
            snapshot[normalize_code(code)] = group.sort_values("dt").reset_index(drop=True)
        return snapshot

    def get_minute_bars_batch(
        self,
        codes: Sequence[str],
        trade_date: datetime,
        *,
        start_time: time,
        end_time: time,
    ) -> dict[str, pd.DataFrame]:
        """
        批量读取指定交易日内的分钟线窗口。

        这一接口通常被执行器和风控模块使用，用来拿入场窗口、出场窗口
        或风控观察窗口内的分钟行情。
        """

        if not codes:
            return {}
        start_dt = datetime.combine(trade_date.date(), start_time)
        end_dt = datetime.combine(trade_date.date(), end_time)
        result = self.repo.minute_kline.get_bars_batch(
            codes,
            start=start_dt,
            end=end_dt,
            standardize=True,
        )
        normalized: dict[str, pd.DataFrame] = {}
        for code, frame in result.items():
            if frame.empty:
                normalized[normalize_code(code)] = frame
                continue
            frame = frame.rename(columns={"trade_time": "dt"}).sort_values("dt").reset_index(drop=True)
            normalized[normalize_code(code)] = frame
        return normalized

    def get_stock_basic(
        self,
        codes: Sequence[str],
        *,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame()
        return self.repo.stock_basic.get_basic_info(codes, fields=fields)
