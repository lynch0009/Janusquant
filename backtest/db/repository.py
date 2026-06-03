from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

import pandas as pd
from pymongo import ASCENDING, DESCENDING

from backtest.utils import records_to_frame, sort_frame, to_pydatetime


KLINE_RENAME_MAP = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "a": "amount",
    "prec": "preclose",
}


def normalize_code(code: str) -> str:
    """Normalize a stock code to the internal Mongo format: sh.600000."""
    if not code:
        raise ValueError("stock code cannot be empty")

    code = code.strip()
    if "." not in code:
        raise ValueError(f"unsupported stock code format: {code}")

    left, right = code.split(".", 1)
    if left.isalpha():
        return f"{left.lower()}.{right}"
    return f"{right.lower()}.{left}"


def minute_collection_name(code: str) -> str:
    normalized = normalize_code(code)
    exchange, stock_code = normalized.split(".")
    return f"{exchange}_{stock_code}_minute_kline"


def _to_datetime(value: datetime | str | pd.Timestamp | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        return pd.to_datetime(value).to_pydatetime()
    normalized = to_pydatetime(value)
    if isinstance(normalized, datetime):
        return normalized
    return pd.to_datetime(normalized).to_pydatetime()


def _projection(fields: Sequence[str] | None) -> dict[str, int] | None:
    if not fields:
        return None
    projection = {field: 1 for field in fields}
    projection["_id"] = 0
    return projection


def _ensure_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _normalize_codes(codes: Sequence[str] | None) -> list[str]:
    if not codes:
        return []
    return [normalize_code(code) for code in codes]


def _date_range_query(
    field: str,
    start: datetime | str | pd.Timestamp,
    end: datetime | str | pd.Timestamp,
    *,
    inclusive_end: bool = True,
) -> dict[str, datetime]:
    start_dt = _to_datetime(start)
    end_dt = _to_datetime(end)
    if start_dt is None or end_dt is None:
        raise ValueError(f"{field} range requires both start and end")
    upper = end_dt + timedelta(days=1) if inclusive_end else end_dt
    return {"$gte": start_dt, "$lt": upper}


def _apply_datetime_columns(frame: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    for field in fields:
        if field in frame.columns:
            frame[field] = pd.to_datetime(frame[field])
    return frame


def _find_frame(
    collection,
    query: dict[str, Any],
    *,
    fields: Sequence[str] | None = None,
    sort: list[tuple[str, int]] | tuple[str, int] | None = None,
    limit: int | None = None,
    datetime_fields: Sequence[str] = (),
) -> pd.DataFrame:
    cursor = collection.find(query, _projection(fields))
    if sort is not None:
        normalized_sort = [sort] if isinstance(sort, tuple) else sort
        cursor = cursor.sort(normalized_sort)
    if limit is not None:
        cursor = cursor.limit(limit)
    frame = records_to_frame(cursor)
    return _apply_datetime_columns(frame, datetime_fields)


def _standardize_kline_frame(
    df: pd.DataFrame,
    *,
    time_field: str,
    keep_code: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df

    rename_map = dict(KLINE_RENAME_MAP)
    rename_map[time_field] = "trade_time"
    result = df.rename(columns=rename_map).copy()

    if "trade_time" in result.columns:
        result["trade_time"] = pd.to_datetime(result["trade_time"])

    ordered_columns = [
        column
        for column in [
            "code" if keep_code else None,
            "trade_time",
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
        ]
        if column and column in result.columns
    ]
    other_columns = [column for column in result.columns if column not in ordered_columns]
    return sort_frame(result[ordered_columns + other_columns], sort_field="trade_time")


@dataclass(frozen=True)
class IndexDefinition:
    collection: str
    keys: tuple[tuple[str, int], ...]
    unique: bool = False

    @property
    def name(self) -> str:
        """为推荐索引生成稳定、可读的名称，避免与历史自动命名撞名。"""

        key_part = "__".join(f"{field}_{direction}" for field, direction in self.keys)
        return f"idx__{self.collection}__{key_part}"


class MongoIndexManager:
    """Central place to describe the research-oriented indexes we expect."""

    def __init__(self, db):
        self.db = db

    @staticmethod
    def recommended_definitions() -> list[IndexDefinition]:
        """返回当前回测链路最关键的推荐索引。

        设计原则：
        - 以代码里的真实查询条件顺序为准，而不是以现网历史索引命名为准。
        - 如果现网已有“更长但前缀匹配”的索引，也允许把它作为标准答案，
          避免重复创建语义重叠的索引。
        """
        return [
            IndexDefinition(
                collection="A_stock_market_day_kline",
                keys=(("code", ASCENDING), ("date", ASCENDING)),
            ),
            IndexDefinition(
                collection="A_stock_market_day_kline",
                keys=(("code", ASCENDING), ("date", DESCENDING)),
            ),
            IndexDefinition(
                collection="A_stock_market_adjust_factor",
                keys=(("code", ASCENDING), ("date", ASCENDING)),
                unique=True,
            ),
            IndexDefinition(
                collection="A_stock_market_feature",
                keys=(("date", ASCENDING), ("code", ASCENDING)),
            ),
            IndexDefinition(
                collection="A_stock_market_feature",
                keys=(("date", ASCENDING), ("liqaMV", ASCENDING)),
            ),
            IndexDefinition(
                collection="A_stock_market_finance_data",
                keys=(("code", ASCENDING), ("pubDate", ASCENDING), ("statDate", ASCENDING)),
                unique=True,
            ),
            IndexDefinition(
                collection="A_stock_market_dividend_data",
                keys=(("code", ASCENDING), ("dividOperateDate", ASCENDING)),
            ),
            IndexDefinition(
                collection="A_stock_market_dividend_data",
                keys=(("code", ASCENDING), ("dividRegistDate", ASCENDING)),
            ),
            IndexDefinition(
                collection="A_stock_market_dividend_data",
                keys=(("code", ASCENDING), ("dividPayDate", ASCENDING)),
            )
        ]

    def current_indexes(self, collection: str) -> list[dict[str, Any]]:
        return list(self.db[collection].list_indexes())

    @staticmethod
    def _index_keys(index: dict[str, Any]) -> tuple[tuple[str, int], ...]:
        return tuple((field, int(direction)) for field, direction in index["key"].items())

    def ensure_recommended_indexes(self) -> list[str]:
        created: list[str] = []
        for definition in self.recommended_definitions():
            current_indexes = self.current_indexes(definition.collection)
            matched = next(
                (
                    index
                    for index in current_indexes
                    if tuple((field, int(direction)) for field, direction in index["key"].items()) == definition.keys
                    and bool(index.get("unique", False)) == definition.unique
                ),
                None,
            )
            if matched is not None:
                created.append(f"{definition.collection}:{matched['name']}")
                continue

            same_name = next(
                (index for index in current_indexes if index.get("name") == definition.name),
                None,
            )
            if same_name is not None:
                self.db[definition.collection].drop_index(definition.name)
                current_indexes = self.current_indexes(definition.collection)

            same_keys = next(
                (
                    index
                    for index in current_indexes
                    if self._index_keys(index) == definition.keys
                    and bool(index.get("unique", False)) != definition.unique
                ),
                None,
            )
            if same_keys is not None:
                self.db[definition.collection].drop_index(same_keys["name"])

            name = self.db[definition.collection].create_index(
                list(definition.keys),
                unique=definition.unique,
                name=definition.name,
            )
            created.append(f"{definition.collection}:{name}")
        return created


class BaseMongoRepository:
    def __init__(self, db):
        self.db = db

    def _find_frame(
        self,
        query: dict[str, Any],
        *,
        fields: Sequence[str] | None = None,
        sort: list[tuple[str, int]] | tuple[str, int] | None = None,
        limit: int | None = None,
        datetime_fields: Sequence[str] = (),
    ) -> pd.DataFrame:
        return _find_frame(
            self.collection,
            query,
            fields=fields,
            sort=sort,
            limit=limit,
            datetime_fields=datetime_fields,
        )


class StockBasicRepository(BaseMongoRepository):
    collection_name = "A_stock_market_basic_info"
    core_fields = ("code", "code_name", "ipoDate", "outDate")
    extra_fields = ("area", "industry", "cnspell", "market", "act_name", "act_ent_type")
    default_fields = core_fields + extra_fields

    @property
    def collection(self):
        return self.db[self.collection_name]

    def get_by_codes(
        self,
        codes: Sequence[str],
        *,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        normalized_codes = _normalize_codes(codes)
        return self._find_frame(
            {"code": {"$in": normalized_codes}},
            fields=fields,
        )

    def get_basic_info(
        self,
        codes: Sequence[str],
        *,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        requested_fields = list(fields) if fields is not None else list(self.default_fields)
        if "code" not in requested_fields:
            requested_fields.insert(0, "code")
        return self.get_by_codes(codes, fields=requested_fields)

    def get_name_map(self, codes: Sequence[str]) -> dict[str, str]:
        frame = self.get_by_codes(codes, fields=["code", "code_name"])
        if frame.empty:
            return {}
        return frame.set_index("code")["code_name"].to_dict()

    def get_info_map(
        self,
        codes: Sequence[str],
        *,
        fields: Sequence[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        frame = self.get_basic_info(codes, fields=fields)
        if frame.empty:
            return {}
        return frame.set_index("code").to_dict("index")


class DayKlineRepository(BaseMongoRepository):
    collection_name = "A_stock_market_day_kline"
    time_field = "date"

    @property
    def collection(self):
        return self.db[self.collection_name]

    def get_bars(
        self,
        codes: Sequence[str],
        start: datetime | str,
        end: datetime | str,
        *,
        fields: Sequence[str] | None = None,
        include_stopped: bool = False,
        standardize: bool = True,
    ) -> pd.DataFrame:
        normalized_codes = _normalize_codes(codes)
        query: dict[str, Any] = {
            "code": {"$in": normalized_codes},
            self.time_field: _date_range_query(self.time_field, start, end),
        }
        if not include_stopped:
            query["tradestatus"] = True

        frame = self._find_frame(
            query,
            fields=fields,
            sort=[("code", ASCENDING), (self.time_field, ASCENDING)],
        )
        if standardize:
            return _standardize_kline_frame(frame, time_field=self.time_field)
        return frame

    def get_grouped_bars(
        self,
        codes: Sequence[str],
        start: datetime | str,
        end: datetime | str,
        *,
        fields: Sequence[str] | None = None,
        include_stopped: bool = False,
        standardize: bool = True,
    ) -> dict[str, pd.DataFrame]:
        frame = self.get_bars(
            codes,
            start,
            end,
            fields=fields,
            include_stopped=include_stopped,
            standardize=standardize,
        )
        if frame.empty:
            return {}

        time_column = "trade_time" if standardize else self.time_field
        grouped: dict[str, pd.DataFrame] = {}
        for code, group in frame.groupby("code"):
            grouped[code] = sort_frame(group, sort_field=time_column)
        return grouped

    def get_latest_bars(
        self,
        codes: Sequence[str],
        *,
        trade_date: datetime | str | None = None,
        standardize: bool = True,
    ) -> pd.DataFrame:
        normalized_codes = [normalize_code(code) for code in codes]
        match_stage: dict[str, Any] = {
            "code": {"$in": normalized_codes},
            "tradestatus": True,
        }
        if trade_date is not None:
            trade_dt = _to_datetime(trade_date)
            match_stage[self.time_field] = {"$lte": trade_dt}

        pipeline = [
            {"$match": match_stage},
            {"$sort": {"code": 1, self.time_field: -1}},
            {"$group": {"_id": "$code", "doc": {"$first": "$$ROOT"}}},
            {"$replaceRoot": {"newRoot": "$doc"}},
        ]
        frame = records_to_frame(self.collection.aggregate(pipeline))
        if standardize:
            return _standardize_kline_frame(frame, time_field=self.time_field)
        return frame


class MinuteKlineRepository(BaseMongoRepository):
    time_field = "dt"

    def get_bars(
        self,
        code: str,
        start: datetime | str,
        end: datetime | str,
        *,
        fields: Sequence[str] | None = None,
        standardize: bool = True,
    ) -> pd.DataFrame:
        normalized_code = normalize_code(code)
        collection = self.db[minute_collection_name(normalized_code)]
        query = {self.time_field: _date_range_query(self.time_field, start, end, inclusive_end=False)}
        frame = _find_frame(
            collection,
            query,
            fields=fields,
            sort=[(self.time_field, ASCENDING)],
        )
        if frame.empty:
            return frame

        frame.insert(0, "code", normalized_code)
        if standardize:
            return _standardize_kline_frame(frame, time_field=self.time_field)
        return frame

    def get_bars_batch(
        self,
        codes: Sequence[str],
        start: datetime | str,
        end: datetime | str,
        *,
        fields: Sequence[str] | None = None,
        standardize: bool = True,
    ) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        for code in codes:
            normalized_code = normalize_code(code)
            result[normalized_code] = self.get_bars(
                normalized_code,
                start,
                end,
                fields=fields,
                standardize=standardize,
            )
        return result

    def get_latest_bar(self, code: str, *, standardize: bool = True) -> pd.DataFrame:
        normalized_code = normalize_code(code)
        collection = self.db[minute_collection_name(normalized_code)]
        doc = collection.find_one(sort=[(self.time_field, DESCENDING)])
        if doc is None:
            return pd.DataFrame()

        frame = records_to_frame([doc]).drop(columns=["_id"], errors="ignore")
        frame.insert(0, "code", normalized_code)
        if standardize:
            return _standardize_kline_frame(frame, time_field=self.time_field)
        return frame


class FinanceRepository(BaseMongoRepository):
    collection_name = "A_stock_market_finance_data"

    @property
    def collection(self):
        return self.db[self.collection_name]

    def get_visible_reports(
        self,
        codes: Sequence[str],
        *,
        as_of: datetime | str,
        start_pub_date: datetime | str | None = None,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        normalized_codes = _normalize_codes(codes)
        query: dict[str, Any] = {
            "code": {"$in": normalized_codes},
            "pubDate": {"$lte": _to_datetime(as_of)},
        }
        if start_pub_date is not None:
            query["pubDate"]["$gte"] = _to_datetime(start_pub_date)

        return self._find_frame(
            query,
            fields=fields,
            sort=[("code", ASCENDING), ("pubDate", ASCENDING), ("statDate", ASCENDING)],
            datetime_fields=("pubDate", "statDate"),
        )

    def get_reports(
        self,
        codes: Sequence[str],
        *,
        start_pub_date: datetime | str | None = None,
        end_pub_date: datetime | str | None = None,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        normalized_codes = _normalize_codes(codes)
        query: dict[str, Any] = {"code": {"$in": normalized_codes}}
        if start_pub_date is not None or end_pub_date is not None:
            pub_date_query: dict[str, datetime] = {}
            if start_pub_date is not None:
                pub_date_query["$gte"] = _to_datetime(start_pub_date)
            if end_pub_date is not None:
                pub_date_query["$lte"] = _to_datetime(end_pub_date)
            query["pubDate"] = pub_date_query

        return self._find_frame(
            query,
            fields=fields,
            sort=[("code", ASCENDING), ("pubDate", ASCENDING), ("statDate", ASCENDING)],
            datetime_fields=("pubDate", "statDate"),
        )


class MinerviniFundamentalFeatureRepository(BaseMongoRepository):
    collection_name = "A_stock_market_minervini_fundamental_feature"

    @property
    def collection(self):
        return self.db[self.collection_name]

    def get_features(
        self,
        codes: Sequence[str],
        *,
        start_pub_date: datetime | str | None = None,
        end_pub_date: datetime | str | None = None,
        fields: Sequence[str] | None = None,
        feature_version: str = "minervini_fundamental_v1",
    ) -> pd.DataFrame:
        normalized_codes = _normalize_codes(codes)
        query: dict[str, Any] = {
            "code": {"$in": normalized_codes},
            "featureVersion": feature_version,
        }
        if start_pub_date is not None or end_pub_date is not None:
            pub_date_query: dict[str, datetime] = {}
            if start_pub_date is not None:
                pub_date_query["$gte"] = _to_datetime(start_pub_date)
            if end_pub_date is not None:
                pub_date_query["$lte"] = _to_datetime(end_pub_date)
            query["pubDate"] = pub_date_query

        return self._find_frame(
            query,
            fields=fields,
            sort=[("code", ASCENDING), ("pubDate", ASCENDING), ("statDate", ASCENDING)],
            datetime_fields=("pubDate", "statDate", "revisionDate", "computedAt"),
        )

    def get_latest_features_before(
        self,
        codes: Sequence[str],
        *,
        as_of: datetime | str,
        fields: Sequence[str] | None = None,
        feature_version: str = "minervini_fundamental_v1",
    ) -> pd.DataFrame:
        return self.get_features(
            codes,
            end_pub_date=as_of,
            fields=fields,
            feature_version=feature_version,
        )


class DividendRepository(BaseMongoRepository):
    collection_name = "A_stock_market_dividend_data"
    operate_date_field = "dividOperateDate"
    datetime_fields = (
        "dividPreNoticeDate",
        "dividAgmPumDate",
        "dividPlanAnnounceDate",
        "dividPlanDate",
        "dividRegistDate",
        "dividOperateDate",
        "dividPayDate",
        "dividStockMarketDate",
    )

    @property
    def collection(self):
        return self.db[self.collection_name]

    def _normalize_datetime_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """把分红送转表中的日期列统一转成 pandas datetime。"""

        return _apply_datetime_columns(frame, self.datetime_fields)

    def get_series(
        self,
        start: datetime | str,
        end: datetime | str,
        *,
        codes: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """按除权除息日区间读取原始分红送转记录。"""

        query: dict[str, Any] = {
            self.operate_date_field: _date_range_query(self.operate_date_field, start, end)
        }
        if codes:
            query["code"] = {"$in": _normalize_codes(codes)}

        frame = self._find_frame(
            query,
            fields=fields,
            sort=[("code", ASCENDING), (self.operate_date_field, ASCENDING)],
        )
        return self._normalize_datetime_columns(frame)

    def get_operate_slice(
        self,
        trade_date: datetime | str,
        *,
        codes: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """读取某个除权除息日的原始分红送转记录。"""

        query: dict[str, Any] = {self.operate_date_field: _to_datetime(trade_date)}
        if codes:
            query["code"] = {"$in": _normalize_codes(codes)}

        frame = self._find_frame(
            query,
            fields=fields,
            sort=[("code", ASCENDING), ("dividRegistDate", ASCENDING), ("dividPayDate", ASCENDING)],
        )
        return self._normalize_datetime_columns(frame)


class AdjustFactorRepository(BaseMongoRepository):
    collection_name = "A_stock_market_adjust_factor"

    @property
    def collection(self):
        return self.db[self.collection_name]

    def get_factors(
        self,
        codes: Sequence[str],
        start: datetime | str,
        end: datetime | str,
        *,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        normalized_codes = _normalize_codes(codes)
        return self._find_frame(
            {
                "code": {"$in": normalized_codes},
                "date": _date_range_query("date", start, end),
            },
            fields=fields,
            sort=[("code", ASCENDING), ("date", ASCENDING)],
            datetime_fields=("date",),
        )


class FeatureRepository(BaseMongoRepository):
    collection_name = "A_stock_market_feature"

    @property
    def collection(self):
        return self.db[self.collection_name]

    def get_cross_section(
        self,
        trade_date: datetime | str,
        *,
        codes: Sequence[str] | None = None,
        filters: dict[str, Any] | None = None,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        query: dict[str, Any] = {"date": _to_datetime(trade_date)}
        if codes:
            query["code"] = {"$in": _normalize_codes(codes)}
        if filters:
            query.update(filters)

        return self._find_frame(
            query,
            fields=fields,
            sort=[("code", ASCENDING)],
            datetime_fields=("date",),
        )

    def get_series(
        self,
        codes: Sequence[str],
        start: datetime | str,
        end: datetime | str,
        *,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        normalized_codes = _normalize_codes(codes)
        return self._find_frame(
            {
                "code": {"$in": normalized_codes},
                "date": _date_range_query("date", start, end),
            },
            fields=fields,
            sort=[("code", ASCENDING), ("date", ASCENDING)],
            datetime_fields=("date",),
        )

    def get_series_by_filters(
        self,
        start: datetime | str,
        end: datetime | str,
        *,
        codes: Sequence[str] | None = None,
        filters: dict[str, Any] | None = None,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        query: dict[str, Any] = {
            "date": _date_range_query("date", start, end),
        }
        if codes:
            query["code"] = {"$in": _normalize_codes(codes)}
        if filters:
            query.update(filters)

        return self._find_frame(
            query,
            fields=fields,
            sort=[("date", ASCENDING), ("code", ASCENDING)],
            datetime_fields=("date",),
        )

    def top_by_field(
        self,
        trade_date: datetime | str,
        field_name: str,
        *,
        limit: int = 50,
        ascending: bool = True,
        filters: dict[str, Any] | None = None,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        query: dict[str, Any] = {"date": _to_datetime(trade_date)}
        if filters:
            query.update(filters)

        projection_fields = list(fields) if fields else ["code", "date", field_name]
        if field_name not in projection_fields:
            projection_fields.append(field_name)

        return self._find_frame(
            query,
            fields=projection_fields,
            sort=[(field_name, ASCENDING if ascending else DESCENDING)],
            limit=limit,
            datetime_fields=("date",),
        )


class MongoRepository:
    """
    Facade for all Mongo-backed research data.

    The goal is to keep business logic unaware of:
    - collection naming differences
    - date vs dt field differences
    - compressed OHLCV field names
    """

    def __init__(self, db_client):
        self.db_client = db_client
        self.db = db_client.db if hasattr(db_client, "db") else db_client
        self.stock_basic = StockBasicRepository(self.db)
        self.day_kline = DayKlineRepository(self.db)
        self.minute_kline = MinuteKlineRepository(self.db)
        self.finance = FinanceRepository(self.db)
        self.minervini_fundamental_feature = MinerviniFundamentalFeatureRepository(self.db)
        self.dividend = DividendRepository(self.db)
        self.adjust_factor = AdjustFactorRepository(self.db)
        self.feature = FeatureRepository(self.db)
        self.indexes = MongoIndexManager(self.db)
