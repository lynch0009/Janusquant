"""DuckDB-backed data portal for local research and backtest runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backtest.db.duckdb import DuckDBConfig
from backtest.utils import normalize_internal_code, to_pydatetime

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
SUPPORTED_PRICE_MODES = ("raw", "qfq", "hfq")
A_STOCK_DAY_KLINE_TABLE = "A_stock_market_day_kline"
A_STOCK_FEATURE_TABLE = "A_stock_market_feature"
A_STOCK_BASIC_TABLE = "A_stock_market_basic_info"
A_STOCK_ADJUST_FACTOR_TABLE = "A_stock_market_adjust_factor"
A_STOCK_DIVIDEND_TABLE = "A_stock_market_dividend_data"
A_STOCK_FINANCE_TABLE = "A_stock_market_finance_data"
A_STOCK_AKSHARE_QUARTERLY_FINANCE_TABLE = "A_stock_market_akshare_quarterly_finance"
A_STOCK_MINERVINI_FUNDAMENTAL_FEATURE_TABLE = "A_stock_market_minervini_fundamental_feature"
A_STOCK_CONCEPT_TABLE = "A_stock_concept"
A_STOCK_CONCEPT_FEATURE_TABLE = "A_stock_concept_feature"


def _to_datetime(value: datetime | str | pd.Timestamp | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        return pd.to_datetime(value).to_pydatetime()
    return to_pydatetime(value)


def _normalize_codes(codes: Sequence[str] | None) -> list[str]:
    if not codes:
        return []
    return [normalize_internal_code(code) for code in codes]


def _placeholders(values: Sequence[Any]) -> str:
    return ", ".join("?" for _ in values)


class DuckDBDataPortal:
    """Data portal implemented against normalized DuckDB tables."""

    def __init__(self, db_client: DuckDBConfig | str | Path, *, calendar_code: str = "sh.000001"):
        self.db_client = db_client if isinstance(db_client, DuckDBConfig) else DuckDBConfig(db_client)
        self.connection = self.db_client.connection
        self.feature_service = self
        self.calendar_code = normalize_internal_code(calendar_code)
        self._feature_cache: dict[tuple[datetime, tuple[str, ...]], pd.DataFrame] = {}

    @staticmethod
    def normalize_price_mode(price_mode: str = "hfq") -> str:
        normalized = str(price_mode).strip().lower()
        if normalized not in SUPPORTED_PRICE_MODES:
            raise ValueError(f"price_mode must be one of {SUPPORTED_PRICE_MODES}, got {price_mode}")
        return normalized

    def apply_price_mode(self, df_kline: pd.DataFrame, *, price_mode: str = "hfq") -> pd.DataFrame:
        normalized_mode = self.normalize_price_mode(price_mode)
        if df_kline is None or df_kline.empty or normalized_mode == "raw":
            return pd.DataFrame() if df_kline is None else df_kline.copy()

        working = df_kline.copy()
        original_columns = list(working.columns)
        price_rename_map = {
            "open": "o",
            "high": "h",
            "low": "l",
            "close": "c",
            "preclose": "prec",
        }
        price_restore_map = {value: key for key, value in price_rename_map.items() if key in original_columns}
        if "trade_date" in working.columns and "date" not in working.columns:
            working = working.rename(columns={"trade_date": "date"})
        working = working.rename(columns={column: mapped for column, mapped in price_rename_map.items() if column in working.columns})
        if not {"code", "date"}.issubset(working.columns):
            raise ValueError("df_kline must contain code and date/trade_date columns")

        working["code"] = working["code"].map(normalize_internal_code)
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
        for column in ("o", "h", "l", "c", "prec"):
            if column in working.columns:
                working[column] = pd.to_numeric(working[column], errors="coerce")

        codes = sorted(working["code"].dropna().unique().tolist())
        factor_df = pd.DataFrame()
        if codes:
            factor_df = self._query_frame(
                A_STOCK_ADJUST_FACTOR_TABLE,
                ["code", "date", "qfq_fac", "hfq_fac"],
                [f"code in ({_placeholders(codes)})", "date <= ?"],
                [*codes, working["date"].max()],
                "code, date",
            )

        if factor_df.empty:
            merged = working.copy()
            merged["qfq_fac"] = 1.0
            merged["hfq_fac"] = 1.0
        else:
            factor_df = factor_df.drop_duplicates(subset=["code", "date"], keep="last").copy()
            factor_df["date"] = pd.to_datetime(factor_df["date"], errors="coerce").dt.normalize()
            factor_df["qfq_fac"] = pd.to_numeric(factor_df["qfq_fac"], errors="coerce")
            factor_df["hfq_fac"] = pd.to_numeric(factor_df["hfq_fac"], errors="coerce")
            merged = pd.merge_asof(
                working.sort_values(["date", "code"]).reset_index(drop=True),
                factor_df.sort_values(["date", "code"]).reset_index(drop=True),
                on="date",
                by="code",
            ).copy()
            merged["qfq_fac"] = merged["qfq_fac"].fillna(1.0)
            merged["hfq_fac"] = merged["hfq_fac"].fillna(1.0)

        factor_column = "qfq_fac" if normalized_mode == "qfq" else "hfq_fac"
        for column in ("o", "h", "l", "c", "prec"):
            if column in merged.columns:
                merged[column] = merged[column] * merged[factor_column]

        if price_restore_map:
            merged = merged.rename(columns=price_restore_map)
        if "trade_date" in original_columns and "date" not in original_columns:
            merged = merged.rename(columns={"date": "trade_date"})
        ordered_columns = original_columns + [column for column in ("qfq_fac", "hfq_fac") if column not in original_columns]
        existing_columns = [column for column in ordered_columns if column in merged.columns]
        remaining_columns = [column for column in merged.columns if column not in existing_columns]
        return merged[existing_columns + remaining_columns].reset_index(drop=True)

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

    def _table_exists(self, table_name: str) -> bool:
        frame = self.db_client.fetch_df(
            "select count(*) as count from information_schema.tables where table_name = ?",
            [table_name],
        )
        return bool(frame["count"].iloc[0])

    def _append_filters(self, clauses: list[str], params: list[Any], filters: dict[str, Any] | None) -> None:
        if not filters:
            return
        for field, value in filters.items():
            if isinstance(value, dict):
                for operator, operand in value.items():
                    if operator == "$in":
                        items = list(operand)
                        if not items:
                            clauses.append("1 = 0")
                        else:
                            clauses.append(f"{field} in ({_placeholders(items)})")
                            params.extend(items)
                    elif operator == "$gte":
                        clauses.append(f"{field} >= ?")
                        params.append(operand)
                    elif operator == "$gt":
                        clauses.append(f"{field} > ?")
                        params.append(operand)
                    elif operator == "$lte":
                        clauses.append(f"{field} <= ?")
                        params.append(operand)
                    elif operator == "$lt":
                        clauses.append(f"{field} < ?")
                        params.append(operand)
                    else:
                        raise ValueError(f"unsupported DuckDB filter operator: {operator}")
            else:
                clauses.append(f"{field} = ?")
                params.append(value)

    def _query_frame(
        self,
        table: str,
        fields: Sequence[str],
        clauses: list[str],
        params: list[Any],
        order_by: str,
        *,
        limit: int | None = None,
    ) -> pd.DataFrame:
        select_fields = ", ".join(fields)
        sql = f"select {select_fields} from {table}"
        if clauses:
            sql += " where " + " and ".join(clauses)
        if order_by:
            sql += f" order by {order_by}"
        if limit is not None:
            sql += " limit ?"
            params.append(int(limit))
        return self.db_client.fetch_df(sql, params)

    def _select_existing_fields(self, table: str, fields: Sequence[str]) -> list[str]:
        return [field for field in fields if self._column_exists(table, field)]

    def get_trade_calendar(self, start_date: datetime, end_date: datetime) -> list[datetime]:
        start_dt = _to_datetime(start_date)
        end_dt = _to_datetime(end_date)
        frame = self.db_client.fetch_df(
            f"""
            select distinct date
            from {A_STOCK_DAY_KLINE_TABLE}
            where code = ? and date >= ? and date <= ? and tradestatus = true
            order by date
            """,
            [self.calendar_code, start_dt, end_dt],
        )
        if len(frame) < 2:
            frame = self.db_client.fetch_df(
                f"""
                select distinct date
                from {A_STOCK_DAY_KLINE_TABLE}
                where date >= ? and date <= ? and tradestatus = true
                order by date
                """,
                [start_dt, end_dt],
            )
        return pd.DatetimeIndex(pd.to_datetime(frame["date"])).to_pydatetime().tolist() if not frame.empty else []

    def get_feature_history(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        fields: Sequence[str] | None = None,
        filters: dict | None = None,
    ) -> pd.DataFrame:
        requested_fields = list(sorted(set((fields or []) + ["code", "date"])))
        clauses = ["date >= ?", "date <= ?"]
        params: list[Any] = [_to_datetime(start_date), _to_datetime(end_date)]
        self._append_filters(clauses, params, filters)
        frame = self._query_frame(A_STOCK_FEATURE_TABLE, requested_fields, clauses, params, "date, code")
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def get_feature_slice(
        self,
        trade_date: datetime,
        *,
        fields: Sequence[str] | None = None,
        filters: dict | None = None,
    ) -> pd.DataFrame:
        requested_fields = tuple(sorted(set((fields or []) + ["code", "date"])))
        cache_key = (pd.Timestamp(trade_date).to_pydatetime(), requested_fields)
        if filters:
            return self._get_feature_slice_uncached(trade_date, requested_fields, filters)
        if cache_key not in self._feature_cache:
            self._feature_cache[cache_key] = self._get_feature_slice_uncached(trade_date, requested_fields, None)
        return self._feature_cache[cache_key].copy()

    def _get_feature_slice_uncached(
        self,
        trade_date: datetime,
        fields: Sequence[str],
        filters: dict | None,
    ) -> pd.DataFrame:
        clauses = ["date = ?"]
        params: list[Any] = [_to_datetime(trade_date)]
        self._append_filters(clauses, params, filters)
        frame = self._query_frame(A_STOCK_FEATURE_TABLE, list(fields), clauses, params, "code")
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def get_stock_basic(self, codes: Sequence[str], *, fields: Sequence[str] | None = None) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame()
        normalized_codes = _normalize_codes(codes)
        requested_fields = list(fields) if fields is not None else [
            "code",
            "code_name",
            "ipoDate",
            "outDate",
            "area",
            "industry",
            "cnspell",
            "market",
            "act_name",
            "act_ent_type",
        ]
        if "code" not in requested_fields:
            requested_fields.insert(0, "code")
        frame = self._query_frame(
            A_STOCK_BASIC_TABLE,
            requested_fields,
            [f"code in ({_placeholders(normalized_codes)})"],
            list(normalized_codes),
            "code",
        )
        for field in ("ipoDate", "outDate"):
            if field in frame.columns:
                frame[field] = pd.to_datetime(frame[field])
        return frame

    def get_stock_name_map(self, codes: Sequence[str], *, preserve_input_code: bool = True) -> dict[str, str]:
        if not codes:
            return {}
        normalized_by_input = {code: normalize_internal_code(code) for code in codes}
        normalized_codes = list(dict.fromkeys(normalized_by_input.values()))
        frame = self.get_stock_basic(normalized_codes, fields=["code", "code_name"])
        if frame.empty:
            return {}
        name_by_normalized = frame.set_index("code")["code_name"].fillna("").astype(str).to_dict()
        if not preserve_input_code:
            return name_by_normalized
        return {
            code: name_by_normalized[normalized_code]
            for code, normalized_code in normalized_by_input.items()
            if normalized_code in name_by_normalized
        }

    def get_visible_finance_slice(
        self,
        trade_date: datetime,
        *,
        codes: Sequence[str],
        fields: Sequence[str] | None = None,
        start_pub_date: datetime | None = None,
    ) -> pd.DataFrame:
        requested_fields = list(sorted(set((fields or []) + ["code", "pubDate", "statDate"])))
        clauses = [f"code in ({_placeholders(_normalize_codes(codes))})", "pubDate <= ?"]
        params: list[Any] = [*_normalize_codes(codes), _to_datetime(trade_date)]
        if start_pub_date is not None:
            clauses.append("pubDate >= ?")
            params.append(_to_datetime(start_pub_date))
        frame = self._query_frame(A_STOCK_FINANCE_TABLE, requested_fields, clauses, params, "code, pubDate, statDate")
        for field in ("pubDate", "statDate"):
            if field in frame.columns:
                frame[field] = pd.to_datetime(frame[field])
        return frame

    def get_finance_reports(
        self,
        *,
        codes: Sequence[str],
        start_pub_date: datetime | None = None,
        end_pub_date: datetime | None = None,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        requested_fields = list(sorted(set((fields or []) + ["code", "pubDate", "statDate"])))
        normalized_codes = _normalize_codes(codes)
        clauses = [f"code in ({_placeholders(normalized_codes)})"]
        params: list[Any] = list(normalized_codes)
        if start_pub_date is not None:
            clauses.append("pubDate >= ?")
            params.append(_to_datetime(start_pub_date))
        if end_pub_date is not None:
            clauses.append("pubDate <= ?")
            params.append(_to_datetime(end_pub_date))
        frame = self._query_frame(A_STOCK_FINANCE_TABLE, requested_fields, clauses, params, "code, pubDate, statDate")
        for field in ("pubDate", "statDate"):
            if field in frame.columns:
                frame[field] = pd.to_datetime(frame[field])
        return frame

    def get_minervini_fundamental_features(
        self,
        *,
        codes: Sequence[str],
        start_pub_date: datetime | None = None,
        end_pub_date: datetime | None = None,
        fields: Sequence[str] | None = None,
        feature_version: str = "minervini_fundamental_v1",
    ) -> pd.DataFrame:
        requested_fields = list(sorted(set((fields or []) + ["code", "pubDate", "statDate", "featureVersion"])))
        normalized_codes = _normalize_codes(codes)
        clauses = [f"code in ({_placeholders(normalized_codes)})", "featureVersion = ?"]
        params: list[Any] = [*normalized_codes, feature_version]
        if start_pub_date is not None:
            clauses.append("pubDate >= ?")
            params.append(_to_datetime(start_pub_date))
        if end_pub_date is not None:
            clauses.append("pubDate <= ?")
            params.append(_to_datetime(end_pub_date))
        frame = self._query_frame(
            A_STOCK_MINERVINI_FUNDAMENTAL_FEATURE_TABLE,
            requested_fields,
            clauses,
            params,
            "code, pubDate, statDate",
        )
        for field in ("pubDate", "statDate", "revisionDate", "computedAt"):
            if field in frame.columns:
                frame[field] = pd.to_datetime(frame[field])
        return frame

    def get_concept_boards(
        self,
        *,
        board_types: str | Sequence[str] | None = None,
        board_names: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        filters: dict | None = None,
    ) -> pd.DataFrame:
        requested_fields = list(sorted(set((fields or []) + ["board_name", "board_type"])))
        clauses: list[str] = []
        params: list[Any] = []
        board_type_list = [board_types] if isinstance(board_types, str) else list(board_types or [])
        if board_type_list:
            clauses.append(f"board_type in ({_placeholders(board_type_list)})")
            params.extend(board_type_list)
        if board_names:
            clauses.append(f"board_name in ({_placeholders(board_names)})")
            params.extend(list(board_names))
        self._append_filters(clauses, params, filters)
        return self._query_frame(A_STOCK_CONCEPT_TABLE, requested_fields, clauses, params, "board_type, board_name")

    def get_stock_concept_map(
        self,
        *,
        codes: Sequence[str],
        board_types: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        normalized_codes = set(_normalize_codes(codes))
        boards = self.get_concept_boards(board_types=board_types, fields=["board_name", "board_type", "stocks"])
        rows: list[dict[str, Any]] = []
        for row in boards.to_dict("records"):
            stocks = row.get("stocks")
            if stocks is None or (isinstance(stocks, float) and pd.isna(stocks)):
                stocks = []
            if isinstance(stocks, str):
                import ast

                try:
                    stocks = ast.literal_eval(stocks)
                except Exception:
                    stocks = []
            for code in stocks:
                normalized = normalize_internal_code(code)
                if normalized in normalized_codes:
                    rows.append({"code": normalized, "board_name": row.get("board_name"), "board_type": row.get("board_type")})
        return pd.DataFrame(rows)

    def get_concept_feature_slice(
        self,
        trade_date: datetime,
        *,
        board_types: str | Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        filters: dict | None = None,
        limit: int | None = None,
        ascending: bool = False,
        sort_field: str = "score",
    ) -> pd.DataFrame:
        requested_fields = list(sorted(set((fields or []) + ["date", "board_type", "board_name", sort_field])))
        clauses = ["date = ?"]
        params: list[Any] = [_to_datetime(trade_date)]
        board_type_list = [board_types] if isinstance(board_types, str) else list(board_types or [])
        if board_type_list:
            clauses.append(f"board_type in ({_placeholders(board_type_list)})")
            params.extend(board_type_list)
        self._append_filters(clauses, params, filters)
        direction = "asc" if ascending else "desc"
        frame = self._query_frame(
            A_STOCK_CONCEPT_FEATURE_TABLE,
            requested_fields,
            clauses,
            params,
            f"{sort_field} {direction}, board_name",
            limit=limit,
        )
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def get_concept_feature_history(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        board_types: str | Sequence[str] | None = None,
        board_names: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        filters: dict | None = None,
    ) -> pd.DataFrame:
        requested_fields = list(sorted(set((fields or []) + ["date", "board_type", "board_name", "score"])))
        clauses = ["date >= ?", "date <= ?"]
        params: list[Any] = [_to_datetime(start_date), _to_datetime(end_date)]
        board_type_list = [board_types] if isinstance(board_types, str) else list(board_types or [])
        if board_type_list:
            clauses.append(f"board_type in ({_placeholders(board_type_list)})")
            params.extend(board_type_list)
        if board_names:
            clauses.append(f"board_name in ({_placeholders(board_names)})")
            params.extend(list(board_names))
        self._append_filters(clauses, params, filters)
        frame = self._query_frame(
            A_STOCK_CONCEPT_FEATURE_TABLE,
            requested_fields,
            clauses,
            params,
            "date, board_type, score desc, board_name",
        )
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        return frame

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
        feature_history = self.get_feature_history(
            start_date,
            end_date,
            fields=feature_fields,
            filters={"code": {"$in": _normalize_codes(codes)}} if codes else None,
        )
        if feature_history.empty:
            return feature_history
        return feature_history.rename(columns={"date": "trade_date"}).reset_index(drop=True)

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
        normalized_price_mode = self.normalize_price_mode(price_mode)
        requested_fields = self._normalize_daily_history_fields(fields)
        if codes and batch_size and len(codes) > batch_size:
            frames = [
                self.get_daily_history(
                    start_date,
                    end_date,
                    codes=[normalize_internal_code(code) for code in codes][offset : offset + batch_size],
                    fields=fields,
                    include_stopped=include_stopped,
                    batch_size=None,
                    price_mode=normalized_price_mode,
                )
                for offset in range(0, len(codes), batch_size)
            ]
            frames = [frame for frame in frames if not frame.empty]
            return pd.concat(frames, ignore_index=True).sort_values(["code", "trade_date"]).reset_index(drop=True) if frames else pd.DataFrame()

        if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
            return pd.DataFrame(columns=requested_fields)

        raw_fields = self._resolve_day_kline_fields(requested_fields)
        clauses = ["date >= ?", "date < ?"]
        params: list[Any] = [_to_datetime(start_date), _to_datetime(end_date)]
        normalized_codes = _normalize_codes(codes)
        if normalized_codes:
            clauses.append(f"code in ({_placeholders(normalized_codes)})")
            params.extend(normalized_codes)
        if not include_stopped:
            clauses.append("tradestatus = true")

        frame = self._query_frame(A_STOCK_DAY_KLINE_TABLE, raw_fields, clauses, params, "code, date")
        if frame.empty:
            return frame

        frame = frame.rename(
            columns={
                "date": "trade_date",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "prec": "preclose",
                "v": "volume",
                "a": "amount",
            }
        ).reset_index(drop=True)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])

        if normalized_price_mode != "raw" and any(field in DAILY_PRICE_FIELDS for field in requested_fields):
            frame = self._apply_price_mode(frame, price_mode=normalized_price_mode)

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
        return frame[available_fields].sort_values(["code", "trade_date"]).reset_index(drop=True)

    def _apply_price_mode(self, frame: pd.DataFrame, *, price_mode: str) -> pd.DataFrame:
        if frame.empty or price_mode == "raw":
            return frame.copy()
        factor_column = "qfq_fac" if price_mode == "qfq" else "hfq_fac"
        codes = sorted(frame["code"].dropna().unique().tolist())
        if not codes:
            return frame.copy()
        max_date = pd.to_datetime(frame["trade_date"]).max()
        factor_frame = self._query_frame(
            A_STOCK_ADJUST_FACTOR_TABLE,
            ["code", "date", "qfq_fac", "hfq_fac"],
            [f"code in ({_placeholders(codes)})", "date <= ?"],
            [*codes, to_pydatetime(max_date)],
            "date, code",
        )
        working = frame.copy()
        if factor_frame.empty:
            working["qfq_fac"] = 1.0
            working["hfq_fac"] = 1.0
            return working
        working["trade_date"] = pd.to_datetime(working["trade_date"]).dt.normalize()
        factor_frame["date"] = pd.to_datetime(factor_frame["date"]).dt.normalize()
        merged = pd.merge_asof(
            working.sort_values(["trade_date", "code"], kind="mergesort"),
            factor_frame.sort_values(["date", "code"], kind="mergesort"),
            left_on="trade_date",
            right_on="date",
            by="code",
        ).drop(columns=["date"], errors="ignore")
        merged["qfq_fac"] = pd.to_numeric(merged["qfq_fac"], errors="coerce").fillna(1.0)
        merged["hfq_fac"] = pd.to_numeric(merged["hfq_fac"], errors="coerce").fillna(1.0)
        for column in ("open", "high", "low", "close", "preclose"):
            if column in merged.columns:
                merged[column] = pd.to_numeric(merged[column], errors="coerce") * merged[factor_column]
        return merged.sort_values(["code", "trade_date"]).reset_index(drop=True)

    def get_daily_bar_snapshot(self, codes: Sequence[str], trade_date: datetime) -> dict[str, pd.DataFrame]:
        if not codes:
            return {}
        normalized_codes = _normalize_codes(codes)
        frame = self._query_frame(
            A_STOCK_DAY_KLINE_TABLE,
            ["code", "date", "o", "h", "l", "c", "prec", "v", "a", "turn", "pctChg", "isST"],
            [f"code in ({_placeholders(normalized_codes)})", "date = ?", "tradestatus = true"],
            [*normalized_codes, _to_datetime(trade_date)],
            "code, date",
        )
        snapshot: dict[str, pd.DataFrame] = {code: pd.DataFrame() for code in normalized_codes}
        if frame.empty:
            return snapshot
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
        for code, group in frame.groupby("code"):
            snapshot[normalize_internal_code(code)] = group.sort_values("dt").reset_index(drop=True)
        return snapshot

    def get_daily_close_map(self, codes: Sequence[str], trade_date: datetime) -> dict[str, float]:
        if not codes:
            return {}
        normalized_codes = _normalize_codes(codes)
        frame = self.db_client.fetch_df(
            f"""
            select code, c as close
            from (
                select code, c, row_number() over (partition by code order by date desc) as rn
                from {A_STOCK_DAY_KLINE_TABLE}
                where code in ({_placeholders(normalized_codes)})
                  and date <= ?
                  and tradestatus = true
            )
            where rn = 1
            """,
            [*normalized_codes, _to_datetime(trade_date)],
        )
        if frame.empty:
            return {}
        return frame.set_index("code")["close"].to_dict()

    def get_market_amount_history(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        code_prefixes: Sequence[str] | None = None,
        include_stopped: bool = False,
    ) -> pd.DataFrame:
        normalized_prefixes = tuple(code_prefixes or ("sh.60", "sh.68", "sz.00", "sz.30"))
        if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
            return pd.DataFrame(columns=["trade_date", "market_amount", "security_count"])
        like_clause = " or ".join("code like ?" for _ in normalized_prefixes)
        params: list[Any] = [_to_datetime(start_date), _to_datetime(end_date), *[f"{prefix}%" for prefix in normalized_prefixes]]
        status_clause = "" if include_stopped else " and tradestatus = true"
        frame = self.db_client.fetch_df(
            f"""
            select date as trade_date, sum(a) as market_amount, count(*) as security_count
            from {A_STOCK_DAY_KLINE_TABLE}
            where date >= ? and date < ?
              and ({like_clause})
              {status_clause}
            group by date
            order by date
            """,
            params,
        )
        if frame.empty:
            return pd.DataFrame(columns=["trade_date", "market_amount", "security_count"])
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        return frame[["trade_date", "market_amount", "security_count"]].reset_index(drop=True)

    @staticmethod
    def _empty_corporate_action_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=list(STANDARDIZED_CORPORATE_ACTION_FIELDS))

    def get_corporate_action_events(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        codes: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        fields = [field for field in [*DIVIDEND_RAW_FIELDS, "dividStockMarketDate"] if self._column_exists(A_STOCK_DIVIDEND_TABLE, field)]
        clauses = ["dividOperateDate >= ?", "dividOperateDate < ?"]
        params: list[Any] = [_to_datetime(start_date), _to_datetime(end_date) + pd.Timedelta(days=1)]
        normalized_codes = _normalize_codes(codes)
        if normalized_codes:
            clauses.append(f"code in ({_placeholders(normalized_codes)})")
            params.extend(normalized_codes)
        dividend_frame = self._query_frame(A_STOCK_DIVIDEND_TABLE, fields, clauses, params, "code, dividOperateDate")
        return self._standardize_dividend_events(dividend_frame)

    def get_corporate_action_event_slice(
        self,
        trade_date: datetime,
        *,
        codes: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read corporate action events for one ex-dividend/operate date."""

        trade_dt = _to_datetime(trade_date)
        return self.get_corporate_action_events(trade_dt, trade_dt, codes=codes)

    def get_minute_bars_batch(self, *args, **kwargs) -> dict:
        raise NotImplementedError("DuckDB v1 does not migrate minute kline tables; minute bars still require the legacy Mongo path.")

    def _column_exists(self, table: str, column: str) -> bool:
        frame = self.db_client.fetch_df(
            """
            select count(*) as count
            from information_schema.columns
            where table_name = ? and column_name = ?
            """,
            [table, column],
        )
        return bool(frame["count"].iloc[0])

    def _standardize_dividend_events(self, dividend_frame: pd.DataFrame) -> pd.DataFrame:
        if dividend_frame.empty:
            return self._empty_corporate_action_frame()
        frame = dividend_frame.copy()
        for field in ("dividOperateDate", "dividPayDate", "dividStockMarketDate"):
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
                rows.append(
                    {
                        "event_type": "cash_dividend",
                        "code": row.code,
                        "operate_date": pd.Timestamp(operate_date).to_pydatetime(),
                        "settle_date": pd.Timestamp(pay_date if pay_date is not None and not pd.isna(pay_date) else operate_date).to_pydatetime(),
                        "cash_dividend_per_share": cash_dividend_per_share,
                        "stock_dividend_ratio": 0.0,
                        "stock_dividend_share_ratio": 0.0,
                        "reserve_to_stock_ratio": 0.0,
                        "raw_text": raw_text,
                    }
                )
            if stock_dividend_ratio > 0:
                rows.append(
                    {
                        "event_type": "stock_dividend",
                        "code": row.code,
                        "operate_date": pd.Timestamp(operate_date).to_pydatetime(),
                        "settle_date": pd.Timestamp(
                            stock_market_date if stock_market_date is not None and not pd.isna(stock_market_date) else operate_date
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
        return standardized.sort_values(["operate_date", "code", "event_type"]).reset_index(drop=True)


class CachedDuckDBDataPortal(DuckDBDataPortal):
    """Parquet-cached wrapper for DuckDBDataPortal."""

    def __init__(self, db_client: DuckDBConfig | str | Path, *, frame_cache, calendar_code: str = "sh.000001"):
        super().__init__(db_client, calendar_code=calendar_code)
        self.frame_cache = frame_cache

    def get_trade_calendar(self, start_date: datetime, end_date: datetime) -> list[datetime]:
        payload = {"calendar_code": self.calendar_code, "start_date": start_date, "end_date": end_date}

        def builder() -> pd.DataFrame:
            return pd.DataFrame({"trade_date": pd.to_datetime(super(CachedDuckDBDataPortal, self).get_trade_calendar(start_date, end_date))})

        frame = self.frame_cache.load_or_build_frame("trade_calendar", payload, builder)
        return sorted(pd.to_datetime(frame["trade_date"]).to_list()) if not frame.empty else []

    def get_feature_history(self, start_date: datetime, end_date: datetime, *, fields: Sequence[str] | None = None, filters: dict | None = None) -> pd.DataFrame:
        requested_fields = list(sorted(set((fields or []) + ["code", "date"])))
        payload = {"start_date": start_date, "end_date": end_date, "fields": requested_fields, "filters": filters or {}}
        return self.frame_cache.load_or_build_frame(
            "feature_history",
            payload,
            lambda: super(CachedDuckDBDataPortal, self).get_feature_history(start_date, end_date, fields=fields, filters=filters),
        )

    def get_stock_basic(self, codes: Sequence[str], *, fields: Sequence[str] | None = None) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame()
        payload = {"codes_signature": self.frame_cache.codes_signature(codes), "fields": list(fields or [])}
        return self.frame_cache.load_or_build_frame(
            "stock_basic",
            payload,
            lambda: super(CachedDuckDBDataPortal, self).get_stock_basic(codes, fields=fields),
        )

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
        requested_fields = self._normalize_daily_history_fields(fields)
        payload = {
            "start_date": start_date,
            "end_date": end_date,
            "codes_signature": self.frame_cache.codes_signature(codes),
            "fields": requested_fields,
            "include_stopped": include_stopped,
            "price_mode": self.normalize_price_mode(price_mode),
        }
        return self.frame_cache.load_or_build_frame(
            "daily_history",
            payload,
            lambda: super(CachedDuckDBDataPortal, self).get_daily_history(
                start_date,
                end_date,
                codes=codes,
                fields=fields,
                include_stopped=include_stopped,
                batch_size=batch_size,
                price_mode=price_mode,
            ),
        )

    def get_corporate_action_events(self, start_date: datetime, end_date: datetime, *, codes: Sequence[str] | None = None) -> pd.DataFrame:
        payload = {"start_date": start_date, "end_date": end_date, "codes_signature": self.frame_cache.codes_signature(codes)}
        return self.frame_cache.load_or_build_frame(
            "corporate_action_events",
            payload,
            lambda: super(CachedDuckDBDataPortal, self).get_corporate_action_events(start_date, end_date, codes=codes),
        )

    def get_corporate_action_event_slice(self, trade_date: datetime, *, codes: Sequence[str] | None = None) -> pd.DataFrame:
        payload = {"trade_date": trade_date, "codes_signature": self.frame_cache.codes_signature(codes)}
        return self.frame_cache.load_or_build_frame(
            "corporate_action_event_slice",
            payload,
            lambda: super(CachedDuckDBDataPortal, self).get_corporate_action_event_slice(trade_date, codes=codes),
        )
