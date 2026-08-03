from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from backtest.db import DuckDBConfig, upsert_frame
from backtest.db.duckdb_write import quote_identifier, table_exists
from backtest.utils import normalize_internal_code
from backtest.utils.log import format_date, log_event

FEATURE_START_DATE = datetime.strptime("2013-01-01", "%Y-%m-%d")
ADJUST_FACTOR_FETCH_START_DATE = datetime.strptime("2013-01-04", "%Y-%m-%d")
FEATURE_FLUSH_ROW_THRESHOLD = 100000
FEATURE_READ_CODE_BATCH_SIZE = 500
SUPPORTED_PRICE_MODES = ("raw", "qfq", "hfq")
A_SHARE_PREFIXES = ("sh.60", "sh.68", "sz.00", "sz.30")
DUCKDB_DAY_KLINE_TABLE = "A_stock_market_day_kline"
DUCKDB_FEATURE_TABLE = "A_stock_market_feature"
DUCKDB_FINANCE_TABLE = "A_stock_market_finance_data"
DUCKDB_ADJUST_FACTOR_TABLE = "A_stock_market_adjust_factor"


def _is_supported_a_share_code(code: str) -> bool:
    normalized = normalize_internal_code(code)
    return normalized.startswith(A_SHARE_PREFIXES)


def _normalize_datetime_column(series: pd.Series, *, normalize: bool) -> pd.Series:
    converted = pd.to_datetime(series, errors="coerce")
    if normalize:
        converted = converted.dt.normalize()
    return pd.Series(
        [None if pd.isna(value) else value.to_pydatetime() for value in converted],
        index=series.index,
        dtype=object,
    )


def feature_frame_to_records(df: pd.DataFrame) -> list[dict]:
    normalized = df.copy()
    for column in ("date", "pubDate", "statDate", "financePubDate"):
        if column in normalized.columns:
            normalized[column] = _normalize_datetime_column(normalized[column], normalize=True)
    return normalized.to_dict("records")


def dedupe_feature_docs_by_key(docs: list[dict]) -> tuple[list[dict], int]:
    keyed_docs: dict[tuple[object, object], dict] = {}
    key_order: list[tuple[object, object]] = []
    passthrough_docs: list[dict] = []
    duplicate_count = 0

    for doc in docs:
        code = doc.get("code")
        date = doc.get("date")
        if code is None or date is None:
            passthrough_docs.append(doc)
            continue
        key = (code, date)
        if key not in keyed_docs:
            key_order.append(key)
        else:
            duplicate_count += 1
        keyed_docs[key] = doc

    deduped = [keyed_docs[key] for key in key_order]
    deduped.extend(passthrough_docs)
    return deduped, duplicate_count


def prepare_frame_for_merge_asof(
    df: pd.DataFrame,
    on: str,
    by: str | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    result = df.copy()
    result[on] = pd.to_datetime(result[on], errors="coerce")
    if result[on].isna().any():
        raise ValueError(f"{on} contains null values and cannot be used with merge_asof")

    by_columns: list[str] = []
    if by is not None:
        by_columns = [by] if isinstance(by, str) else list(by)
        for column in by_columns:
            if result[column].isna().any():
                raise ValueError(f"{column} contains null values and cannot be used with merge_asof")

    return result.sort_values([on, *by_columns], kind="mergesort").reset_index(drop=True)


class DuckDBFeature:
    """DuckDB-backed feature generator."""

    def __init__(self, db_client: DuckDBConfig | None = None):
        self.db_client = db_client if isinstance(db_client, DuckDBConfig) else DuckDBConfig()
        self.connection = self.db_client.connection
        self._feature_index_ensured = False
        self.ensure_indexes()

    def ensure_indexes(self) -> None:
        if table_exists(self.db_client, DUCKDB_FEATURE_TABLE):
            self.db_client.execute(
                f"create unique index if not exists idx_feature_code_date "
                f"on {quote_identifier(DUCKDB_FEATURE_TABLE)}(code, date)"
            )
            self._feature_index_ensured = True

    @staticmethod
    def normalize_price_mode(price_mode: str = "hfq") -> str:
        normalized = str(price_mode).strip().lower()
        if normalized not in SUPPORTED_PRICE_MODES:
            raise ValueError(f"price_mode must be one of {SUPPORTED_PRICE_MODES}, got {price_mode}")
        return normalized

    @staticmethod
    def _placeholders(values: list[Any]) -> str:
        return ", ".join("?" for _ in values)

    def _fetch_df(self, sql: str, params: list[Any] | tuple[Any, ...] | None = None) -> pd.DataFrame:
        return self.db_client.fetch_df(sql, params or [])

    def _get_target_codes(self, codes: list[str] | None = None) -> list[str]:
        if codes:
            return sorted({normalize_internal_code(code) for code in codes})
        like_clause = " or ".join("code like ?" for _ in A_SHARE_PREFIXES)
        frame = self._fetch_df(
            f"""
            select distinct code
            from {DUCKDB_DAY_KLINE_TABLE}
            where {like_clause}
            order by code
            """,
            [f"{prefix}%" for prefix in A_SHARE_PREFIXES],
        )
        if frame.empty:
            return []
        return sorted(
            {
                normalize_internal_code(code)
                for code in frame["code"].dropna().astype(str)
                if _is_supported_a_share_code(code)
            }
        )

    def _get_latest_day_dates(self, codes: list[str]) -> dict[str, datetime]:
        if not codes:
            return {}
        frame = self._fetch_df(
            f"""
            select code, max(date) as date
            from {DUCKDB_DAY_KLINE_TABLE}
            where code in ({self._placeholders(codes)}) and tradestatus = true
            group by code
            """,
            codes,
        )
        return {
            row["code"]: pd.to_datetime(row["date"]).to_pydatetime()
            for row in frame.to_dict("records")
            if row.get("date") is not None and not pd.isna(row.get("date"))
        }

    def _get_latest_feature_snapshots(self, codes: list[str]) -> dict[str, dict]:
        if not codes:
            return {}
        frame = self._fetch_df(
            f"""
            select code, date, financePubDate
            from (
                select
                    code,
                    date,
                    financePubDate,
                    row_number() over (partition by code order by date desc) as rn
                from {DUCKDB_FEATURE_TABLE}
                where code in ({self._placeholders(codes)})
            )
            where rn = 1
            """,
            codes,
        )
        return {
            row["code"]: {
                "date": pd.to_datetime(row["date"]).to_pydatetime()
                if row.get("date") is not None and not pd.isna(row.get("date"))
                else None,
                "financePubDate": (
                    pd.to_datetime(row["financePubDate"]).to_pydatetime()
                    if row.get("financePubDate") is not None and not pd.isna(row.get("financePubDate"))
                    else None
                ),
            }
            for row in frame.to_dict("records")
        }

    def _get_latest_finance_pub_dates(self, codes: list[str]) -> dict[str, datetime]:
        if not codes:
            return {}
        frame = self._fetch_df(
            f"""
            select code, max(pubDate) as pubDate
            from {DUCKDB_FINANCE_TABLE}
            where code in ({self._placeholders(codes)})
            group by code
            """,
            codes,
        )
        return {
            row["code"]: pd.to_datetime(row["pubDate"]).to_pydatetime()
            for row in frame.to_dict("records")
            if row.get("pubDate") is not None and not pd.isna(row.get("pubDate"))
        }

    def build_incremental_sync_plan(self, codes: list[str] | None = None, force_full_refresh: bool = False):
        target_codes = self._get_target_codes(codes)
        if not target_codes:
            return []

        latest_day_dates = self._get_latest_day_dates(target_codes)
        latest_feature_snapshots = self._get_latest_feature_snapshots(target_codes)
        latest_finance_pub_dates = self._get_latest_finance_pub_dates(target_codes)
        sync_plan = []
        reason_stats = {"full_refresh": 0, "missing_feature": 0, "new_day_kline": 0, "new_finance": 0}

        for code in target_codes:
            latest_day_date = latest_day_dates.get(code)
            if latest_day_date is None:
                log_event("warning", "feature sync skip", code=code, reason="no_trade_date")
                continue

            if force_full_refresh:
                start_date = FEATURE_START_DATE
                reason = "full_refresh"
            else:
                latest_feature_snapshot = latest_feature_snapshots.get(code, {})
                latest_feature_date = latest_feature_snapshot.get("date")
                latest_feature_finance_pub_date = latest_feature_snapshot.get("financePubDate")
                latest_finance_pub_date = latest_finance_pub_dates.get(code)
                if latest_finance_pub_date is not None and latest_finance_pub_date > latest_day_date:
                    latest_finance_pub_date = None

                candidate_dates = []
                reasons = []
                if latest_feature_date is None:
                    candidate_dates.append(FEATURE_START_DATE)
                    reasons.append("missing_feature")
                elif latest_feature_date < latest_day_date:
                    candidate_dates.append(latest_feature_date + timedelta(days=1))
                    reasons.append("new_day_kline")
                if latest_finance_pub_date is not None and (
                    latest_feature_finance_pub_date is None
                    or latest_feature_finance_pub_date < latest_finance_pub_date
                ):
                    candidate_dates.append(latest_finance_pub_date)
                    reasons.append("new_finance")
                if not candidate_dates:
                    continue
                start_date = max(FEATURE_START_DATE, min(candidate_dates))
                reason = ",".join(reasons)

            if start_date > latest_day_date:
                continue
            sync_plan.append({"code": code, "start_date": start_date, "end_date": latest_day_date, "reason": reason})
            for item in reason.split(","):
                if item in reason_stats:
                    reason_stats[item] += 1

        log_event(
            "info",
            "feature sync plan built",
            backend="duckdb",
            target_codes=len(target_codes),
            planned_codes=len(sync_plan),
            full_refresh=reason_stats["full_refresh"],
            missing_feature=reason_stats["missing_feature"],
            new_day_kline=reason_stats["new_day_kline"],
            new_finance=reason_stats["new_finance"],
        )
        return sync_plan

    def _build_feature_frame(self, code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        day_kline_df = self._fetch_df(
            f"""
            select code, date, o, h, l, c, prec
            from {DUCKDB_DAY_KLINE_TABLE}
            where code = ? and date >= ? and date <= ? and tradestatus = true
            order by date
            """,
            [code, start_date, end_date],
        )
        return self._build_feature_frame_from_sources(code, start_date, end_date, day_kline_df, None)

    def _build_feature_frame_from_sources(
        self,
        code: str,
        start_date: datetime,
        end_date: datetime,
        day_kline_df: pd.DataFrame,
        finance_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if day_kline_df.empty:
            log_event(
                "warning",
                "feature source empty",
                code=code,
                source="day_kline",
                start_date=format_date(start_date),
                end_date=format_date(end_date),
            )
            return pd.DataFrame()
        day_kline_df["date"] = pd.to_datetime(day_kline_df["date"])

        if finance_df is None:
            finance_df = self._fetch_df(
                f"""
                select code, pubDate, statDate, totalShare, liqaShare
                from {DUCKDB_FINANCE_TABLE}
                where code = ? and pubDate >= ? and pubDate <= ?
                order by pubDate, statDate
                """,
                [code, FEATURE_START_DATE, end_date],
            )
        if finance_df.empty:
            log_event(
                "warning",
                "feature source empty",
                code=code,
                source="finance",
                start_date=format_date(FEATURE_START_DATE),
                end_date=format_date(end_date),
            )
            return pd.DataFrame()

        finance_df = finance_df.drop_duplicates(subset=["code", "pubDate", "statDate"], keep="last").copy()
        finance_df["pubDate"] = pd.to_datetime(finance_df["pubDate"], errors="coerce")
        finance_df["statDate"] = pd.to_datetime(finance_df["statDate"], errors="coerce")
        finance_df["totalShare"] = pd.to_numeric(finance_df["totalShare"], errors="coerce")
        finance_df["liqaShare"] = pd.to_numeric(finance_df["liqaShare"], errors="coerce")
        finance_df = finance_df.dropna(subset=["pubDate"])
        if finance_df.empty:
            log_event("warning", "feature source empty after cleanup", code=code, source="finance", end_date=format_date(end_date))
            return pd.DataFrame()

        df_merged = pd.merge_asof(
            prepare_frame_for_merge_asof(day_kline_df, on="date", by="code"),
            prepare_frame_for_merge_asof(finance_df, on="pubDate", by="code"),
            left_on="date",
            right_on="pubDate",
            by="code",
        ).copy()
        df_merged = df_merged[df_merged["pubDate"].notna()].copy()
        if df_merged.empty:
            log_event(
                "warning",
                "feature merge empty",
                code=code,
                stage="finance_asof",
                start_date=format_date(start_date),
                end_date=format_date(end_date),
            )
            return pd.DataFrame()

        df_merged = df_merged[df_merged["date"] >= max(start_date, FEATURE_START_DATE)].reset_index(drop=True)
        if df_merged.empty:
            return pd.DataFrame()

        df_merged.loc[:, "totalMV"] = pd.to_numeric(df_merged["prec"], errors="coerce") * df_merged["totalShare"]
        df_merged.loc[:, "liqaMV"] = pd.to_numeric(df_merged["prec"], errors="coerce") * df_merged["liqaShare"]
        df_merged.loc[:, "financePubDate"] = df_merged["pubDate"]
        df_merged.drop(
            ["o", "h", "l", "c", "prec", "totalShare", "liqaShare", "pubDate", "statDate"],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        log_event(
            "info",
            "feature frame built",
            backend="duckdb",
            code=code,
            rows=len(df_merged),
            start_date=format_date(df_merged["date"].min()),
            end_date=format_date(df_merged["date"].max()),
        )
        return df_merged

    def _build_feature_frames_batch(self, plan_items: list[dict[str, Any]]) -> list[pd.DataFrame]:
        if not plan_items:
            return []
        codes = [item["code"] for item in plan_items]
        min_start_date = min(item["start_date"] for item in plan_items)
        max_end_date = max(item["end_date"] for item in plan_items)
        placeholders = self._placeholders(codes)
        day_frame = self._fetch_df(
            f"""
            select code, date, o, h, l, c, prec
            from {DUCKDB_DAY_KLINE_TABLE}
            where code in ({placeholders}) and date >= ? and date <= ? and tradestatus = true
            order by code, date
            """,
            [*codes, min_start_date, max_end_date],
        )
        finance_frame = self._fetch_df(
            f"""
            select code, pubDate, statDate, totalShare, liqaShare
            from {DUCKDB_FINANCE_TABLE}
            where code in ({placeholders}) and pubDate >= ? and pubDate <= ?
            order by code, pubDate, statDate
            """,
            [*codes, FEATURE_START_DATE, max_end_date],
        )

        day_by_code = {code: group.copy() for code, group in day_frame.groupby("code")} if not day_frame.empty else {}
        finance_by_code = {code: group.copy() for code, group in finance_frame.groupby("code")} if not finance_frame.empty else {}
        frames: list[pd.DataFrame] = []
        for item in plan_items:
            code = item["code"]
            start_date = item["start_date"]
            end_date = item["end_date"]
            source_day = day_by_code.get(code, pd.DataFrame()).copy()
            if not source_day.empty:
                source_day["date"] = pd.to_datetime(source_day["date"])
                source_day = source_day[(source_day["date"] >= start_date) & (source_day["date"] <= end_date)].copy()
            source_finance = finance_by_code.get(code, pd.DataFrame()).copy()
            if not source_finance.empty:
                source_finance["pubDate"] = pd.to_datetime(source_finance["pubDate"], errors="coerce")
                source_finance = source_finance[source_finance["pubDate"] <= end_date].copy()
            feature_frame = self._build_feature_frame_from_sources(
                code,
                start_date,
                end_date,
                source_day,
                source_finance,
            )
            if not feature_frame.empty:
                frames.append(feature_frame)
        return frames

    def save_feature_daily(self, df, collection: str = DUCKDB_FEATURE_TABLE):
        if df is None or df.empty:
            return 0
        docs = feature_frame_to_records(df)
        docs, duplicate_count = dedupe_feature_docs_by_key(docs)
        frame = pd.DataFrame(docs)
        if frame.empty:
            return 0
        for column in ("date", "financePubDate"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        if duplicate_count:
            log_event(
                "warning",
                "feature write batch input deduped",
                backend="duckdb",
                duplicate_rows=duplicate_count,
                remaining_rows=len(frame),
            )
        upsert_frame(self.db_client, collection or DUCKDB_FEATURE_TABLE, frame, key_columns=("code", "date"))
        if (collection or DUCKDB_FEATURE_TABLE) == DUCKDB_FEATURE_TABLE and not self._feature_index_ensured:
            self.ensure_indexes()
        log_event("info", "feature write finished", backend="duckdb", collection=collection, written_rows=len(frame))
        return len(frame)

    def generate_feature(self, codes: list[str] | None = None, force_full_refresh: bool = False):
        log_event(
            "info",
            "feature sync start",
            backend="duckdb",
            input_codes=0 if codes is None else len(codes),
            force_full_refresh=force_full_refresh,
        )
        sync_plan = self.build_incremental_sync_plan(codes=codes, force_full_refresh=force_full_refresh)
        if not sync_plan:
            log_event("info", "feature sync skipped", backend="duckdb", reason="already_up_to_date")
            return {"planned_codes": 0, "updated_rows": 0}

        updated_rows = 0
        pending_frames: list[pd.DataFrame] = []
        pending_rows = 0

        def flush_pending_frames() -> int:
            nonlocal pending_frames, pending_rows
            if not pending_frames:
                return 0
            merged = pd.concat(pending_frames, ignore_index=True)
            log_event("info", "feature flush start", backend="duckdb", frame_count=len(pending_frames), rows=len(merged))
            written = self.save_feature_daily(merged, collection=DUCKDB_FEATURE_TABLE)
            pending_frames = []
            pending_rows = 0
            log_event("info", "feature flush done", backend="duckdb", written_rows=written)
            return written

        for batch_start in range(0, len(sync_plan), FEATURE_READ_CODE_BATCH_SIZE):
            batch_items = sync_plan[batch_start : batch_start + FEATURE_READ_CODE_BATCH_SIZE]
            log_event(
                "info",
                "feature sync batch start",
                backend="duckdb",
                batch=f"{batch_start // FEATURE_READ_CODE_BATCH_SIZE + 1}/{(len(sync_plan) + FEATURE_READ_CODE_BATCH_SIZE - 1) // FEATURE_READ_CODE_BATCH_SIZE}",
                codes=len(batch_items),
            )
            batch_frames = self._build_feature_frames_batch(batch_items)
            for feature_df in batch_frames:
                pending_frames.append(feature_df)
                pending_rows += len(feature_df)
                if pending_rows >= FEATURE_FLUSH_ROW_THRESHOLD:
                    updated_rows += flush_pending_frames()
            if batch_start == 0 or batch_start + FEATURE_READ_CODE_BATCH_SIZE >= len(sync_plan):
                log_event(
                    "info",
                    "feature sync progress",
                    backend="duckdb",
                    index=f"{min(batch_start + FEATURE_READ_CODE_BATCH_SIZE, len(sync_plan))}/{len(sync_plan)}",
                    rows=sum(len(frame) for frame in batch_frames),
                )

        updated_rows += flush_pending_frames()
        log_event("info", "feature sync completed", backend="duckdb", planned_codes=len(sync_plan), updated_rows=updated_rows)
        return {"planned_codes": len(sync_plan), "updated_rows": updated_rows}

    def find_liqaMV_by_date(self, find_date: datetime, limit: int = 20):
        frame = self._fetch_df(
            f"""
            select code, liqaMV
            from {DUCKDB_FEATURE_TABLE}
            where date = ? and liqaMV is not null
            order by liqaMV asc
            limit ?
            """,
            [find_date, limit],
        )
        return frame.to_dict("records")

    def get_stocks_below_liqaMV(self, start_date: datetime, end_date: datetime, cap_threshold: int = 8000000000.0):
        frame = self._fetch_df(
            f"""
            select code, date, liqaMV
            from {DUCKDB_FEATURE_TABLE}
            where date >= ? and date <= ? and liqaMV > ? and liqaMV <= ?
            order by date, code
            """,
            [start_date, end_date, 5e8, cap_threshold],
        )
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        return frame


Feature = DuckDBFeature


if __name__ == "__main__":
    feature_process = DuckDBFeature()
    feature_process.generate_feature()
