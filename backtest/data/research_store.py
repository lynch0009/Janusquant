"""研究层日线历史缓存。

用于复用“策略选股研究”和“收盘确认类风控”所需的日线历史数据，避免同一口径反复查库。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import pandas as pd

from backtest.utils import normalize_internal_code


@dataclass
class CachedDailyHistory:
    """一份已加载到内存的研究日线数据。"""

    start_date: datetime
    end_date: datetime
    price_mode: str
    include_stopped: bool
    fields: tuple[str, ...]
    codes: set[str]
    frame: pd.DataFrame

    def covers(
        self,
        *,
        start_date: datetime,
        end_date: datetime,
        codes: Sequence[str],
        fields: Sequence[str],
        price_mode: str,
        include_stopped: bool,
    ) -> bool:
        """判断当前缓存是否完整覆盖一次新的研究请求。"""

        requested_codes = {normalize_internal_code(code) for code in codes}
        requested_fields = set(fields) | {"code", "trade_date"}
        return (
            self.price_mode == price_mode
            and self.include_stopped == include_stopped
            and self.start_date <= start_date
            and self.end_date >= end_date
            and self.codes.issuperset(requested_codes)
            and set(self.fields).issuperset(requested_fields)
        )


@dataclass
class CachedVisibleFinance:
    """一份已缓存到内存的“截至某日可见”的财报数据。"""

    as_of: datetime
    start_pub_date: datetime | None
    fields: tuple[str, ...]
    codes: set[str]
    frame: pd.DataFrame

    def covers(
        self,
        *,
        as_of: datetime,
        start_pub_date: datetime | None,
        codes: Sequence[str],
        fields: Sequence[str],
    ) -> bool:
        requested_codes = {normalize_internal_code(code) for code in codes}
        requested_fields = set(fields) | {"code", "pubDate", "statDate"}
        return (
            self.as_of == as_of
            and (
                self.start_pub_date == start_pub_date
                or (self.start_pub_date is None and start_pub_date is None)
            )
            and self.codes.issuperset(requested_codes)
            and set(self.fields).issuperset(requested_fields)
        )


@dataclass
class CachedFinanceReports:
    """???????????????????"""

    start_pub_date: datetime | None
    end_pub_date: datetime | None
    fields: tuple[str, ...]
    codes: set[str]
    frame: pd.DataFrame

    def covers(
        self,
        *,
        start_pub_date: datetime | None,
        end_pub_date: datetime | None,
        codes: Sequence[str],
        fields: Sequence[str],
    ) -> bool:
        requested_codes = {normalize_internal_code(code) for code in codes}
        requested_fields = set(fields) | {"code", "pubDate", "statDate"}
        start_ok = self.start_pub_date is None or (start_pub_date is not None and self.start_pub_date <= start_pub_date)
        end_ok = self.end_pub_date is None or (end_pub_date is not None and self.end_pub_date >= end_pub_date)
        return start_ok and end_ok and self.codes.issuperset(requested_codes) and set(self.fields).issuperset(requested_fields)


@dataclass
class CachedMinerviniFundamentalFeatures:
    """一份已缓存到内存的 Minervini 基本面特征时间线。"""

    start_pub_date: datetime | None
    end_pub_date: datetime | None
    feature_version: str
    fields: tuple[str, ...]
    codes: set[str]
    frame: pd.DataFrame

    def covers(
        self,
        *,
        start_pub_date: datetime | None,
        end_pub_date: datetime | None,
        feature_version: str,
        codes: Sequence[str],
        fields: Sequence[str],
    ) -> bool:
        requested_codes = {normalize_internal_code(code) for code in codes}
        requested_fields = set(fields) | {"code", "pubDate", "statDate", "featureVersion"}
        start_ok = self.start_pub_date is None or (start_pub_date is not None and self.start_pub_date <= start_pub_date)
        end_ok = self.end_pub_date is None or (end_pub_date is not None and self.end_pub_date >= end_pub_date)
        return (
            self.feature_version == feature_version
            and start_ok
            and end_ok
            and self.codes.issuperset(requested_codes)
            and set(self.fields).issuperset(requested_fields)
        )


class ResearchDailyHistoryStore:
    """共享研究日线缓存。

    设计目标：
    1. 同一份研究价历史只查库一次
    2. 后续规则若只需要其子集字段/子集股票/更短区间，直接从内存切片
    3. 保持接口足够简单，方便策略层和风控层共用
    """

    def __init__(self, data_portal):
        self.data_portal = data_portal
        self._datasets: list[CachedDailyHistory] = []
        self._visible_finance_datasets: list[CachedVisibleFinance] = []
        self._finance_report_datasets: list[CachedFinanceReports] = []
        self._minervini_fundamental_datasets: list[CachedMinerviniFundamentalFeatures] = []

    @staticmethod
    def _normalize_fields(fields: Sequence[str] | None) -> tuple[str, ...]:
        requested_fields = list(fields or [])
        normalized_fields: list[str] = []
        for field in requested_fields:
            normalized_field = "trade_date" if field == "date" else field
            if normalized_field not in normalized_fields:
                normalized_fields.append(normalized_field)
        if "code" not in normalized_fields:
            normalized_fields.insert(0, "code")
        if "trade_date" not in normalized_fields:
            normalized_fields.insert(1, "trade_date")
        return tuple(normalized_fields)

    @staticmethod
    def _filter_frame(
        frame: pd.DataFrame,
        *,
        start_date: datetime,
        end_date: datetime,
        codes: Sequence[str],
        fields: Sequence[str],
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=list(fields))

        filtered = frame[
            (frame["trade_date"] >= pd.Timestamp(start_date))
            & (frame["trade_date"] < pd.Timestamp(end_date))
            & (frame["code"].isin([normalize_internal_code(code) for code in codes]))
        ].copy()
        available_fields = [field for field in fields if field in filtered.columns]
        return filtered[available_fields].sort_values(["code", "trade_date"]).reset_index(drop=True)

    def load_daily_history(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        codes: Sequence[str],
        fields: Sequence[str],
        price_mode: str = "qfq",
        include_stopped: bool = False,
        batch_size: int | None = 1000,
    ) -> pd.DataFrame:
        """读取研究日线，并优先复用已缓存的数据集。"""

        normalized_codes = sorted({normalize_internal_code(code) for code in codes})
        normalized_fields = self._normalize_fields(fields)
        if not normalized_codes:
            return pd.DataFrame(columns=list(normalized_fields))

        for dataset in self._datasets:
            if dataset.covers(
                start_date=start_date,
                end_date=end_date,
                codes=normalized_codes,
                fields=normalized_fields,
                price_mode=price_mode,
                include_stopped=include_stopped,
            ):
                return self._filter_frame(
                    dataset.frame,
                    start_date=start_date,
                    end_date=end_date,
                    codes=normalized_codes,
                    fields=normalized_fields,
                )

        frame = self.data_portal.get_daily_history(
            start_date,
            end_date,
            codes=normalized_codes,
            fields=list(normalized_fields),
            include_stopped=include_stopped,
            batch_size=batch_size,
            price_mode=price_mode,
        )
        if not frame.empty:
            frame = frame.sort_values(["code", "trade_date"]).reset_index(drop=True)

        self._datasets.append(
            CachedDailyHistory(
                start_date=start_date,
                end_date=end_date,
                price_mode=price_mode,
                include_stopped=include_stopped,
                fields=normalized_fields,
                codes=set(normalized_codes),
                frame=frame,
            )
        )
        return self._filter_frame(
            frame,
            start_date=start_date,
            end_date=end_date,
            codes=normalized_codes,
            fields=normalized_fields,
        )

    @staticmethod
    def _normalize_finance_fields(fields: Sequence[str] | None) -> tuple[str, ...]:
        requested_fields = list(fields or [])
        normalized_fields: list[str] = []
        for field in requested_fields:
            if field not in normalized_fields:
                normalized_fields.append(field)
        for field in ("code", "pubDate", "statDate"):
            if field not in normalized_fields:
                normalized_fields.insert(0 if field == "code" else len(normalized_fields), field)
        return tuple(normalized_fields)

    @staticmethod
    def _filter_visible_finance_frame(
        frame: pd.DataFrame,
        *,
        codes: Sequence[str],
        fields: Sequence[str],
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=list(fields))

        filtered = frame[frame["code"].isin([normalize_internal_code(code) for code in codes])].copy()
        available_fields = [field for field in fields if field in filtered.columns]
        sort_fields = [field for field in ("code", "pubDate", "statDate") if field in filtered.columns]
        return filtered[available_fields].sort_values(sort_fields).reset_index(drop=True)

    @staticmethod
    def _filter_finance_reports_frame(
        frame: pd.DataFrame,
        *,
        start_pub_date: datetime | None,
        end_pub_date: datetime | None,
        codes: Sequence[str],
        fields: Sequence[str],
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=list(fields))

        filtered = frame[frame["code"].isin([normalize_internal_code(code) for code in codes])].copy()
        if start_pub_date is not None:
            filtered = filtered[filtered["pubDate"] >= pd.Timestamp(start_pub_date)]
        if end_pub_date is not None:
            filtered = filtered[filtered["pubDate"] <= pd.Timestamp(end_pub_date)]
        available_fields = [field for field in fields if field in filtered.columns]
        sort_fields = [field for field in ("code", "pubDate", "statDate") if field in filtered.columns]
        return filtered[available_fields].sort_values(sort_fields).reset_index(drop=True)

    @staticmethod
    def _filter_minervini_fundamental_frame(
        frame: pd.DataFrame,
        *,
        start_pub_date: datetime | None,
        end_pub_date: datetime | None,
        codes: Sequence[str],
        fields: Sequence[str],
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=list(fields))

        filtered = frame[frame["code"].isin([normalize_internal_code(code) for code in codes])].copy()
        if start_pub_date is not None:
            filtered = filtered[filtered["pubDate"] >= pd.Timestamp(start_pub_date)]
        if end_pub_date is not None:
            filtered = filtered[filtered["pubDate"] <= pd.Timestamp(end_pub_date)]
        available_fields = [field for field in fields if field in filtered.columns]
        sort_fields = [field for field in ("code", "pubDate", "statDate") if field in filtered.columns]
        return filtered[available_fields].sort_values(sort_fields).reset_index(drop=True)

    def load_visible_finance(
        self,
        as_of: datetime,
        *,
        codes: Sequence[str],
        fields: Sequence[str],
        start_pub_date: datetime | None = None,
    ) -> pd.DataFrame:
        """读取截至指定日期可见的财报，并优先复用缓存。"""

        normalized_codes = sorted({normalize_internal_code(code) for code in codes})
        normalized_fields = self._normalize_finance_fields(fields)
        if not normalized_codes:
            return pd.DataFrame(columns=list(normalized_fields))

        normalized_as_of = pd.Timestamp(as_of).to_pydatetime()
        normalized_start_pub_date = (
            pd.Timestamp(start_pub_date).to_pydatetime() if start_pub_date is not None else None
        )

        for dataset in self._visible_finance_datasets:
            if dataset.covers(
                as_of=normalized_as_of,
                start_pub_date=normalized_start_pub_date,
                codes=normalized_codes,
                fields=normalized_fields,
            ):
                return self._filter_visible_finance_frame(
                    dataset.frame,
                    codes=normalized_codes,
                    fields=normalized_fields,
                )

        frame = self.data_portal.get_visible_finance_slice(
            normalized_as_of,
            codes=normalized_codes,
            fields=list(normalized_fields),
            start_pub_date=normalized_start_pub_date,
        )
        if not frame.empty:
            if "pubDate" in frame.columns:
                frame["pubDate"] = pd.to_datetime(frame["pubDate"])
            if "statDate" in frame.columns:
                frame["statDate"] = pd.to_datetime(frame["statDate"])
            frame = frame.sort_values(["code", "pubDate", "statDate"]).reset_index(drop=True)

        self._visible_finance_datasets.append(
            CachedVisibleFinance(
                as_of=normalized_as_of,
                start_pub_date=normalized_start_pub_date,
                fields=normalized_fields,
                codes=set(normalized_codes),
                frame=frame,
            )
        )
        return self._filter_visible_finance_frame(
            frame,
            codes=normalized_codes,
            fields=normalized_fields,
        )


    def load_finance_reports(
        self,
        start_pub_date: datetime | None,
        end_pub_date: datetime | None,
        *,
        codes: Sequence[str],
        fields: Sequence[str],
    ) -> pd.DataFrame:
        """???????????????????"""

        normalized_codes = sorted({normalize_internal_code(code) for code in codes})
        normalized_fields = self._normalize_finance_fields(fields)
        if not normalized_codes:
            return pd.DataFrame(columns=list(normalized_fields))

        normalized_start_pub_date = (
            pd.Timestamp(start_pub_date).to_pydatetime() if start_pub_date is not None else None
        )
        normalized_end_pub_date = (
            pd.Timestamp(end_pub_date).to_pydatetime() if end_pub_date is not None else None
        )

        for dataset in self._finance_report_datasets:
            if dataset.covers(
                start_pub_date=normalized_start_pub_date,
                end_pub_date=normalized_end_pub_date,
                codes=normalized_codes,
                fields=normalized_fields,
            ):
                return self._filter_finance_reports_frame(
                    dataset.frame,
                    start_pub_date=normalized_start_pub_date,
                    end_pub_date=normalized_end_pub_date,
                    codes=normalized_codes,
                    fields=normalized_fields,
                )

        frame = self.data_portal.get_finance_reports(
            codes=normalized_codes,
            start_pub_date=normalized_start_pub_date,
            end_pub_date=normalized_end_pub_date,
            fields=list(normalized_fields),
        )
        if not frame.empty:
            if "pubDate" in frame.columns:
                frame["pubDate"] = pd.to_datetime(frame["pubDate"])
            if "statDate" in frame.columns:
                frame["statDate"] = pd.to_datetime(frame["statDate"])
            frame = frame.sort_values(["code", "pubDate", "statDate"]).reset_index(drop=True)

        self._finance_report_datasets.append(
            CachedFinanceReports(
                start_pub_date=normalized_start_pub_date,
                end_pub_date=normalized_end_pub_date,
                fields=normalized_fields,
                codes=set(normalized_codes),
                frame=frame,
            )
        )
        return self._filter_finance_reports_frame(
            frame,
            start_pub_date=normalized_start_pub_date,
            end_pub_date=normalized_end_pub_date,
            codes=normalized_codes,
            fields=normalized_fields,
        )

    def load_minervini_fundamental_features(
        self,
        start_pub_date: datetime | None,
        end_pub_date: datetime | None,
        *,
        codes: Sequence[str],
        fields: Sequence[str],
        feature_version: str = "minervini_fundamental_v1",
    ) -> pd.DataFrame:
        """读取 Minervini 基本面特征时间线，并优先复用缓存。"""

        normalized_codes = sorted({normalize_internal_code(code) for code in codes})
        normalized_fields = self._normalize_finance_fields(fields)
        if "featureVersion" not in normalized_fields:
            normalized_fields = tuple(list(normalized_fields) + ["featureVersion"])
        if not normalized_codes:
            return pd.DataFrame(columns=list(normalized_fields))

        normalized_start_pub_date = (
            pd.Timestamp(start_pub_date).to_pydatetime() if start_pub_date is not None else None
        )
        normalized_end_pub_date = (
            pd.Timestamp(end_pub_date).to_pydatetime() if end_pub_date is not None else None
        )

        for dataset in self._minervini_fundamental_datasets:
            if dataset.covers(
                start_pub_date=normalized_start_pub_date,
                end_pub_date=normalized_end_pub_date,
                feature_version=feature_version,
                codes=normalized_codes,
                fields=normalized_fields,
            ):
                return self._filter_minervini_fundamental_frame(
                    dataset.frame,
                    start_pub_date=normalized_start_pub_date,
                    end_pub_date=normalized_end_pub_date,
                    codes=normalized_codes,
                    fields=normalized_fields,
                )

        frame = self.data_portal.get_minervini_fundamental_features(
            codes=normalized_codes,
            start_pub_date=normalized_start_pub_date,
            end_pub_date=normalized_end_pub_date,
            fields=list(normalized_fields),
            feature_version=feature_version,
        )
        if not frame.empty:
            for field in ("pubDate", "statDate", "revisionDate", "computedAt"):
                if field in frame.columns:
                    frame[field] = pd.to_datetime(frame[field])
            frame = frame.sort_values(["code", "pubDate", "statDate"]).reset_index(drop=True)

        self._minervini_fundamental_datasets.append(
            CachedMinerviniFundamentalFeatures(
                start_pub_date=normalized_start_pub_date,
                end_pub_date=normalized_end_pub_date,
                feature_version=feature_version,
                fields=normalized_fields,
                codes=set(normalized_codes),
                frame=frame,
            )
        )
        return self._filter_minervini_fundamental_frame(
            frame,
            start_pub_date=normalized_start_pub_date,
            end_pub_date=normalized_end_pub_date,
            codes=normalized_codes,
            fields=normalized_fields,
        )
