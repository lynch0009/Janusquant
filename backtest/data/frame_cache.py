"""DataFrame 级别的本地 parquet 缓存。

用于回测研究中反复读取同一批 Mongo 数据、反复计算同一批中间表的场景。
缓存命中规则非常直接：同一个 stage + 同一组关键参数会生成同一个 key；
缓存文件存在则读 parquet，不存在才调用 builder 访问数据库或重新计算。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from hashlib import sha1
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backtest.db import normalize_code
from backtest.data.portal import MongoDataPortal


def _json_ready(value: Any) -> Any:
    """把 cache payload 标准化成稳定 JSON 可序列化对象。"""

    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


@dataclass
class FrameCache:
    """按 stage/key 管理 DataFrame 的内存 + 磁盘缓存。"""

    root: Path
    version: str = "v1"
    _memory_cache: dict[str, pd.DataFrame] = field(default_factory=dict)
    _stats: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def codes_signature(codes: Sequence[str] | None) -> str:
        if codes is None:
            return "all"
        ordered = sorted({normalize_code(code) for code in codes if pd.notna(code)})
        if not ordered:
            return "empty"
        return sha1("|".join(ordered).encode("utf-8")).hexdigest()[:20]

    def cache_key(self, stage: str, payload: dict[str, Any]) -> str:
        normalized_payload = _json_ready({"cache_version": self.version, "stage": stage, **payload})
        encoded = json.dumps(normalized_payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
        return sha1(encoded).hexdigest()[:20]

    def cache_file(self, stage: str, payload: dict[str, Any]) -> Path:
        stage_dir = self.root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir / f"{self.cache_key(stage, payload)}.parquet"

    def load_or_build_frame(
        self,
        stage: str,
        payload: dict[str, Any],
        builder: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """优先读内存，其次读 parquet；都没有才执行 builder 并写入 parquet。"""

        key = self.cache_key(stage, payload)
        memory_key = f"{stage}:{key}"
        cached = self._memory_cache.get(memory_key)
        if cached is not None:
            self._stats[stage]["memory_hit"] += 1
            return cached.copy()

        path = self.cache_file(stage, payload)
        if path.exists():
            frame = pd.read_parquet(path)
            self._memory_cache[memory_key] = frame
            self._stats[stage]["disk_hit"] += 1
            return frame.copy()

        self._stats[stage]["miss"] += 1
        frame = builder()
        if frame is None:
            frame = pd.DataFrame()
        frame.to_parquet(path, index=False)
        self._memory_cache[memory_key] = frame
        self._stats[stage]["write"] += 1
        return frame.copy()

    def summary(self) -> dict[str, dict[str, int]]:
        return {stage: dict(stats) for stage, stats in sorted(self._stats.items())}


class CachedMongoDataPortal(MongoDataPortal):
    """带 parquet 缓存的 MongoDataPortal。

    只缓存回测准备阶段的大块读库结果；执行阶段的每日行情快照暂不缓存，
    避免生成大量碎片文件。
    """

    def __init__(self, db_client, *, frame_cache: FrameCache, calendar_code: str = "sh.000001"):
        super().__init__(db_client, calendar_code=calendar_code)
        self.frame_cache = frame_cache

    def get_trade_calendar(self, start_date: datetime, end_date: datetime) -> list[datetime]:
        payload = {
            "calendar_code": self.calendar_code,
            "start_date": start_date,
            "end_date": end_date,
        }

        def builder() -> pd.DataFrame:
            dates = super(CachedMongoDataPortal, self).get_trade_calendar(start_date, end_date)
            return pd.DataFrame({"trade_date": pd.to_datetime(dates)})

        frame = self.frame_cache.load_or_build_frame("trade_calendar", payload, builder)
        if frame.empty:
            return []
        return sorted(pd.to_datetime(frame["trade_date"]).to_list())

    def get_feature_history(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        fields: Sequence[str] | None = None,
        filters: dict | None = None,
    ) -> pd.DataFrame:
        requested_fields = list(sorted(set((fields or []) + ["code", "date"])))
        payload = {
            "start_date": start_date,
            "end_date": end_date,
            "fields": requested_fields,
            "filters": filters or {},
        }
        return self.frame_cache.load_or_build_frame(
            "feature_history",
            payload,
            lambda: super(CachedMongoDataPortal, self).get_feature_history(
                start_date,
                end_date,
                fields=fields,
                filters=filters,
            ),
        )

    def get_stock_basic(
        self,
        codes: Sequence[str],
        *,
        fields: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame()
        payload = {
            "codes_signature": self.frame_cache.codes_signature(codes),
            "fields": list(fields or []),
        }
        return self.frame_cache.load_or_build_frame(
            "stock_basic",
            payload,
            lambda: super(CachedMongoDataPortal, self).get_stock_basic(codes, fields=fields),
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
            "price_mode": self.feature_service.normalize_price_mode(price_mode),
        }
        return self.frame_cache.load_or_build_frame(
            "daily_history",
            payload,
            lambda: super(CachedMongoDataPortal, self).get_daily_history(
                start_date,
                end_date,
                codes=codes,
                fields=fields,
                include_stopped=include_stopped,
                batch_size=batch_size,
                price_mode=price_mode,
            ),
        )

    def get_market_amount_history(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        code_prefixes: Sequence[str] | None = None,
        include_stopped: bool = False,
    ) -> pd.DataFrame:
        payload = {
            "start_date": start_date,
            "end_date": end_date,
            "code_prefixes": tuple(code_prefixes or ("sh.60", "sh.68", "sz.00", "sz.30")),
            "include_stopped": include_stopped,
        }
        return self.frame_cache.load_or_build_frame(
            "market_amount_history",
            payload,
            lambda: super(CachedMongoDataPortal, self).get_market_amount_history(
                start_date,
                end_date,
                code_prefixes=code_prefixes,
                include_stopped=include_stopped,
            ),
        )

    def get_corporate_action_events(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        codes: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        payload = {
            "start_date": start_date,
            "end_date": end_date,
            "codes_signature": self.frame_cache.codes_signature(codes),
        }
        return self.frame_cache.load_or_build_frame(
            "corporate_action_events",
            payload,
            lambda: super(CachedMongoDataPortal, self).get_corporate_action_events(
                start_date,
                end_date,
                codes=codes,
            ),
        )
