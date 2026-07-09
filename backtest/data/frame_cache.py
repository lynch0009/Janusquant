"""DataFrame 级别的本地 parquet 缓存。

用于回测研究中反复读取同一批 DuckDB 数据、反复计算同一批中间表的场景。
缓存命中规则非常直接：同一个 stage + 同一组关键参数会生成同一个 key；
缓存文件存在则读 parquet，不存在才调用 builder 访问数据库或重新计算。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from hashlib import sha1
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backtest.utils import normalize_internal_code
from backtest.utils.dataframe_cache import DataFrameCache

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


class FrameCache(DataFrameCache):
    """按 stage/key 管理 DataFrame 的内存 + 磁盘缓存。"""

    @staticmethod
    def codes_signature(codes: Sequence[str] | None) -> str:
        if codes is None:
            return "all"
        ordered = sorted({normalize_internal_code(code) for code in codes if pd.notna(code)})
        if not ordered:
            return "empty"
        return sha1("|".join(ordered).encode("utf-8")).hexdigest()[:20]

    def cache_key(self, stage: str, payload: dict[str, Any]) -> str:
        return self.key(stage, _json_ready(payload))

    def cache_file(self, stage: str, payload: dict[str, Any]) -> Path:
        path = self.path(stage, _json_ready(payload))
        if path is None:  # FrameCache always requires a root.
            raise ValueError("FrameCache root cannot be None")
        return path

    def load_or_build_frame(
        self,
        stage: str,
        payload: dict[str, Any],
        builder: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """优先读内存，其次读 parquet；都没有才执行 builder 并写入 parquet。"""

        return self.load_or_build(stage, _json_ready(payload), builder)

