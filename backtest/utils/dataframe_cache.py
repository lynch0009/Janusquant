"""Reusable in-memory and parquet DataFrame cache."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from hashlib import sha1
import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pandas as pd


def json_ready(value: Any) -> Any:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(json_ready(item) for item in value)
    return value


@dataclass
class DataFrameCache:
    root: Path | None
    version: str = "v1"
    _memory: dict[str, pd.DataFrame] = field(default_factory=dict)
    _stats: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    def __post_init__(self) -> None:
        self.root = Path(self.root) if self.root else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def key(self, stage: str, payload: dict[str, Any]) -> str:
        normalized = json_ready({"cache_version": self.version, "stage": stage, **payload})
        encoded = json.dumps(normalized, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
        return sha1(encoded).hexdigest()[:20]

    def path(self, stage: str, payload: dict[str, Any]) -> Path | None:
        if self.root is None:
            return None
        stage_dir = self.root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir / f"{self.key(stage, payload)}.parquet"

    def load_or_build(
        self,
        stage: str,
        payload: dict[str, Any],
        builder: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        cache_key = self.key(stage, payload)
        memory_key = f"{stage}:{cache_key}"
        cached = self._memory.get(memory_key)
        if cached is not None:
            self._stats[stage]["memory_hit"] += 1
            return cached.copy()

        path = self.path(stage, payload)
        if path is not None and path.exists():
            try:
                frame = pd.read_parquet(path)
            except Exception:
                self._stats[stage]["rebuild"] += 1
                path.unlink(missing_ok=True)
            else:
                self._memory[memory_key] = frame
                self._stats[stage]["disk_hit"] += 1
                return frame.copy()

        self._stats[stage]["miss"] += 1
        frame = builder()
        if frame is None:
            frame = pd.DataFrame()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"cache builder for {stage} must return DataFrame, got {type(frame)!r}")
        if path is not None:
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                frame.to_parquet(temporary, index=False)
                pd.read_parquet(temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            self._stats[stage]["write"] += 1
        self._memory[memory_key] = frame
        return frame.copy()

    def summary(self) -> dict[str, dict[str, int]]:
        return {stage: dict(values) for stage, values in sorted(self._stats.items())}
