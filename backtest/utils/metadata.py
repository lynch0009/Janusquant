"""metadata 读写相关的共享工具。"""

from __future__ import annotations

from typing import Any, Mapping


def copy_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """安全复制 metadata，统一处理 None。"""

    return dict(metadata or {})


def merge_metadata(*metadata_items: Mapping[str, Any] | None) -> dict[str, Any]:
    """按顺序合并多份 metadata，后者覆盖前者。"""

    merged: dict[str, Any] = {}
    for item in metadata_items:
        if item:
            merged.update(dict(item))
    return merged
