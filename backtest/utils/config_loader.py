"""轻量配置加载工具。"""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path


def project_root_from(anchor_file: str) -> Path:
    """根据模块文件路径推导项目根目录。"""

    return Path(anchor_file).resolve().parents[2]


def load_ini_section(path: str | Path, section: str) -> dict[str, str]:
    """读取 INI 配置中的某个 section，不存在时返回空字典。"""

    parser = ConfigParser()
    parser.read(Path(path), encoding="utf-8")
    if not parser.has_section(section):
        return {}
    return {key: value for key, value in parser.items(section)}


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def parse_bool(value: str | bool | None, *, default: bool = False, strict: bool = False) -> bool:
    """Parse common string boolean values."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    if strict:
        raise ValueError(f"invalid boolean value: {value!r}")
    return default


def parse_bool_strict(value: str | bool | None, *, default: bool = False) -> bool:
    """Parse a boolean config value and reject unknown non-empty strings."""

    return parse_bool(value, default=default, strict=True)
