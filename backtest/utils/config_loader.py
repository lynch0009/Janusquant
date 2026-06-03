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


def parse_bool(value: str | bool | None, *, default: bool = False) -> bool:
    """把常见字符串布尔值统一解析成 bool。"""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
