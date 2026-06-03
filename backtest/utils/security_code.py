"""证券代码格式转换工具。"""

from __future__ import annotations


def normalize_internal_code(code: str) -> str:
    """统一成 Mongo 内部格式：sh.600000 / sz.000001 / bj.920118。"""

    value = str(code or "").strip()
    if not value:
        raise ValueError("stock code cannot be empty")
    if "." not in value:
        raise ValueError(f"unsupported stock code format: {code}")

    left, right = value.split(".", 1)
    if left.isalpha():
        exchange = left.lower()
        symbol = right
    else:
        exchange = right.lower()
        symbol = left
    return f"{exchange}.{symbol.zfill(6)}"


def to_xt_code(code: str) -> str:
    """把内部代码转成 xtquant 格式：600000.SH。"""

    normalized = normalize_internal_code(code)
    exchange, symbol = normalized.split(".", 1)
    return f"{symbol}.{exchange.upper()}"


def is_supported_a_stock_xt_code(xt_code: str) -> bool:
    """判断 xtquant 代码是否属于当前支持的沪深 A 股范围。"""

    value = str(xt_code or "").strip().upper()
    if "." not in value:
        return False
    symbol, exchange = value.split(".", 1)
    if len(symbol) != 6 or not symbol.isdigit():
        return False
    if exchange == "SH":
        return symbol.startswith(("60", "68", "69"))
    if exchange == "SZ":
        return symbol.startswith(("00", "30"))
    return False


def is_bj_code(code: str) -> bool:
    return normalize_internal_code(code).startswith("bj.")


def is_hs_a_share_code(code: str) -> bool:
    normalized = normalize_internal_code(code)
    exchange, symbol = normalized.split(".", 1)
    if exchange == "sh":
        return symbol.startswith(("60", "68", "69"))
    if exchange == "sz":
        return symbol.startswith(("00", "30"))
    return False
