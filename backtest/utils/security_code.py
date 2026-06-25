"""证券代码格式转换工具。"""

from __future__ import annotations


INDEX_CODE_PREFIXES = ("sh.000", "sz.399", "bj.899")


def normalize_internal_code(code: str) -> str:
    """统一成 Mongo 内部格式：sh.600000 / sz.000001 / bj.920118。"""

    value = str(code or "").strip()
    if not value:
        raise ValueError("stock code cannot be empty")

    if "." in value:
        left, right = value.split(".", 1)
        if left.isalpha():
            exchange = left.lower()
            symbol = right
        else:
            exchange = right.lower()
            symbol = left
    else:
        symbol = value[-6:]
        exchange = infer_exchange_from_symbol(symbol)

    symbol = str(symbol).strip().zfill(6)
    if exchange not in {"sh", "sz", "bj"} or not symbol.isdigit():
        raise ValueError(f"unsupported stock code format: {code}")
    return f"{exchange}.{symbol}"


def infer_exchange_from_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().zfill(6)
    if not value.isdigit():
        raise ValueError(f"unsupported stock symbol: {symbol}")
    if value.startswith(("60", "68", "69")):
        return "sh"
    if value.startswith(("00", "30")):
        return "sz"
    if value.startswith(("43", "83", "87", "88", "92")):
        return "bj"
    raise ValueError(f"unsupported stock symbol: {symbol}")


def plain_code(code: str) -> str:
    return normalize_internal_code(code).split(".", 1)[1]


def to_xt_code(code: str) -> str:
    """把内部代码转成 xtquant 格式：600000.SH。"""

    normalized = normalize_internal_code(code)
    exchange, symbol = normalized.split(".", 1)
    return f"{symbol}.{exchange.upper()}"


def to_akshare_em_symbol(code: str) -> str:
    """东方财富财报接口格式：SH600000。"""

    exchange, symbol = normalize_internal_code(code).split(".", 1)
    return f"{exchange.upper()}{symbol}"


def to_akshare_indicator_symbol(code: str) -> str:
    """东方财富财务指标接口格式：600000.SH。"""

    return to_xt_code(code)


def to_akshare_tx_symbol(code: str) -> str:
    """AkShare 腾讯接口格式：sh600000。"""

    exchange, symbol = normalize_internal_code(code).split(".", 1)
    return f"{exchange}{symbol}"


def split_code_list(raw_codes: str | None) -> list[str]:
    if raw_codes is None:
        return []
    result: list[str] = []
    for item in str(raw_codes).split(","):
        text = item.strip()
        if text:
            result.append(normalize_internal_code(text))
    return result


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


def is_index_code(code: str) -> bool:
    """判断内部证券代码是否属于当前支持的沪深北指数前缀。"""

    return str(code or "").strip().lower().startswith(INDEX_CODE_PREFIXES)
