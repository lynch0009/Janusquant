from __future__ import annotations

import contextlib
import io
import socket
import time
from datetime import datetime
from typing import Any, Callable

import baostock as bs
import pandas as pd

from backtest.utils import to_trade_datetime


DEFAULT_RETRY_TIMES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 20.0
BAOSTOCK_SOCKET_TIMEOUT_SECONDS = 30.0

BAOSTOCK_ACCOUNT_ERROR_CODES = frozenset(
    {
        "10001002",  # Username or password error.
        "10001005",  # Account login count limit.
        "10001006",  # Insufficient permission.
        "10001007",  # Account activation required.
        "10001008",  # Empty username.
        "10001009",  # Empty password.
        "10001011",  # Blacklisted user.
    }
)
BAOSTOCK_ACCOUNT_ERROR_MARKERS = (
    "用户名或密码错误",
    "账号登陆数达到上限",
    "账号登录数达到上限",
    "用户权限不足",
    "需要登录激活",
    "用户名为空",
    "密码为空",
    "黑名单用户",
)


class BaostockQueryError(RuntimeError):
    def __init__(self, message: str, retries_used: int) -> None:
        super().__init__(message)
        self.retries_used = retries_used


class BaostockAccountError(RuntimeError):
    """A permanent Baostock account-state error that must not be retried."""

    def __init__(self, error_code: str | None, error_message: str) -> None:
        self.error_code = str(error_code or "").strip()
        self.error_message = str(error_message or "").strip() or "unknown account error"
        display_code = self.error_code or "unknown"
        super().__init__(f"error_code={display_code} message={self.error_message}")


def is_baostock_account_error(error_code: Any, error_message: Any) -> bool:
    code = str(error_code or "").strip()
    message = str(error_message or "").strip()
    return code in BAOSTOCK_ACCOUNT_ERROR_CODES or any(
        marker in message for marker in BAOSTOCK_ACCOUNT_ERROR_MARKERS
    )


def _raise_if_account_error(error_code: Any, error_message: Any) -> None:
    if is_baostock_account_error(error_code, error_message):
        raise BaostockAccountError(str(error_code or "").strip() or None, str(error_message or ""))


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _call_quietly(func: Callable[..., Any], *args, quiet: bool, **kwargs):
    if not quiet:
        return func(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


def _set_active_socket_timeout() -> None:
    """Apply the fixed timeout to Baostock's process-global active socket."""

    from baostock.common import context as baostock_context

    active_socket = getattr(baostock_context, "default_socket", None)
    if active_socket is not None and hasattr(active_socket, "settimeout"):
        active_socket.settimeout(BAOSTOCK_SOCKET_TIMEOUT_SECONDS)


def _login_once(*, quiet: bool) -> Any:
    """Create Baostock's socket with a timeout without leaking a global default."""

    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(BAOSTOCK_SOCKET_TIMEOUT_SECONDS)
    try:
        return _call_quietly(bs.login, quiet=quiet)
    finally:
        socket.setdefaulttimeout(previous_timeout)

def safe_logout(*, quiet: bool = True) -> None:
    with contextlib.suppress(Exception):
        _set_active_socket_timeout()
        _call_quietly(bs.logout, quiet=quiet)


def login_with_retry(
    *,
    retry_times: int = DEFAULT_RETRY_TIMES,
    retry_sleep_seconds: float = DEFAULT_RETRY_SLEEP_SECONDS,
    quiet: bool = False,
    progress: Callable[[str], None] | None = _print_progress,
) -> None:
    last_error = ""
    retry_times = max(1, retry_times)

    for attempt in range(1, retry_times + 1):
        try:
            login_result = _login_once(quiet=quiet)
            if login_result.error_code == "0":
                _set_active_socket_timeout()
                return
            _raise_if_account_error(login_result.error_code, login_result.error_msg)
            last_error = login_result.error_msg or f"error_code={login_result.error_code}"
        except BaostockAccountError:
            raise
        except Exception as exc:
            _raise_if_account_error(None, str(exc))
            last_error = str(exc)

        if attempt < retry_times:
            if progress is not None:
                progress(
                    "baostock login failed, "
                    f"socket_timeout={BAOSTOCK_SOCKET_TIMEOUT_SECONDS:g}s, "
                    f"retry after {retry_sleep_seconds}s: {last_error}"
                )
            time.sleep(max(0.0, retry_sleep_seconds))

    raise RuntimeError(f"baostock login failed after {retry_times} retries: {last_error}")


def fetch_query_dataframe(
    query_func: Callable[..., Any],
    *query_args,
    retry_times: int = DEFAULT_RETRY_TIMES,
    retry_sleep_seconds: float = DEFAULT_RETRY_SLEEP_SECONDS,
    relogin_sleep_seconds: float | None = None,
    quiet: bool = False,
    progress: Callable[[str], None] | None = _print_progress,
    context: str = "",
    **query_kwargs,
) -> tuple[pd.DataFrame, int]:
    last_error = ""
    retry_times = max(1, retry_times)
    label = context or getattr(query_func, "__name__", "baostock query")
    sleep_seconds = retry_sleep_seconds if relogin_sleep_seconds is None else relogin_sleep_seconds

    for attempt in range(1, retry_times + 1):
        try:
            _set_active_socket_timeout()
            rs = _call_quietly(query_func, *query_args, quiet=quiet, **query_kwargs)
            if rs.error_code != "0":
                _raise_if_account_error(rs.error_code, rs.error_msg)
                last_error = rs.error_msg or f"error_code={rs.error_code}"
            else:
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    return pd.DataFrame(columns=rs.fields), attempt - 1
                return pd.DataFrame(rows, columns=rs.fields), attempt - 1
        except BaostockAccountError:
            raise
        except Exception as exc:
            _raise_if_account_error(None, str(exc))
            last_error = str(exc)

        if attempt < retry_times:
            if progress is not None:
                progress(
                    f"{label} failed, socket_timeout={BAOSTOCK_SOCKET_TIMEOUT_SECONDS:g}s, "
                    f"retry after {sleep_seconds}s: {last_error}"
                )
            safe_logout(quiet=quiet)
            time.sleep(max(0.0, sleep_seconds))
            login_with_retry(
                retry_times=retry_times,
                retry_sleep_seconds=sleep_seconds,
                quiet=quiet,
                progress=progress,
            )

    raise BaostockQueryError(f"{label} failed after {retry_times} retries: {last_error}", retry_times - 1)


def fetch_trade_calendar(start_date: datetime, end_date: datetime) -> list[datetime]:
    frame, _ = fetch_query_dataframe(
        bs.query_trade_dates,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        context=f"query_trade_dates {start_date:%Y-%m-%d} ~ {end_date:%Y-%m-%d}",
    )
    if frame.empty:
        return []
    return [
        to_trade_datetime(value)
        for value in frame.loc[frame["is_trading_day"].astype(str) == "1", "calendar_date"].tolist()
    ]
