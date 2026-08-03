from __future__ import annotations

import contextlib
import io
import time
from datetime import datetime
from typing import Any, Callable

import baostock as bs
import pandas as pd

from backtest.utils import to_trade_datetime


DEFAULT_RETRY_TIMES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 20.0


class BaostockQueryError(RuntimeError):
    def __init__(self, message: str, retries_used: int) -> None:
        super().__init__(message)
        self.retries_used = retries_used


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _call_quietly(func: Callable[..., Any], *args, quiet: bool, **kwargs):
    if not quiet:
        return func(*args, **kwargs)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


def safe_logout(*, quiet: bool = True) -> None:
    with contextlib.suppress(Exception):
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
            login_result = _call_quietly(bs.login, quiet=quiet)
            if login_result.error_code == "0":
                return
            last_error = login_result.error_msg or f"error_code={login_result.error_code}"
        except Exception as exc:
            last_error = str(exc)

        if attempt < retry_times:
            if progress is not None:
                progress(f"baostock login failed, retry after {retry_sleep_seconds}s: {last_error}")
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
            rs = _call_quietly(query_func, *query_args, quiet=quiet, **query_kwargs)
            if rs.error_code != "0":
                last_error = rs.error_msg or f"error_code={rs.error_code}"
            else:
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    return pd.DataFrame(columns=rs.fields), attempt - 1
                return pd.DataFrame(rows, columns=rs.fields), attempt - 1
        except Exception as exc:
            last_error = str(exc)

        if attempt < retry_times:
            if progress is not None:
                progress(f"{label} failed, retry after {sleep_seconds}s: {last_error}")
            safe_logout(quiet=quiet)
            time.sleep(max(0.0, sleep_seconds))
            login_with_retry(
                retry_times=retry_times,
                retry_sleep_seconds=sleep_seconds,
                quiet=quiet,
                progress=progress,
            )
            if relogin_sleep_seconds is not None:
                time.sleep(max(0.0, relogin_sleep_seconds))

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
