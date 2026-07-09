#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import loguru
from loguru import logger


class Logger:
    def __init__(self):
        self.log_path = os.path.join(Path(__file__).resolve().parent.parent, 'log')

    def log(self) -> loguru.Logger:
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        # 日志文件
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        process_id = os.getpid()
        log_stdout_file = os.path.join(self.log_path, f"normal_info_{run_id}_{process_id}.log")
        log_stderr_file = os.path.join(self.log_path, f"error_{run_id}_{process_id}.log")

        # loguru 日志: https://loguru.readthedocs.io/en/stable/api/logger.html#loguru._logger.Logger.add
        log_config = dict(enqueue=True, backtrace=False, diagnose=False)
        # stdout
        logger.add(
            log_stdout_file,
            level='INFO',
            filter=lambda record: record['level'].name == 'INFO' or record['level'].no <= 25,
            **log_config,
        )
        # stderr
        logger.add(
            log_stderr_file,
            level='ERROR',
            filter=lambda record: record['level'].name == 'ERROR' or record['level'].no >= 30,
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )

        return logger


_configured_log: loguru.Logger | None = None


def get_logger() -> loguru.Logger:
    """Return the project logger, configuring file sinks on first use."""

    global _configured_log
    if _configured_log is None:
        _configured_log = Logger().log()
    return _configured_log


class LazyLogger:
    """Proxy that preserves the old ``log.info(...)`` style without import side effects."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_logger(), name)


log = LazyLogger()


def format_date(value: Any, *, fmt: str = "%Y-%m-%d", default: str = "None") -> str:
    """把日期时间对象统一格式化成字符串，便于拼接日志。"""
    if value is None:
        return default
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.strftime(fmt)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day).strftime(fmt)
    return str(value)


def format_value(value: Any) -> str:
    """把常见对象格式化成日志友好的单行字符串。"""
    if isinstance(value, (datetime, date)) or hasattr(value, "to_pydatetime"):
        return format_date(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, set)):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}={format_value(item)}" for key, item in value.items()) + "}"
    return str(value)


def format_fields(**fields: Any) -> str:
    """把键值参数格式化成统一的 key=value 日志片段。"""
    return ", ".join(f"{key}={format_value(value)}" for key, value in fields.items())


def build_log_message(event: str, **fields: Any) -> str:
    """构造统一格式的事件日志消息。"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not fields:
        return f"[{timestamp}] {event}"
    return f"[{timestamp}] {event}: {format_fields(**fields)}"


def log_event(level: str, event: str, **fields: Any) -> None:
    """用统一格式输出普通事件日志。"""
    getattr(get_logger(), level)(build_log_message(event, **fields))


def log_exception(event: str, **fields: Any) -> None:
    """用统一格式输出异常日志，并保留 traceback。"""
    get_logger().exception(build_log_message(event, **fields))
