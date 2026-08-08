from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.utils import load_ini_section, parse_bool_strict, project_root_from
import duckdb


def require_duckdb():
    return duckdb

@dataclass(frozen=True)
class DuckDBSettings:
    path: Path
    read_only: bool = False
    threads: int | None = None
    memory_limit: str | None = None
    temp_directory: Path | None = None

    @classmethod
    def from_sources(
        cls,
        *,
        path: str | Path | None = None,
        read_only: bool | None = None,
        config_path: str | Path | None = None,
        threads: int | None = None,
        memory_limit: str | None = None,
        temp_directory: str | Path | None = None,
    ) -> "DuckDBSettings":
        project_root = project_root_from(__file__)
        default_config_path = project_root / "config" / "duckdb.conf"
        resolved_config_path = Path(config_path) if config_path is not None else default_config_path
        file_config = load_ini_section(resolved_config_path, "duckdb")

        def pick_str(explicit: str | Path | None, env_key: str, config_key: str, default: str | Path | None = None) -> str | Path | None:
            if explicit is not None:
                return explicit
            env_value = os.getenv(env_key)
            if env_value not in {None, ""}:
                return env_value
            config_value = file_config.get(config_key)
            if config_value not in {None, ""}:
                return config_value
            return default

        def pick_int(explicit: int | None, env_key: str, config_key: str) -> int | None:
            if explicit is not None:
                return int(explicit)
            env_value = os.getenv(env_key)
            if env_value not in {None, ""}:
                return int(env_value)
            config_value = file_config.get(config_key)
            if config_value not in {None, ""}:
                return int(config_value)
            return None

        def pick_bool(explicit: bool | None, env_key: str, config_key: str, default: bool) -> bool:
            if explicit is not None:
                return bool(explicit)
            env_value = os.getenv(env_key)
            if env_value not in {None, ""}:
                return parse_bool_strict(env_value, default=default)
            return parse_bool_strict(file_config.get(config_key), default=default)

        raw_path = pick_str(path, "DUCKDB_PATH", "path")
        if raw_path in {None, ""}:
            raw_path = project_root / "db_data" / "duckdb" / "quant.duckdb"
        resolved_path = Path(raw_path)
        if not resolved_path.is_absolute():
            resolved_path = project_root / resolved_path

        raw_temp_directory = pick_str(temp_directory, "DUCKDB_TEMP_DIRECTORY", "temp_directory")
        resolved_temp_directory = Path(raw_temp_directory) if raw_temp_directory not in {None, ""} else None
        if resolved_temp_directory is not None and not resolved_temp_directory.is_absolute():
            resolved_temp_directory = project_root / resolved_temp_directory

        return cls(
            path=resolved_path,
            read_only=pick_bool(read_only, "DUCKDB_READ_ONLY", "read_only", False),
            threads=pick_int(threads, "DUCKDB_THREADS", "threads"),
            memory_limit=pick_str(memory_limit, "DUCKDB_MEMORY_LIMIT", "memory_limit"),
            temp_directory=resolved_temp_directory,
        )


class DuckDBConfig:
    """Small local DuckDB connection wrapper for research and data jobs."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        read_only: bool | None = None,
        config_path: str | Path | None = None,
        threads: int | None = None,
        memory_limit: str | None = None,
        temp_directory: str | Path | None = None,
    ) -> None:
        self.settings = DuckDBSettings.from_sources(
            path=path,
            read_only=read_only,
            config_path=config_path,
            threads=threads,
            memory_limit=memory_limit,
            temp_directory=temp_directory,
        )
        self.path = self.settings.path
        if not self.settings.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.path), read_only=self.settings.read_only)
        self._configure_connection()

    def _configure_connection(self) -> None:
        if self.settings.threads is not None:
            self.connection.execute(f"pragma threads={int(self.settings.threads)}")
        if self.settings.memory_limit:
            self.connection.execute("set memory_limit = ?", [str(self.settings.memory_limit)])
        if self.settings.temp_directory is not None:
            if not self.settings.read_only:
                self.settings.temp_directory.mkdir(parents=True, exist_ok=True)
            self.connection.execute("set temp_directory = ?", [str(self.settings.temp_directory)])

    def close(self) -> None:
        self.connection.close()

    @property
    def cache_revision(self) -> str:
        """Return a cheap file revision token for external read caches.

        Parquet caches live outside DuckDB, so their keys must include the
        database file revision. A later read-only process then cannot silently
        reuse frames produced before the latest ingest or feature build.
        """

        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return f"{self.path.resolve()}:missing"
        return f"{self.path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"

    @contextmanager
    def registered_frame(self, name: str, frame: pd.DataFrame):
        self.connection.register(name, frame)
        try:
            yield
        finally:
            self.connection.unregister(name)

    def fetch_df(self, sql: str, parameters: list[Any] | tuple[Any, ...] | None = None) -> pd.DataFrame:
        return self.connection.execute(sql, parameters or []).fetchdf()

    def execute(self, sql: str, parameters: list[Any] | tuple[Any, ...] | None = None) -> None:
        self.connection.execute(sql, parameters or [])
