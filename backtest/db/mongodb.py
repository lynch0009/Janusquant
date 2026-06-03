from __future__ import annotations

from dataclasses import dataclass
import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote_plus

from pymongo import MongoClient

from backtest.utils import load_ini_section, parse_bool, project_root_from


@dataclass(frozen=True)
class MongoDBSettings:
    host: str
    port: int
    db_name: str
    username: str | None = None
    password: str | None = None
    auth_mechanism: str = "SCRAM-SHA-256"
    connect_timeout_ms: int = 10_000
    server_selection_timeout_ms: int = 10_000
    socket_timeout_ms: int = 30_000
    max_pool_size: int = 100
    min_pool_size: int = 0
    retry_reads: bool = True
    retry_writes: bool = True
    connect_eagerly: bool = False

    @classmethod
    def from_sources(
        cls,
        *,
        config_path: str | Path | None = None,
        host: str | None = None,
        port: int | None = None,
        db_name: str | None = None,
        username: str | None = None,
        password: str | None = None,
        auth_mechanism: str | None = None,
        connect_timeout_ms: int | None = None,
        server_selection_timeout_ms: int | None = None,
        socket_timeout_ms: int | None = None,
        max_pool_size: int | None = None,
        min_pool_size: int | None = None,
        retry_reads: bool | None = None,
        retry_writes: bool | None = None,
        connect_eagerly: bool | None = None,
    ) -> "MongoDBSettings":
        default_path = project_root_from(__file__) / "config" / "mongodb.conf"
        resolved_path = Path(config_path) if config_path is not None else default_path
        file_config = load_ini_section(resolved_path, "mongodb")

        def pick_str(explicit: str | None, env_key: str, config_key: str, default: str | None = None) -> str | None:
            if explicit is not None:
                return explicit
            env_value = os.getenv(env_key)
            if env_value not in {None, ""}:
                return env_value
            config_value = file_config.get(config_key)
            if config_value not in {None, ""}:
                return config_value
            return default

        def pick_int(explicit: int | None, env_key: str, config_key: str, default: int) -> int:
            if explicit is not None:
                return int(explicit)
            env_value = os.getenv(env_key)
            if env_value not in {None, ""}:
                return int(env_value)
            config_value = file_config.get(config_key)
            if config_value not in {None, ""}:
                return int(config_value)
            return default

        def pick_bool(explicit: bool | None, env_key: str, config_key: str, default: bool) -> bool:
            if explicit is not None:
                return bool(explicit)
            env_value = os.getenv(env_key)
            if env_value not in {None, ""}:
                return parse_bool(env_value, default=default)
            return parse_bool(file_config.get(config_key), default=default)

        return cls(
            host=pick_str(host, "MONGO_HOST", "host", "localhost") or "localhost",
            port=pick_int(port, "MONGO_PORT", "port", 27017),
            db_name=pick_str(db_name, "MONGO_DB_NAME", "db_name", "quant") or "quant",
            username=pick_str(username, "MONGO_USERNAME", "username"),
            password=pick_str(password, "MONGO_PASSWORD", "password"),
            auth_mechanism=pick_str(auth_mechanism, "MONGO_AUTH_MECHANISM", "auth_mechanism", "SCRAM-SHA-256")
            or "SCRAM-SHA-256",
            connect_timeout_ms=pick_int(connect_timeout_ms, "MONGO_CONNECT_TIMEOUT_MS", "connect_timeout_ms", 10_000),
            server_selection_timeout_ms=pick_int(
                server_selection_timeout_ms,
                "MONGO_SERVER_SELECTION_TIMEOUT_MS",
                "server_selection_timeout_ms",
                10_000,
            ),
            socket_timeout_ms=pick_int(socket_timeout_ms, "MONGO_SOCKET_TIMEOUT_MS", "socket_timeout_ms", 30_000),
            max_pool_size=pick_int(max_pool_size, "MONGO_MAX_POOL_SIZE", "max_pool_size", 100),
            min_pool_size=pick_int(min_pool_size, "MONGO_MIN_POOL_SIZE", "min_pool_size", 0),
            retry_reads=pick_bool(retry_reads, "MONGO_RETRY_READS", "retry_reads", True),
            retry_writes=pick_bool(retry_writes, "MONGO_RETRY_WRITES", "retry_writes", True),
            connect_eagerly=pick_bool(connect_eagerly, "MONGO_CONNECT_EAGERLY", "connect_eagerly", False),
        )

    def build_uri(self) -> str:
        if self.username and self.password:
            return (
                f"mongodb://{quote_plus(self.username)}:{quote_plus(self.password)}@{self.host}:{self.port}/"
                f"?authSource={self.db_name}"
            )
        return f"mongodb://{self.host}:{self.port}/"


class MongoDBConfig:
    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        host: str | None = None,
        db_name: str | None = None,
        port: int | None = None,
        auth_mechanism: str | None = None,
        connect_timeout_ms: int | None = None,
        server_selection_timeout_ms: int | None = None,
        socket_timeout_ms: int | None = None,
        max_pool_size: int | None = None,
        min_pool_size: int | None = None,
        retry_reads: bool | None = None,
        retry_writes: bool | None = None,
        connect: bool | None = None,
        config_path: str | Path | None = None,
    ):
        self.settings = MongoDBSettings.from_sources(
            config_path=config_path,
            host=host,
            port=port,
            db_name=db_name,
            username=username,
            password=password,
            auth_mechanism=auth_mechanism,
            connect_timeout_ms=connect_timeout_ms,
            server_selection_timeout_ms=server_selection_timeout_ms,
            socket_timeout_ms=socket_timeout_ms,
            max_pool_size=max_pool_size,
            min_pool_size=min_pool_size,
            retry_reads=retry_reads,
            retry_writes=retry_writes,
            connect_eagerly=connect,
        )
        self.uri = self.settings.build_uri()
        self.client = MongoClient(
            self.uri,
            authMechanism=self.settings.auth_mechanism,
            # connectTimeoutMS=self.settings.connect_timeout_ms,
            # serverSelectionTimeoutMS=self.settings.server_selection_timeout_ms,
            # socketTimeoutMS=self.settings.socket_timeout_ms,
            maxPoolSize=self.settings.max_pool_size,
            minPoolSize=self.settings.min_pool_size,
            retryReads=self.settings.retry_reads,
            retryWrites=self.settings.retry_writes,
            connect=self.settings.connect_eagerly,
        )
        self.db = self.client[self.settings.db_name]

    @contextmanager
    def session(self):
        with self.client.start_session() as session:
            with session.start_transaction():
                yield session


if __name__ == "__main__":
    config = MongoDBConfig()
    try:
        count = config.db["orders"].count_documents({})
        print(f"MongoDB connected successfully, documents in orders: {count}")
    except Exception as exc:
        print(f"MongoDB connection failed: {exc}")
