from __future__ import annotations

import base64
import json
import random
import sqlite3
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Any

import ormsgpack
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import _msgpack_ext_hook_to_json

from lc_templates.core.config import get_settings, resolve_runtime_path
from lc_templates.core.hooks import emit_event
from lc_templates.core.logging import get_logger
from lc_templates.core.schemas import ExecutionMetadata

logger = get_logger(__name__)


class SqliteCheckpointSaver(
    BaseCheckpointSaver[str], AbstractContextManager, AbstractAsyncContextManager
):
    def __init__(
        self,
        path: str | Path,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    checkpoint_type TEXT NOT NULL,
                    checkpoint_blob BLOB NOT NULL,
                    checkpoint_preview TEXT NOT NULL DEFAULT '',
                    metadata_type TEXT NOT NULL,
                    metadata_blob BLOB NOT NULL,
                    metadata_preview TEXT NOT NULL DEFAULT '',
                    parent_checkpoint_id TEXT,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                );

                CREATE TABLE IF NOT EXISTS blobs (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    version TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    value_blob BLOB NOT NULL,
                    value_preview TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                );

                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    write_idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    value_blob BLOB NOT NULL,
                    value_preview TEXT NOT NULL DEFAULT '',
                    task_path TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx)
                );
                """
            )
            self._ensure_column("checkpoints", "checkpoint_preview", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("checkpoints", "metadata_preview", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("blobs", "value_preview", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("writes", "value_preview", "TEXT NOT NULL DEFAULT ''")
            self._conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {str(row["name"]) for row in rows}
        if column not in existing:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _typed_value_to_preview(self, value_type: str, value_blob: bytes) -> str:
        if value_type == "null":
            return "null"
        if value_type == "empty":
            return ""
        if value_type == "json":
            try:
                return value_blob.decode("utf-8")
            except UnicodeDecodeError:
                return value_blob.decode("utf-8", errors="replace")
        if value_type == "msgpack":
            try:
                parsed = ormsgpack.unpackb(
                    value_blob,
                    ext_hook=_msgpack_ext_hook_to_json,
                    option=ormsgpack.OPT_NON_STR_KEYS,
                )
                return json.dumps(parsed, ensure_ascii=False, indent=2, default=str)
            except Exception:
                pass
        if value_type in {"bytes", "bytearray"}:
            try:
                decoded = value_blob.decode("utf-8")
                return decoded
            except UnicodeDecodeError:
                encoded = base64.b64encode(value_blob).decode("ascii")
                return f"[base64]{encoded}"
        try:
            return value_blob.decode("utf-8")
        except UnicodeDecodeError:
            encoded = base64.b64encode(value_blob).decode("ascii")
            return f"[binary:{value_type}]{encoded}"

    def __enter__(self) -> SqliteCheckpointSaver:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None

    async def __aenter__(self) -> SqliteCheckpointSaver:
        return self

    async def __aexit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_conn", None) is not None:
                self._conn.close()
                self._conn = None

    def _thread_id(self, config: RunnableConfig) -> str:
        return str(config["configurable"]["thread_id"])

    def _checkpoint_ns(self, config: RunnableConfig) -> str:
        return str(config["configurable"].get("checkpoint_ns", ""))

    def _load_blobs(
        self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        with self._lock:
            for channel, version in versions.items():
                row = self._conn.execute(
                    """
                    SELECT value_type, value_blob
                    FROM blobs
                    WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? AND version = ?
                    """,
                    (thread_id, checkpoint_ns, channel, str(version)),
                ).fetchone()
                if row and row["value_type"] != "empty":
                    values[channel] = self.serde.loads_typed(
                        (row["value_type"], row["value_blob"])
                    )
        return values

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT task_id, channel, value_type, value_blob
                FROM writes
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                ORDER BY write_idx ASC
                """,
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchall()
        return [
            (
                row["task_id"],
                row["channel"],
                self.serde.loads_typed((row["value_type"], row["value_blob"])),
            )
            for row in rows
        ]

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = self._thread_id(config)
        checkpoint_ns = self._checkpoint_ns(config)
        checkpoint_id = get_checkpoint_id(config)

        with self._lock:
            if checkpoint_id:
                row = self._conn.execute(
                    """
                    SELECT *
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            else:
                row = self._conn.execute(
                    """
                    SELECT *
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                ).fetchone()

        if row is None:
            return None

        checkpoint: Checkpoint = self.serde.loads_typed(
            (row["checkpoint_type"], row["checkpoint_blob"])
        )
        resolved_checkpoint_id = str(row["checkpoint_id"])
        parent_checkpoint_id = row["parent_checkpoint_id"]
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": resolved_checkpoint_id,
                }
            },
            checkpoint={
                **checkpoint,
                "channel_values": self._load_blobs(
                    thread_id, checkpoint_ns, checkpoint["channel_versions"]
                ),
            },
            metadata=self.serde.loads_typed((row["metadata_type"], row["metadata_blob"])),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=self._load_pending_writes(
                thread_id, checkpoint_ns, resolved_checkpoint_id
            ),
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        params: list[Any] = []
        query = "SELECT * FROM checkpoints"
        where_parts: list[str] = []

        if config:
            where_parts.append("thread_id = ?")
            params.append(str(config["configurable"]["thread_id"]))
            where_parts.append("checkpoint_ns = ?")
            params.append(str(config["configurable"].get("checkpoint_ns", "")))
            if checkpoint_id := get_checkpoint_id(config):
                where_parts.append("checkpoint_id = ?")
                params.append(checkpoint_id)

        if before and (before_checkpoint_id := get_checkpoint_id(before)):
            where_parts.append("checkpoint_id < ?")
            params.append(before_checkpoint_id)

        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)
        query += " ORDER BY checkpoint_id DESC"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        for row in rows:
            metadata = self.serde.loads_typed((row["metadata_type"], row["metadata_blob"]))
            if filter and not all(metadata.get(key) == value for key, value in filter.items()):
                continue
            checkpoint: Checkpoint = self.serde.loads_typed(
                (row["checkpoint_type"], row["checkpoint_blob"])
            )
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": row["thread_id"],
                        "checkpoint_ns": row["checkpoint_ns"],
                        "checkpoint_id": row["checkpoint_id"],
                    }
                },
                checkpoint={
                    **checkpoint,
                    "channel_values": self._load_blobs(
                        row["thread_id"],
                        row["checkpoint_ns"],
                        checkpoint["channel_versions"],
                    ),
                },
                metadata=metadata,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": row["thread_id"],
                            "checkpoint_ns": row["checkpoint_ns"],
                            "checkpoint_id": row["parent_checkpoint_id"],
                        }
                    }
                    if row["parent_checkpoint_id"]
                    else None
                ),
                pending_writes=self._load_pending_writes(
                    row["thread_id"], row["checkpoint_ns"], row["checkpoint_id"]
                ),
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = self._thread_id(config)
        checkpoint_ns = self._checkpoint_ns(config)
        values = checkpoint.get("channel_values", {})
        checkpoint_copy = checkpoint.copy()
        checkpoint_copy.pop("channel_values", None)

        with self._lock:
            for channel, version in new_versions.items():
                if channel in values:
                    value_type, value_blob = self.serde.dumps_typed(values[channel])
                else:
                    value_type, value_blob = ("empty", b"")
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO blobs
                    (
                        thread_id, checkpoint_ns, channel, version,
                        value_type, value_blob, value_preview
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        channel,
                        str(version),
                        value_type,
                        value_blob,
                        self._typed_value_to_preview(value_type, value_blob),
                    ),
                )

            checkpoint_type, checkpoint_blob = self.serde.dumps_typed(checkpoint_copy)
            metadata_type, metadata_blob = self.serde.dumps_typed(
                get_checkpoint_metadata(config, metadata)
            )
            self._conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (
                    thread_id, checkpoint_ns, checkpoint_id,
                    checkpoint_type, checkpoint_blob, checkpoint_preview,
                    metadata_type, metadata_blob, metadata_preview,
                    parent_checkpoint_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_type,
                    checkpoint_blob,
                    self._typed_value_to_preview(checkpoint_type, checkpoint_blob),
                    metadata_type,
                    metadata_blob,
                    self._typed_value_to_preview(metadata_type, metadata_blob),
                    config["configurable"].get("checkpoint_id"),
                ),
            )
            self._conn.commit()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = self._thread_id(config)
        checkpoint_ns = self._checkpoint_ns(config)
        checkpoint_id = str(config["configurable"]["checkpoint_id"])
        with self._lock:
            for idx, (channel, value) in enumerate(writes):
                write_idx = WRITES_IDX_MAP.get(channel, idx)
                value_type, value_blob = self.serde.dumps_typed(value)
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO writes
                    (
                        thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx,
                        channel, value_type, value_blob, value_preview, task_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_idx,
                        channel,
                        value_type,
                        value_blob,
                        self._typed_value_to_preview(value_type, value_blob),
                        task_path,
                    ),
                )
            self._conn.commit()

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            self._conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            self._conn.execute("DELETE FROM blobs WHERE thread_id = ?", (thread_id,))
            self._conn.commit()

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        if not run_ids:
            return
        with self._lock:
            rows = self._conn.execute(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, metadata_type, metadata_blob "
                "FROM checkpoints WHERE metadata_blob IS NOT NULL"
            ).fetchall()
            for row in rows:
                metadata = self.serde.loads_typed((row["metadata_type"], row["metadata_blob"]))
                if metadata.get("run_id") in run_ids:
                    self._conn.execute(
                        """
                        DELETE FROM checkpoints
                        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                        """,
                        (row["thread_id"], row["checkpoint_ns"], row["checkpoint_id"]),
                    )
                    self._conn.execute(
                        """
                        DELETE FROM writes
                        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                        """,
                        (row["thread_id"], row["checkpoint_ns"], row["checkpoint_id"]),
                    )
            self._conn.commit()

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        for table in ("checkpoints", "writes", "blobs"):
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT * FROM {table} WHERE thread_id = ?", (source_thread_id,)
                ).fetchall()
                for row in rows:
                    columns = list(row.keys())
                    values = [
                        target_thread_id if column == "thread_id" else row[column]
                        for column in columns
                    ]
                    placeholders = ", ".join("?" for _ in columns)
                    columns_sql = ", ".join(columns)
                    self._conn.execute(
                        f"INSERT OR REPLACE INTO {table} ({columns_sql}) VALUES ({placeholders})",
                        values,
                    )
        with self._lock:
            self._conn.commit()

    def prune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        for thread_id in thread_ids:
            if strategy == "delete":
                self.delete_thread(thread_id)
                continue
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT checkpoint_ns, MAX(checkpoint_id) AS checkpoint_id
                    FROM checkpoints
                    WHERE thread_id = ?
                    GROUP BY checkpoint_ns
                    """,
                    (thread_id,),
                ).fetchall()
                keep = {(row["checkpoint_ns"], row["checkpoint_id"]) for row in rows}
                checkpoints = self._conn.execute(
                    "SELECT checkpoint_ns, checkpoint_id FROM checkpoints WHERE thread_id = ?",
                    (thread_id,),
                ).fetchall()
                for row in checkpoints:
                    key = (row["checkpoint_ns"], row["checkpoint_id"])
                    if key in keep:
                        continue
                    self._conn.execute(
                        """
                        DELETE FROM checkpoints
                        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                        """,
                        (thread_id, row["checkpoint_ns"], row["checkpoint_id"]),
                    )
                    self._conn.execute(
                        """
                        DELETE FROM writes
                        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                        """,
                        (thread_id, row["checkpoint_ns"], row["checkpoint_id"]),
                    )
        with self._lock:
            self._conn.commit()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        self.delete_for_runs(run_ids)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        self.copy_thread(source_thread_id, target_thread_id)

    async def aprune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        self.prune(thread_ids, strategy=strategy)

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        next_v = current_v + 1
        return f"{next_v:032}.{random.random():016}"


_CHECKPOINTER: SqliteCheckpointSaver | InMemorySaver | None = None
_CHECKPOINTER_KEY: tuple[str, str] | None = None


def build_memory_checkpointer() -> SqliteCheckpointSaver | InMemorySaver:
    global _CHECKPOINTER, _CHECKPOINTER_KEY

    settings = get_settings()
    memory = settings.runtime.memory
    backend = memory.backend
    if backend == "sqlite":
        sqlite_path = resolve_runtime_path(memory.sqlite_path) or Path(memory.sqlite_path)
        key = (backend, str(sqlite_path))
        if _CHECKPOINTER is None or _CHECKPOINTER_KEY != key:
            _CHECKPOINTER = SqliteCheckpointSaver(sqlite_path)
            _CHECKPOINTER_KEY = key
            logger.info("Initialized SQLite memory checkpointer. path=%s", sqlite_path)
            emit_event(
                "memory.checkpointer_initialized",
                message="SQLite memory checkpointer initialized.",
                meta=ExecutionMetadata(operation="memory_checkpoint"),
                payload={"backend": backend, "path": str(sqlite_path)},
            )
        return _CHECKPOINTER

    key = (backend, "memory")
    if _CHECKPOINTER is None or _CHECKPOINTER_KEY != key:
        _CHECKPOINTER = InMemorySaver()
        _CHECKPOINTER_KEY = key
        logger.info("Initialized in-memory checkpointer.")
        emit_event(
            "memory.checkpointer_initialized",
            message="In-memory checkpointer initialized.",
            meta=ExecutionMetadata(operation="memory_checkpoint"),
            payload={"backend": backend},
        )
    return _CHECKPOINTER
