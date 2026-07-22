"""
TitanFrame Arrow IPC I/O
=========================

Read/write Arrow IPC (Inter-Process Communication) format — the canonical
serialization format used by TitanFrame for NVMe spilling and fast data
exchange.

Arrow IPC provides zero-copy deserialization: data can be memory-mapped
directly from disk without any parsing overhead.

Two variants:
    - **Streaming format**: sequential batches, good for spilling (append-only).
    - **File format**: random access to individual record batches.

Example::

    >>> from titanframe.io.arrow_ipc import write_ipc, read_ipc, infer_ipc_schema
    >>> write_ipc(table, "data.arrow")
    >>> restored = read_ipc("data.arrow")
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator, Optional

import pyarrow as pa
import pyarrow.ipc as ipc

from titanframe.core.chunk import Chunk
from titanframe.core.schema import Schema
from titanframe.core.table import Table


def infer_ipc_schema(path: str | Path) -> Schema:
    """
    Read the schema from an Arrow IPC file without loading data.

    Args:
        path: Path to the Arrow IPC file.

    Returns:
        The file's :class:`Schema`.
    """
    reader = ipc.open_file(str(path))
    return Schema.from_arrow(reader.schema)


def read_ipc(path: str | Path) -> Table:
    """
    Read an Arrow IPC file into a TitanFrame Table.

    Uses the file format (random access) for efficient loading.

    Args:
        path: Path to the Arrow IPC file.

    Returns:
        A :class:`Table`.
    """
    return Table.from_ipc_file(str(path))


def read_ipc_streaming(path: str | Path) -> Iterator[Chunk]:
    """
    Stream an Arrow IPC file as an iterator of Chunks.

    Args:
        path: Path to the Arrow IPC streaming file.

    Yields:
        :class:`Chunk` instances.
    """
    with open(str(path), "rb") as f:
        reader = ipc.open_stream(f)
        for batch in reader:
            yield Chunk(batch)


def read_ipc_bytes(data: bytes) -> Table:
    """
    Deserialize a Table from Arrow IPC bytes (streaming format).

    Args:
        data: Raw bytes in Arrow IPC streaming format.

    Returns:
        A :class:`Table`.
    """
    return Table.from_ipc_bytes(data)


def write_ipc(
    table: Table,
    path: str | Path,
    compression: Optional[str] = None,
) -> None:
    """
    Write a TitanFrame Table to an Arrow IPC file.

    Args:
        table: Table to write.
        path: Output file path.
        compression: Optional compression codec ("lz4", "zstd", None).
    """
    path = str(path)
    arrow_table = table.to_arrow()

    options = None
    if compression:
        options = ipc.IpcWriteOptions(
            compression=pa.Codec(compression),
        )

    writer = ipc.new_file(path, arrow_table.schema, options=options)
    for batch in arrow_table.to_batches():
        writer.write_batch(batch)
    writer.close()


def write_ipc_streaming(
    table: Table,
    path: str | Path,
    compression: Optional[str] = None,
) -> None:
    """
    Write a TitanFrame Table to an Arrow IPC streaming file.

    The streaming format is used for NVMe spilling since it supports
    append-only writes.

    Args:
        table: Table to write.
        path: Output file path.
        compression: Optional compression codec.
    """
    path = str(path)
    arrow_schema = table.schema.to_arrow()

    options = None
    if compression:
        options = ipc.IpcWriteOptions(
            compression=pa.Codec(compression),
        )

    with open(path, "wb") as f:
        writer = ipc.new_stream(f, arrow_schema, options=options)
        for chunk in table.iter_chunks():
            writer.write_batch(chunk.data)
        writer.close()


def to_ipc_bytes(table: Table) -> bytes:
    """
    Serialize a Table to Arrow IPC bytes (streaming format).

    Args:
        table: Table to serialize.

    Returns:
        Raw bytes in Arrow IPC streaming format.
    """
    return table.to_ipc_bytes()


def write_ipc_batch(
    path: str | Path,
    schema: Schema,
) -> IPCBatchWriter:
    """
    Open an IPC file for incremental batch writing.

    Returns a context manager that accepts individual Chunks::

        >>> with write_ipc_batch("output.arrow", schema) as writer:
        ...     for chunk in stream:
        ...         writer.write(chunk)
    """
    return IPCBatchWriter(str(path), schema)


class IPCBatchWriter:
    """
    Incrementally write Arrow IPC batches to a file.

    Used by the execution engine to stream output without buffering
    the entire result in memory.
    """

    __slots__ = ("_path", "_schema", "_writer", "_file")

    def __init__(self, path: str, schema: Schema):
        self._path = path
        self._schema = schema
        self._writer: Optional[ipc.RecordBatchFileWriter] = None
        self._file = None

    def __enter__(self) -> IPCBatchWriter:
        self._file = open(self._path, "wb")
        self._writer = ipc.new_file(self._file, self._schema.to_arrow())
        return self

    def write(self, chunk: Chunk) -> None:
        """Write a single chunk to the file."""
        if self._writer is None:
            raise RuntimeError("Writer not opened. Use 'with' statement.")
        self._writer.write_batch(chunk.data)

    def __exit__(self, *args: object) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._file is not None:
            self._file.close()

    def close(self) -> None:
        """Explicitly close the writer."""
        self.__exit__()
