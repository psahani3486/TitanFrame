"""
TitanFrame Table
================

A Table is an immutable collection of Chunks sharing a common Schema. It is
the primary data container — equivalent to a materialized DataFrame before
the API layer adds lazy semantics.

Design notes:
    - Tables are **immutable** — every "mutation" returns a new Table.
    - Tables are **chunked** — they hold a list of Chunks, each of which is
      an Arrow RecordBatch. This enables streaming, out-of-core, and multi-GPU
      processing without ever loading the full dataset.
    - Tables can be constructed from Arrow Tables, Python dicts, or Pandas
      DataFrames with zero/minimal copy.

Example::

    >>> from titanframe.core.table import Table
    >>> t = Table.from_pydict({"name": ["Alice", "Bob"], "age": [30, 25]})
    >>> t.num_rows
    2
    >>> t.schema
    Schema({'name': Utf8, 'age': Int64})
"""

from __future__ import annotations

import io
from typing import Any, Iterator, Optional, Sequence

import pyarrow as pa
import pyarrow.ipc as ipc

from titanframe.core.chunk import Chunk, DeviceLocation
from titanframe.core.column import ChunkedColumn
from titanframe.core.dtypes import DType, from_arrow
from titanframe.core.schema import Schema, SchemaError


class Table:
    """
    An immutable, chunked, columnar table.

    A Table holds a list of :class:`Chunk` objects, all sharing the same
    :class:`Schema`. Data flows through the execution engine as Tables,
    and all DataFrame operations ultimately produce Tables.

    Args:
        schema: The schema describing all columns and their types.
        chunks: List of :class:`Chunk` instances. Each must conform to ``schema``.
    """

    __slots__ = ("_schema", "_chunks")

    def __init__(self, schema: Schema, chunks: list[Chunk] | None = None):
        self._schema = schema
        self._chunks = chunks or []

        for i, chunk in enumerate(self._chunks):
            if chunk.schema != schema:
                raise SchemaError(
                    f"Chunk {i} schema {chunk.schema} does not match "
                    f"table schema {schema}"
                )


    @property
    def schema(self) -> Schema:
        """The schema of this table."""
        return self._schema

    @property
    def chunks(self) -> list[Chunk]:
        """The underlying chunks (read-only view)."""
        return list(self._chunks)

    @property
    def num_chunks(self) -> int:
        """Number of chunks in this table."""
        return len(self._chunks)

    @property
    def num_rows(self) -> int:
        """Total number of rows across all chunks."""
        return sum(chunk.num_rows for chunk in self._chunks)

    @property
    def num_columns(self) -> int:
        """Number of columns."""
        return self._schema.num_columns

    @property
    def num_bytes(self) -> int:
        """Total byte size of all data."""
        return sum(chunk.num_bytes for chunk in self._chunks)

    @property
    def column_names(self) -> list[str]:
        """Column names in order."""
        return self._schema.names

    @property
    def dtypes(self) -> list[DType]:
        """Column data types in order."""
        return self._schema.dtypes

    @property
    def is_empty(self) -> bool:
        """Whether the table has zero rows."""
        return self.num_rows == 0

    @property
    def shape(self) -> tuple[int, int]:
        """(num_rows, num_columns) — like Pandas."""
        return (self.num_rows, self.num_columns)


    def column(self, name: str) -> ChunkedColumn:
        """
        Get a column as a :class:`ChunkedColumn`.

        Args:
            name: Column name.

        Returns:
            A ChunkedColumn containing arrays from all chunks.

        Raises:
            SchemaError: If the column doesn't exist.
        """
        idx = self._schema.index(name)
        dtype = self._schema[name]
        arrays = [chunk.column_by_index(idx) for chunk in self._chunks]
        return ChunkedColumn(name, dtype, arrays)

    def columns(self, names: Sequence[str]) -> list[ChunkedColumn]:
        """Get multiple columns by name."""
        return [self.column(name) for name in names]


    def select(self, names: Sequence[str]) -> Table:
        """
        Return a new Table with only the specified columns (zero-copy).

        Args:
            names: Column names to keep.
        """
        new_schema = self._schema.select(names)
        new_chunks = [chunk.select(names) for chunk in self._chunks]
        return Table(new_schema, new_chunks)

    def drop(self, names: Sequence[str]) -> Table:
        """Return a new Table without the specified columns."""
        new_schema = self._schema.drop(names)
        keep = new_schema.names
        new_chunks = [chunk.select(keep) for chunk in self._chunks]
        return Table(new_schema, new_chunks)

    def rename(self, mapping: dict[str, str]) -> Table:
        """Return a new Table with columns renamed."""
        new_schema = self._schema.rename(mapping)
        new_chunks = [chunk.rename(mapping) for chunk in self._chunks]
        return Table(new_schema, new_chunks)

    def slice(self, offset: int, length: Optional[int] = None) -> Table:
        """
        Return a new Table with a slice of rows.

        Works across chunk boundaries — may produce partial chunks.
        """
        if length is None:
            length = self.num_rows - offset

        if offset < 0 or offset + length > self.num_rows:
            raise IndexError(
                f"Slice [{offset}:{offset + length}] out of range [0, {self.num_rows})"
            )

        new_chunks: list[Chunk] = []
        remaining_offset = offset
        remaining_length = length

        for chunk in self._chunks:
            chunk_len = chunk.num_rows

            if remaining_offset >= chunk_len:
                remaining_offset -= chunk_len
                continue

            take_start = remaining_offset
            take_length = min(remaining_length, chunk_len - take_start)

            if take_length > 0:
                new_chunks.append(chunk.slice(take_start, take_length))
                remaining_length -= take_length

            remaining_offset = 0

            if remaining_length <= 0:
                break

        return Table(self._schema, new_chunks)

    def head(self, n: int = 5) -> Table:
        """Return the first ``n`` rows."""
        return self.slice(0, min(n, self.num_rows))

    def tail(self, n: int = 5) -> Table:
        """Return the last ``n`` rows."""
        start = max(0, self.num_rows - n)
        return self.slice(start)

    def append_column(self, name: str, column: ChunkedColumn) -> Table:
        """
        Return a new Table with an additional column.

        The ChunkedColumn must have the same number of chunks and rows
        as this table (aligned chunk boundaries).
        """
        if column.num_rows != self.num_rows:
            raise ValueError(
                f"Column has {column.num_rows} rows but table has {self.num_rows}"
            )
        if column.num_chunks != self.num_chunks:
            raise ValueError(
                f"Column has {column.num_chunks} chunks but table has {self.num_chunks}. "
                f"Rechunk the column first."
            )

        new_schema = self._schema.append(name, column.dtype)
        new_chunks = []
        for table_chunk, col_chunk in zip(self._chunks, column.chunks):
            new_batch = table_chunk.data.append_column(name, col_chunk)
            new_chunks.append(Chunk(new_batch, table_chunk.device))
        return Table(new_schema, new_chunks)

    def vstack(self, other: Table) -> Table:
        """
        Vertically stack (concatenate) another table below this one.

        Both tables must have identical schemas.
        """
        self._schema.assert_compatible(other._schema, context="vstack")
        new_chunks = self._chunks + other._chunks
        return Table(self._schema, new_chunks)

    def rechunk(self, target_chunks: int = 1) -> Table:
        """
        Consolidate into ``target_chunks`` chunks.

        Useful before operations that benefit from large contiguous arrays
        (e.g., GPU transfer, sorting).
        """
        if len(self._chunks) <= target_chunks:
            return self

        arrow_table = self.to_arrow()
        combined = arrow_table.combine_chunks()

        if target_chunks == 1:
            batch = combined.to_batches()[0] if combined.num_rows > 0 else pa.record_batch(
                {name: pa.array([], type=self._schema[name].arrow_type)
                 for name in self._schema.names}
            )
            return Table(self._schema, [Chunk(batch)])

        total = self.num_rows
        rows_per = (total + target_chunks - 1) // target_chunks
        batches = []
        offset = 0
        while offset < total:
            length = min(rows_per, total - offset)
            batches.append(combined.slice(offset, length).to_batches()[0])
            offset += length

        return Table(self._schema, [Chunk(b) for b in batches])


    def iter_chunks(self) -> Iterator[Chunk]:
        """Iterate over chunks."""
        return iter(self._chunks)

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over rows as dictionaries.

        .. warning::
            This is slow for large tables. Prefer columnar operations.
        """
        for chunk in self._chunks:
            pydict = chunk.to_pydict()
            for i in range(chunk.num_rows):
                yield {name: values[i] for name, values in pydict.items()}


    def to_arrow(self) -> pa.Table:
        """Convert to a ``pyarrow.Table``."""
        if not self._chunks:
            return pa.table(
                {name: pa.array([], type=dtype.arrow_type)
                 for name, dtype in zip(self._schema.names, self._schema.dtypes)}
            )
        batches = [chunk.data for chunk in self._chunks]
        return pa.Table.from_batches(batches, schema=self._schema.to_arrow())

    @classmethod
    def from_arrow(
        cls,
        arrow_table: pa.Table,
        chunk_size: Optional[int] = None,
    ) -> Table:
        """
        Construct a Table from a ``pyarrow.Table``.

        Args:
            arrow_table: Source Arrow table.
            chunk_size: If provided, split into chunks of this many rows.
                        If None, preserves the Arrow table's existing chunking.
        """
        schema = Schema.from_arrow(arrow_table.schema)

        if chunk_size is not None:
            batches = arrow_table.to_batches(max_chunksize=chunk_size)
        else:
            batches = arrow_table.to_batches()

        chunks = [Chunk(batch) for batch in batches if batch.num_rows > 0]
        return cls(schema, chunks)


    def to_pandas(self) -> Any:
        """
        Convert to a ``pandas.DataFrame``.

        Requires pandas to be installed.
        """
        return self.to_arrow().to_pandas()

    @classmethod
    def from_pandas(cls, df: Any, chunk_size: Optional[int] = None) -> Table:
        """
        Construct a Table from a ``pandas.DataFrame``.

        Args:
            df: Pandas DataFrame.
            chunk_size: Optional chunk size for splitting.
        """
        arrow_table = pa.Table.from_pandas(df, preserve_index=False)
        return cls.from_arrow(arrow_table, chunk_size=chunk_size)


    def to_pydict(self) -> dict[str, list[Any]]:
        """Convert to a Python dictionary of lists."""
        return self.to_arrow().to_pydict()

    @classmethod
    def from_pydict(
        cls,
        data: dict[str, list[Any]],
        chunk_size: Optional[int] = None,
    ) -> Table:
        """
        Construct a Table from a Python dictionary of lists.

        Example::

            >>> Table.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        """
        arrow_table = pa.table(data)
        return cls.from_arrow(arrow_table, chunk_size=chunk_size)


    def to_ipc_bytes(self) -> bytes:
        """
        Serialize to Arrow IPC streaming format (bytes).

        This is the canonical format for spilling to NVMe.
        """
        sink = io.BytesIO()
        writer = ipc.new_stream(sink, self._schema.to_arrow())
        for chunk in self._chunks:
            writer.write_batch(chunk.data)
        writer.close()
        return sink.getvalue()

    @classmethod
    def from_ipc_bytes(cls, data: bytes) -> Table:
        """
        Deserialize from Arrow IPC streaming format.
        """
        reader = ipc.open_stream(data)
        schema = Schema.from_arrow(reader.schema)
        chunks = []
        for batch in reader:
            chunks.append(Chunk(batch))
        return cls(schema, chunks)

    def to_ipc_file(self, path: str) -> None:
        """Write to an Arrow IPC file."""
        arrow_table = self.to_arrow()
        writer = ipc.new_file(path, arrow_table.schema)
        for batch in arrow_table.to_batches():
            writer.write_batch(batch)
        writer.close()

    @classmethod
    def from_ipc_file(cls, path: str) -> Table:
        """Read from an Arrow IPC file."""
        reader = ipc.open_file(path)
        schema = Schema.from_arrow(reader.schema)
        chunks = []
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            chunks.append(Chunk(batch))
        return cls(schema, chunks)


    def __repr__(self) -> str:
        lines = [f"Table(rows={self.num_rows}, cols={self.num_columns}, chunks={self.num_chunks})"]
        lines.append(f"Schema: {self._schema}")

        if self.num_rows > 0:
            preview = self.head(min(5, self.num_rows))
            pydict = preview.to_pydict()
            col_widths = {}
            for name in self.column_names:
                values = [str(v) for v in pydict[name]]
                col_widths[name] = max(len(name), max(len(v) for v in values) if values else 0)

            header = " | ".join(name.ljust(col_widths[name]) for name in self.column_names)
            separator = "-+-".join("-" * col_widths[name] for name in self.column_names)
            lines.append(header)
            lines.append(separator)

            for i in range(preview.num_rows):
                row = " | ".join(
                    str(pydict[name][i]).ljust(col_widths[name])
                    for name in self.column_names
                )
                lines.append(row)

            if self.num_rows > 5:
                lines.append(f"... ({self.num_rows - 5} more rows)")

        return "\n".join(lines)

    def __len__(self) -> int:
        return self.num_rows

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Table):
            return NotImplemented
        if self._schema != other._schema:
            return False
        return self.to_arrow().equals(other.to_arrow())
