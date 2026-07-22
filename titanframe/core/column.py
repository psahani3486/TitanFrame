"""
TitanFrame ChunkedColumn
========================

A ChunkedColumn represents a single logical column as a list of Arrow arrays
(chunks). This mirrors ``pyarrow.ChunkedArray`` but adds TitanFrame-specific
metadata like device tracking and null statistics caching.

The chunked design is essential for out-of-core processing: each chunk can
reside on a different device (GPU_0, GPU_1, CPU, NVMe) and be processed
independently.

Example::

    >>> import pyarrow as pa
    >>> from titanframe.core.column import ChunkedColumn
    >>> from titanframe.core.dtypes import Int64
    >>> col = ChunkedColumn("revenue", Int64, [pa.array([1, 2, 3]), pa.array([4, 5])])
    >>> col.num_rows
    5
    >>> col.null_count
    0
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import pyarrow as pa

from titanframe.core.dtypes import DType, from_arrow


class ChunkedColumn:
    """
    A single column represented as a list of Arrow arrays.

    Each array chunk can potentially reside on a different device, enabling
    distributed column storage across multiple GPUs.

    Args:
        name: Column name.
        dtype: TitanFrame data type.
        chunks: List of ``pyarrow.Array`` instances. All must have the
                same type (compatible with ``dtype``).
    """

    __slots__ = ("_name", "_dtype", "_chunks", "_null_count_cache")

    def __init__(self, name: str, dtype: DType, chunks: Sequence[pa.Array] | None = None):
        self._name = name
        self._dtype = dtype
        self._chunks: list[pa.Array] = list(chunks) if chunks else []
        self._null_count_cache: Optional[int] = None

        # Validate chunk types
        for i, chunk in enumerate(self._chunks):
            chunk_dtype = from_arrow(chunk.type)
            if chunk_dtype != dtype:
                raise TypeError(
                    f"Chunk {i} has type {chunk_dtype} but column {name!r} "
                    f"expects {dtype}"
                )

    # ---- Properties ----

    @property
    def name(self) -> str:
        """Column name."""
        return self._name

    @property
    def dtype(self) -> DType:
        """TitanFrame data type."""
        return self._dtype

    @property
    def num_chunks(self) -> int:
        """Number of chunks."""
        return len(self._chunks)

    @property
    def num_rows(self) -> int:
        """Total number of rows across all chunks."""
        return sum(len(chunk) for chunk in self._chunks)

    @property
    def nbytes(self) -> int:
        """Total byte size of all chunks (including validity bitmaps)."""
        return sum(chunk.nbytes for chunk in self._chunks)

    @property
    def null_count(self) -> int:
        """Total number of null values across all chunks (cached)."""
        if self._null_count_cache is None:
            self._null_count_cache = sum(
                chunk.null_count for chunk in self._chunks
            )
        return self._null_count_cache

    @property
    def has_nulls(self) -> bool:
        """Whether this column contains any null values."""
        return self.null_count > 0

    @property
    def chunks(self) -> list[pa.Array]:
        """The underlying Arrow array chunks (read-only view)."""
        return list(self._chunks)

    # ---- Chunk operations ----

    def chunk(self, index: int) -> pa.Array:
        """Get a chunk by index."""
        if index < 0 or index >= len(self._chunks):
            raise IndexError(
                f"Chunk index {index} out of range [0, {len(self._chunks)})"
            )
        return self._chunks[index]

    def append_chunk(self, array: pa.Array) -> ChunkedColumn:
        """
        Return a new ChunkedColumn with an additional chunk appended.

        Args:
            array: Arrow array to append. Must have a type compatible with
                   this column's dtype.

        Returns:
            A new ChunkedColumn with the chunk added.
        """
        chunk_dtype = from_arrow(array.type)
        if chunk_dtype != self._dtype:
            raise TypeError(
                f"Cannot append array of type {chunk_dtype} to column "
                f"{self._name!r} of type {self._dtype}"
            )
        new_chunks = self._chunks + [array]
        return ChunkedColumn(self._name, self._dtype, new_chunks)

    def rechunk(self, target_chunks: int = 1) -> ChunkedColumn:
        """
        Consolidate into ``target_chunks`` chunks.

        When ``target_chunks=1``, this produces a single contiguous array
        (useful before GPU transfer to minimize kernel launch overhead).

        Returns:
            A new ChunkedColumn with consolidated chunks.
        """
        if len(self._chunks) <= target_chunks:
            return self

        # Combine all chunks into a single ChunkedArray, then rechunk
        combined = pa.chunked_array(self._chunks)

        if target_chunks == 1:
            single = combined.combine_chunks()
            return ChunkedColumn(self._name, self._dtype, [single])

        # For target > 1: split evenly
        total_rows = self.num_rows
        rows_per_chunk = (total_rows + target_chunks - 1) // target_chunks
        single = combined.combine_chunks()
        new_chunks = []
        offset = 0
        while offset < total_rows:
            length = min(rows_per_chunk, total_rows - offset)
            new_chunks.append(single.slice(offset, length))
            offset += length

        return ChunkedColumn(self._name, self._dtype, new_chunks)

    def slice(self, offset: int, length: Optional[int] = None) -> ChunkedColumn:
        """
        Return a new ChunkedColumn with a slice of the data.

        This may produce fewer or more chunks depending on chunk boundaries.
        """
        if length is None:
            length = self.num_rows - offset

        if offset < 0 or offset + length > self.num_rows:
            raise IndexError(
                f"Slice [{offset}:{offset + length}] out of range [0, {self.num_rows})"
            )

        new_chunks: list[pa.Array] = []
        remaining_offset = offset
        remaining_length = length

        for chunk in self._chunks:
            chunk_len = len(chunk)

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

        return ChunkedColumn(self._name, self._dtype, new_chunks)

    # ---- Conversion ----

    def to_pyarrow(self) -> pa.ChunkedArray:
        """Convert to a ``pyarrow.ChunkedArray``."""
        if not self._chunks:
            return pa.chunked_array([], type=self._dtype.arrow_type)
        return pa.chunked_array(self._chunks)

    @classmethod
    def from_pyarrow(cls, name: str, chunked: pa.ChunkedArray) -> ChunkedColumn:
        """Construct from a ``pyarrow.ChunkedArray``."""
        dtype = from_arrow(chunked.type)
        return cls(name, dtype, chunked.chunks)

    def to_pylist(self) -> list[Any]:
        """Convert to a Python list."""
        return self.to_pyarrow().to_pylist()

    @classmethod
    def from_pylist(cls, name: str, values: list[Any], dtype: DType) -> ChunkedColumn:
        """Construct from a Python list with an explicit type."""
        array = pa.array(values, type=dtype.arrow_type)
        return cls(name, dtype, [array])

    def rename(self, new_name: str) -> ChunkedColumn:
        """Return a new ChunkedColumn with a different name."""
        return ChunkedColumn(new_name, self._dtype, self._chunks)

    # ---- Display ----

    def __repr__(self) -> str:
        return (
            f"ChunkedColumn({self._name!r}, dtype={self._dtype}, "
            f"chunks={self.num_chunks}, rows={self.num_rows}, nulls={self.null_count})"
        )

    def __len__(self) -> int:
        return self.num_rows

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChunkedColumn):
            return NotImplemented
        if self._name != other._name or self._dtype != other._dtype:
            return False
        return self.to_pyarrow().equals(other.to_pyarrow())
