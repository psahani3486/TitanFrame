"""
TitanFrame Chunk
================

A Chunk is a thin wrapper around a ``pyarrow.RecordBatch`` — the fundamental
unit of data that flows through the execution engine. Every physical operator
produces and consumes chunks.

Key properties:
    - Tracks **device location** (CPU, GPU_0, GPU_1, NVMe) so the memory
      manager knows where the data lives.
    - Supports **lazy deserialization**: metadata (schema, row count) is
      available immediately; the underlying buffers are materialized on demand.
    - All mutations return new Chunk instances (immutable design).

Example::

    >>> import pyarrow as pa
    >>> from titanframe.core.chunk import Chunk
    >>> batch = pa.record_batch({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    >>> chunk = Chunk(batch)
    >>> chunk.num_rows
    3
    >>> chunk.schema
    Schema({'a': Int64, 'b': Utf8})
"""

from __future__ import annotations

import enum
from typing import Any, Optional, Sequence

import pyarrow as pa

from titanframe.core.dtypes import DType, from_arrow
from titanframe.core.schema import Schema


class DeviceLocation(enum.Enum):
    """Where a chunk's data buffers currently reside."""
    CPU = "cpu"
    GPU_0 = "gpu:0"
    GPU_1 = "gpu:1"
    GPU_2 = "gpu:2"
    GPU_3 = "gpu:3"
    GPU_4 = "gpu:4"
    GPU_5 = "gpu:5"
    GPU_6 = "gpu:6"
    GPU_7 = "gpu:7"
    NVME = "nvme"
    UNKNOWN = "unknown"

    @classmethod
    def gpu(cls, device_id: int = 0) -> DeviceLocation:
        """Get the GPU device location for a given device ID."""
        return cls(f"gpu:{device_id}")

    @property
    def is_gpu(self) -> bool:
        return self.value.startswith("gpu:")

    @property
    def is_cpu(self) -> bool:
        return self == DeviceLocation.CPU

    @property
    def is_nvme(self) -> bool:
        return self == DeviceLocation.NVME

    @property
    def gpu_id(self) -> int:
        """Extract the GPU device ID. Raises ValueError if not a GPU location."""
        if not self.is_gpu:
            raise ValueError(f"{self} is not a GPU device")
        return int(self.value.split(":")[1])


class Chunk:
    """
    A wrapper around ``pyarrow.RecordBatch`` with device tracking.

    This is the atomic unit of data in TitanFrame. The execution engine
    processes data one Chunk at a time, enabling streaming/out-of-core
    execution.

    Args:
        data: A ``pyarrow.RecordBatch`` containing the columnar data.
        device: The device where the data currently resides.
        chunk_id: Optional unique identifier for memory tracking.

    Invariants:
        - ``data`` is always a valid ``pyarrow.RecordBatch``.
        - ``schema`` is derived from ``data`` and cached.
        - ``device`` reflects the current physical location of buffers.
    """

    __slots__ = ("_data", "_schema", "_device", "_chunk_id")

    def __init__(
        self,
        data: pa.RecordBatch,
        device: DeviceLocation = DeviceLocation.CPU,
        chunk_id: Optional[int] = None,
    ):
        if not isinstance(data, pa.RecordBatch):
            raise TypeError(f"Expected pyarrow.RecordBatch, got {type(data).__name__}")
        self._data = data
        self._schema = Schema.from_arrow(data.schema)
        self._device = device
        self._chunk_id = chunk_id

    # ---- Properties ----

    @property
    def data(self) -> pa.RecordBatch:
        """The underlying Arrow RecordBatch."""
        return self._data

    @property
    def schema(self) -> Schema:
        """The schema of this chunk."""
        return self._schema

    @property
    def device(self) -> DeviceLocation:
        """Where this chunk's data currently resides."""
        return self._device

    @property
    def chunk_id(self) -> Optional[int]:
        """Unique identifier for memory tracking."""
        return self._chunk_id

    @property
    def num_rows(self) -> int:
        """Number of rows in this chunk."""
        return self._data.num_rows

    @property
    def num_columns(self) -> int:
        """Number of columns in this chunk."""
        return self._data.num_columns

    @property
    def num_bytes(self) -> int:
        """
        Total byte size of all buffers in this chunk.

        This includes validity bitmaps, offsets, and data buffers.
        """
        return self._data.nbytes

    @property
    def column_names(self) -> list[str]:
        """Column names in order."""
        return self._schema.names

    # ---- Column access ----

    def column(self, name: str) -> pa.Array:
        """
        Get a column by name as a ``pyarrow.Array``.

        Raises:
            KeyError: If the column doesn't exist.
        """
        idx = self._schema.index(name)
        return self._data.column(idx)

    def column_by_index(self, index: int) -> pa.Array:
        """Get a column by positional index."""
        if index < 0 or index >= self.num_columns:
            raise IndexError(f"Column index {index} out of range [0, {self.num_columns})")
        return self._data.column(index)

    def columns(self, names: Sequence[str]) -> list[pa.Array]:
        """Get multiple columns by name."""
        return [self.column(name) for name in names]

    # ---- Transformations (return new Chunk) ----

    def select(self, names: Sequence[str]) -> Chunk:
        """
        Return a new chunk with only the specified columns.

        This is a zero-copy operation (shares underlying Arrow buffers).
        """
        indices = [self._schema.index(name) for name in names]
        new_batch = self._data.select(indices)
        return Chunk(new_batch, self._device, self._chunk_id)

    def drop(self, names: Sequence[str]) -> Chunk:
        """Return a new chunk without the specified columns."""
        keep = [n for n in self.column_names if n not in set(names)]
        return self.select(keep)

    def rename(self, mapping: dict[str, str]) -> Chunk:
        """Return a new chunk with columns renamed."""
        new_names = [mapping.get(name, name) for name in self.column_names]
        new_batch = self._data.rename_columns(new_names)
        return Chunk(new_batch, self._device, self._chunk_id)

    def slice(self, offset: int, length: Optional[int] = None) -> Chunk:
        """
        Return a zero-copy slice of this chunk.

        Args:
            offset: Start row index.
            length: Number of rows. If None, takes all remaining rows.
        """
        if length is None:
            length = self.num_rows - offset
        new_batch = self._data.slice(offset, length)
        return Chunk(new_batch, self._device, self._chunk_id)

    def append_column(self, name: str, array: pa.Array) -> Chunk:
        """Return a new chunk with an additional column."""
        new_batch = self._data.append_column(name, array)
        return Chunk(new_batch, self._device, self._chunk_id)

    def with_device(self, device: DeviceLocation) -> Chunk:
        """Return a new Chunk pointing to the same data but with a different device tag."""
        return Chunk(self._data, device, self._chunk_id)

    # ---- Serialization ----

    def to_arrow(self) -> pa.RecordBatch:
        """Return the underlying Arrow RecordBatch."""
        return self._data

    @classmethod
    def from_arrow(
        cls,
        batch: pa.RecordBatch,
        device: DeviceLocation = DeviceLocation.CPU,
    ) -> Chunk:
        """Construct a Chunk from an Arrow RecordBatch."""
        return cls(batch, device)

    @classmethod
    def from_pydict(
        cls,
        data: dict[str, list[Any]],
        device: DeviceLocation = DeviceLocation.CPU,
    ) -> Chunk:
        """
        Construct a Chunk from a Python dictionary.

        Example::

            >>> Chunk.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        """
        batch = pa.RecordBatch.from_pydict(data)
        return cls(batch, device)

    def to_pydict(self) -> dict[str, list[Any]]:
        """Convert to a Python dictionary of lists."""
        return self._data.to_pydict()

    # ---- Display ----

    def __repr__(self) -> str:
        return (
            f"Chunk(rows={self.num_rows}, cols={self.num_columns}, "
            f"bytes={self.num_bytes}, device={self._device.value})"
        )

    def __len__(self) -> int:
        return self.num_rows

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return self._data.equals(other._data)
