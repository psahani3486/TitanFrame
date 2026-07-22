from __future__ import annotations
import enum
from typing import Any, Optional, Sequence
import pyarrow as pa
from titanframe.core.dtypes import DType, from_arrow
from titanframe.core.schema import Schema

class DeviceLocation(enum.Enum):
    CPU = 'cpu'
    GPU_0 = 'gpu:0'
    GPU_1 = 'gpu:1'
    GPU_2 = 'gpu:2'
    GPU_3 = 'gpu:3'
    GPU_4 = 'gpu:4'
    GPU_5 = 'gpu:5'
    GPU_6 = 'gpu:6'
    GPU_7 = 'gpu:7'
    NVME = 'nvme'
    UNKNOWN = 'unknown'

    @classmethod
    def gpu(cls, device_id: int=0) -> DeviceLocation:
        return cls(f'gpu:{device_id}')

    @property
    def is_gpu(self) -> bool:
        return self.value.startswith('gpu:')

    @property
    def is_cpu(self) -> bool:
        return self == DeviceLocation.CPU

    @property
    def is_nvme(self) -> bool:
        return self == DeviceLocation.NVME

    @property
    def gpu_id(self) -> int:
        if not self.is_gpu:
            raise ValueError(f'{self} is not a GPU device')
        return int(self.value.split(':')[1])

class Chunk:
    __slots__ = ('_data', '_schema', '_device', '_chunk_id')

    def __init__(self, data: pa.RecordBatch, device: DeviceLocation=DeviceLocation.CPU, chunk_id: Optional[int]=None):
        if not isinstance(data, pa.RecordBatch):
            raise TypeError(f'Expected pyarrow.RecordBatch, got {type(data).__name__}')
        self._data = data
        self._schema = Schema.from_arrow(data.schema)
        self._device = device
        self._chunk_id = chunk_id

    @property
    def data(self) -> pa.RecordBatch:
        return self._data

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def device(self) -> DeviceLocation:
        return self._device

    @property
    def chunk_id(self) -> Optional[int]:
        return self._chunk_id

    @property
    def num_rows(self) -> int:
        return self._data.num_rows

    @property
    def num_columns(self) -> int:
        return self._data.num_columns

    @property
    def num_bytes(self) -> int:
        return self._data.nbytes

    @property
    def column_names(self) -> list[str]:
        return self._schema.names

    def column(self, name: str) -> pa.Array:
        idx = self._schema.index(name)
        return self._data.column(idx)

    def column_by_index(self, index: int) -> pa.Array:
        if index < 0 or index >= self.num_columns:
            raise IndexError(f'Column index {index} out of range [0, {self.num_columns})')
        return self._data.column(index)

    def columns(self, names: Sequence[str]) -> list[pa.Array]:
        return [self.column(name) for name in names]

    def select(self, names: Sequence[str]) -> Chunk:
        indices = [self._schema.index(name) for name in names]
        new_batch = self._data.select(indices)
        return Chunk(new_batch, self._device, self._chunk_id)

    def drop(self, names: Sequence[str]) -> Chunk:
        keep = [n for n in self.column_names if n not in set(names)]
        return self.select(keep)

    def rename(self, mapping: dict[str, str]) -> Chunk:
        new_names = [mapping.get(name, name) for name in self.column_names]
        new_batch = self._data.rename_columns(new_names)
        return Chunk(new_batch, self._device, self._chunk_id)

    def slice(self, offset: int, length: Optional[int]=None) -> Chunk:
        if length is None:
            length = self.num_rows - offset
        new_batch = self._data.slice(offset, length)
        return Chunk(new_batch, self._device, self._chunk_id)

    def append_column(self, name: str, array: pa.Array) -> Chunk:
        new_batch = self._data.append_column(name, array)
        return Chunk(new_batch, self._device, self._chunk_id)

    def with_device(self, device: DeviceLocation) -> Chunk:
        return Chunk(self._data, device, self._chunk_id)

    def to_arrow(self) -> pa.RecordBatch:
        return self._data

    @classmethod
    def from_arrow(cls, batch: pa.RecordBatch, device: DeviceLocation=DeviceLocation.CPU) -> Chunk:
        return cls(batch, device)

    @classmethod
    def from_pydict(cls, data: dict[str, list[Any]], device: DeviceLocation=DeviceLocation.CPU) -> Chunk:
        batch = pa.RecordBatch.from_pydict(data)
        return cls(batch, device)

    def to_pydict(self) -> dict[str, list[Any]]:
        return self._data.to_pydict()

    def __repr__(self) -> str:
        return f'Chunk(rows={self.num_rows}, cols={self.num_columns}, bytes={self.num_bytes}, device={self._device.value})'

    def __len__(self) -> int:
        return self.num_rows

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return self._data.equals(other._data)
