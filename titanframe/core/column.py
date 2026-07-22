from __future__ import annotations
from typing import Any, Optional, Sequence
import pyarrow as pa
from titanframe.core.dtypes import DType, from_arrow

class ChunkedColumn:
    __slots__ = ('_name', '_dtype', '_chunks', '_null_count_cache')

    def __init__(self, name: str, dtype: DType, chunks: Sequence[pa.Array] | None=None):
        self._name = name
        self._dtype = dtype
        self._chunks: list[pa.Array] = list(chunks) if chunks else []
        self._null_count_cache: Optional[int] = None
        for i, chunk in enumerate(self._chunks):
            chunk_dtype = from_arrow(chunk.type)
            if chunk_dtype != dtype:
                raise TypeError(f'Chunk {i} has type {chunk_dtype} but column {name!r} expects {dtype}')

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def num_rows(self) -> int:
        return sum((len(chunk) for chunk in self._chunks))

    @property
    def nbytes(self) -> int:
        return sum((chunk.nbytes for chunk in self._chunks))

    @property
    def null_count(self) -> int:
        if self._null_count_cache is None:
            self._null_count_cache = sum((chunk.null_count for chunk in self._chunks))
        return self._null_count_cache

    @property
    def has_nulls(self) -> bool:
        return self.null_count > 0

    @property
    def chunks(self) -> list[pa.Array]:
        return list(self._chunks)

    def chunk(self, index: int) -> pa.Array:
        if index < 0 or index >= len(self._chunks):
            raise IndexError(f'Chunk index {index} out of range [0, {len(self._chunks)})')
        return self._chunks[index]

    def append_chunk(self, array: pa.Array) -> ChunkedColumn:
        chunk_dtype = from_arrow(array.type)
        if chunk_dtype != self._dtype:
            raise TypeError(f'Cannot append array of type {chunk_dtype} to column {self._name!r} of type {self._dtype}')
        new_chunks = self._chunks + [array]
        return ChunkedColumn(self._name, self._dtype, new_chunks)

    def rechunk(self, target_chunks: int=1) -> ChunkedColumn:
        if len(self._chunks) <= target_chunks:
            return self
        combined = pa.chunked_array(self._chunks)
        if target_chunks == 1:
            single = combined.combine_chunks()
            return ChunkedColumn(self._name, self._dtype, [single])
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

    def slice(self, offset: int, length: Optional[int]=None) -> ChunkedColumn:
        if length is None:
            length = self.num_rows - offset
        if offset < 0 or offset + length > self.num_rows:
            raise IndexError(f'Slice [{offset}:{offset + length}] out of range [0, {self.num_rows})')
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

    def to_pyarrow(self) -> pa.ChunkedArray:
        if not self._chunks:
            return pa.chunked_array([], type=self._dtype.arrow_type)
        return pa.chunked_array(self._chunks)

    @classmethod
    def from_pyarrow(cls, name: str, chunked: pa.ChunkedArray) -> ChunkedColumn:
        dtype = from_arrow(chunked.type)
        return cls(name, dtype, chunked.chunks)

    def to_pylist(self) -> list[Any]:
        return self.to_pyarrow().to_pylist()

    @classmethod
    def from_pylist(cls, name: str, values: list[Any], dtype: DType) -> ChunkedColumn:
        array = pa.array(values, type=dtype.arrow_type)
        return cls(name, dtype, [array])

    def rename(self, new_name: str) -> ChunkedColumn:
        return ChunkedColumn(new_name, self._dtype, self._chunks)

    def __repr__(self) -> str:
        return f'ChunkedColumn({self._name!r}, dtype={self._dtype}, chunks={self.num_chunks}, rows={self.num_rows}, nulls={self.null_count})'

    def __len__(self) -> int:
        return self.num_rows

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChunkedColumn):
            return NotImplemented
        if self._name != other._name or self._dtype != other._dtype:
            return False
        return self.to_pyarrow().equals(other.to_pyarrow())
