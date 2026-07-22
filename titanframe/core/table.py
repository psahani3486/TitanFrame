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
    __slots__ = ('_schema', '_chunks')

    def __init__(self, schema: Schema, chunks: list[Chunk] | None=None):
        self._schema = schema
        self._chunks = chunks or []
        for i, chunk in enumerate(self._chunks):
            if chunk.schema != schema:
                raise SchemaError(f'Chunk {i} schema {chunk.schema} does not match table schema {schema}')

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def num_rows(self) -> int:
        return sum((chunk.num_rows for chunk in self._chunks))

    @property
    def num_columns(self) -> int:
        return self._schema.num_columns

    @property
    def num_bytes(self) -> int:
        return sum((chunk.num_bytes for chunk in self._chunks))

    @property
    def column_names(self) -> list[str]:
        return self._schema.names

    @property
    def dtypes(self) -> list[DType]:
        return self._schema.dtypes

    @property
    def is_empty(self) -> bool:
        return self.num_rows == 0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_rows, self.num_columns)

    def column(self, name: str) -> ChunkedColumn:
        idx = self._schema.index(name)
        dtype = self._schema[name]
        arrays = [chunk.column_by_index(idx) for chunk in self._chunks]
        return ChunkedColumn(name, dtype, arrays)

    def columns(self, names: Sequence[str]) -> list[ChunkedColumn]:
        return [self.column(name) for name in names]

    def select(self, names: Sequence[str]) -> Table:
        new_schema = self._schema.select(names)
        new_chunks = [chunk.select(names) for chunk in self._chunks]
        return Table(new_schema, new_chunks)

    def drop(self, names: Sequence[str]) -> Table:
        new_schema = self._schema.drop(names)
        keep = new_schema.names
        new_chunks = [chunk.select(keep) for chunk in self._chunks]
        return Table(new_schema, new_chunks)

    def rename(self, mapping: dict[str, str]) -> Table:
        new_schema = self._schema.rename(mapping)
        new_chunks = [chunk.rename(mapping) for chunk in self._chunks]
        return Table(new_schema, new_chunks)

    def slice(self, offset: int, length: Optional[int]=None) -> Table:
        if length is None:
            length = self.num_rows - offset
        if offset < 0 or offset + length > self.num_rows:
            raise IndexError(f'Slice [{offset}:{offset + length}] out of range [0, {self.num_rows})')
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

    def head(self, n: int=5) -> Table:
        return self.slice(0, min(n, self.num_rows))

    def tail(self, n: int=5) -> Table:
        start = max(0, self.num_rows - n)
        return self.slice(start)

    def append_column(self, name: str, column: ChunkedColumn) -> Table:
        if column.num_rows != self.num_rows:
            raise ValueError(f'Column has {column.num_rows} rows but table has {self.num_rows}')
        if column.num_chunks != self.num_chunks:
            raise ValueError(f'Column has {column.num_chunks} chunks but table has {self.num_chunks}. Rechunk the column first.')
        new_schema = self._schema.append(name, column.dtype)
        new_chunks = []
        for table_chunk, col_chunk in zip(self._chunks, column.chunks):
            new_batch = table_chunk.data.append_column(name, col_chunk)
            new_chunks.append(Chunk(new_batch, table_chunk.device))
        return Table(new_schema, new_chunks)

    def vstack(self, other: Table) -> Table:
        self._schema.assert_compatible(other._schema, context='vstack')
        new_chunks = self._chunks + other._chunks
        return Table(self._schema, new_chunks)

    def rechunk(self, target_chunks: int=1) -> Table:
        if len(self._chunks) <= target_chunks:
            return self
        arrow_table = self.to_arrow()
        combined = arrow_table.combine_chunks()
        if target_chunks == 1:
            batch = combined.to_batches()[0] if combined.num_rows > 0 else pa.record_batch({name: pa.array([], type=self._schema[name].arrow_type) for name in self._schema.names})
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
        return iter(self._chunks)

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        for chunk in self._chunks:
            pydict = chunk.to_pydict()
            for i in range(chunk.num_rows):
                yield {name: values[i] for name, values in pydict.items()}

    def to_arrow(self) -> pa.Table:
        if not self._chunks:
            return pa.table({name: pa.array([], type=dtype.arrow_type) for name, dtype in zip(self._schema.names, self._schema.dtypes)})
        batches = [chunk.data for chunk in self._chunks]
        return pa.Table.from_batches(batches, schema=self._schema.to_arrow())

    @classmethod
    def from_arrow(cls, arrow_table: pa.Table, chunk_size: Optional[int]=None) -> Table:
        schema = Schema.from_arrow(arrow_table.schema)
        if chunk_size is not None:
            batches = arrow_table.to_batches(max_chunksize=chunk_size)
        else:
            batches = arrow_table.to_batches()
        chunks = [Chunk(batch) for batch in batches if batch.num_rows > 0]
        return cls(schema, chunks)

    def to_pandas(self) -> Any:
        return self.to_arrow().to_pandas()

    @classmethod
    def from_pandas(cls, df: Any, chunk_size: Optional[int]=None) -> Table:
        arrow_table = pa.Table.from_pandas(df, preserve_index=False)
        return cls.from_arrow(arrow_table, chunk_size=chunk_size)

    def to_pydict(self) -> dict[str, list[Any]]:
        return self.to_arrow().to_pydict()

    @classmethod
    def from_pydict(cls, data: dict[str, list[Any]], chunk_size: Optional[int]=None) -> Table:
        arrow_table = pa.table(data)
        return cls.from_arrow(arrow_table, chunk_size=chunk_size)

    def to_ipc_bytes(self) -> bytes:
        sink = io.BytesIO()
        writer = ipc.new_stream(sink, self._schema.to_arrow())
        for chunk in self._chunks:
            writer.write_batch(chunk.data)
        writer.close()
        return sink.getvalue()

    @classmethod
    def from_ipc_bytes(cls, data: bytes) -> Table:
        reader = ipc.open_stream(data)
        schema = Schema.from_arrow(reader.schema)
        chunks = []
        for batch in reader:
            chunks.append(Chunk(batch))
        return cls(schema, chunks)

    def to_ipc_file(self, path: str) -> None:
        arrow_table = self.to_arrow()
        writer = ipc.new_file(path, arrow_table.schema)
        for batch in arrow_table.to_batches():
            writer.write_batch(batch)
        writer.close()

    @classmethod
    def from_ipc_file(cls, path: str) -> Table:
        reader = ipc.open_file(path)
        schema = Schema.from_arrow(reader.schema)
        chunks = []
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            chunks.append(Chunk(batch))
        return cls(schema, chunks)

    def __repr__(self) -> str:
        lines = [f'Table(rows={self.num_rows}, cols={self.num_columns}, chunks={self.num_chunks})']
        lines.append(f'Schema: {self._schema}')
        if self.num_rows > 0:
            preview = self.head(min(5, self.num_rows))
            pydict = preview.to_pydict()
            col_widths = {}
            for name in self.column_names:
                values = [str(v) for v in pydict[name]]
                col_widths[name] = max(len(name), max((len(v) for v in values)) if values else 0)
            header = ' | '.join((name.ljust(col_widths[name]) for name in self.column_names))
            separator = '-+-'.join(('-' * col_widths[name] for name in self.column_names))
            lines.append(header)
            lines.append(separator)
            for i in range(preview.num_rows):
                row = ' | '.join((str(pydict[name][i]).ljust(col_widths[name]) for name in self.column_names))
                lines.append(row)
            if self.num_rows > 5:
                lines.append(f'... ({self.num_rows - 5} more rows)')
        return '\n'.join(lines)

    def __len__(self) -> int:
        return self.num_rows

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Table):
            return NotImplemented
        if self._schema != other._schema:
            return False
        return self.to_arrow().equals(other.to_arrow())
