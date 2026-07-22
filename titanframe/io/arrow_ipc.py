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
    reader = ipc.open_file(str(path))
    return Schema.from_arrow(reader.schema)

def read_ipc(path: str | Path) -> Table:
    return Table.from_ipc_file(str(path))

def read_ipc_streaming(path: str | Path) -> Iterator[Chunk]:
    with open(str(path), 'rb') as f:
        reader = ipc.open_stream(f)
        for batch in reader:
            yield Chunk(batch)

def read_ipc_bytes(data: bytes) -> Table:
    return Table.from_ipc_bytes(data)

def write_ipc(table: Table, path: str | Path, compression: Optional[str]=None) -> None:
    path = str(path)
    arrow_table = table.to_arrow()
    options = None
    if compression:
        options = ipc.IpcWriteOptions(compression=pa.Codec(compression))
    writer = ipc.new_file(path, arrow_table.schema, options=options)
    for batch in arrow_table.to_batches():
        writer.write_batch(batch)
    writer.close()

def write_ipc_streaming(table: Table, path: str | Path, compression: Optional[str]=None) -> None:
    path = str(path)
    arrow_schema = table.schema.to_arrow()
    options = None
    if compression:
        options = ipc.IpcWriteOptions(compression=pa.Codec(compression))
    with open(path, 'wb') as f:
        writer = ipc.new_stream(f, arrow_schema, options=options)
        for chunk in table.iter_chunks():
            writer.write_batch(chunk.data)
        writer.close()

def to_ipc_bytes(table: Table) -> bytes:
    return table.to_ipc_bytes()

def write_ipc_batch(path: str | Path, schema: Schema) -> IPCBatchWriter:
    return IPCBatchWriter(str(path), schema)

class IPCBatchWriter:
    __slots__ = ('_path', '_schema', '_writer', '_file')

    def __init__(self, path: str, schema: Schema):
        self._path = path
        self._schema = schema
        self._writer: Optional[ipc.RecordBatchFileWriter] = None
        self._file = None

    def __enter__(self) -> IPCBatchWriter:
        self._file = open(self._path, 'wb')
        self._writer = ipc.new_file(self._file, self._schema.to_arrow())
        return self

    def write(self, chunk: Chunk) -> None:
        if self._writer is None:
            raise RuntimeError("Writer not opened. Use 'with' statement.")
        self._writer.write_batch(chunk.data)

    def __exit__(self, *args: object) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._file is not None:
            self._file.close()

    def close(self) -> None:
        self.__exit__()
