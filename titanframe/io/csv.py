from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional, Sequence
import pyarrow as pa
import pyarrow.csv as pcsv
from titanframe.core.chunk import Chunk
from titanframe.core.schema import Schema
from titanframe.core.table import Table

def infer_csv_schema(path: str | Path, sample_rows: int=1000, delimiter: str=',') -> Schema:
    parse_options = pcsv.ParseOptions(delimiter=delimiter)
    read_options = pcsv.ReadOptions(block_size=1024 * 1024)
    reader = pcsv.open_csv(str(path), parse_options=parse_options, read_options=read_options)
    first_batch = next(reader)
    arrow_schema = first_batch.schema
    return Schema.from_arrow(arrow_schema)

def read_csv_to_table(path: str | Path, chunk_size: int=65536, columns: Optional[Sequence[str]]=None, delimiter: str=',', has_header: bool=True, skip_rows: int=0, null_values: Optional[list[str]]=None, **kwargs: Any) -> Table:
    path = str(path)
    parse_options = pcsv.ParseOptions(delimiter=delimiter)
    read_options = pcsv.ReadOptions(skip_rows=skip_rows, block_size=max(chunk_size * 256, 1024 * 1024))
    if not has_header:
        read_options.autogenerate_column_names = True
    convert_options = pcsv.ConvertOptions()
    if columns:
        convert_options.include_columns = list(columns)
    if null_values:
        convert_options.null_values = null_values
    arrow_table = pcsv.read_csv(path, parse_options=parse_options, read_options=read_options, convert_options=convert_options)
    return Table.from_arrow(arrow_table, chunk_size=chunk_size)

def read_csv_streaming(path: str | Path, chunk_size: int=65536, columns: Optional[Sequence[str]]=None, delimiter: str=','):
    parse_options = pcsv.ParseOptions(delimiter=delimiter)
    read_options = pcsv.ReadOptions(block_size=chunk_size * 256)
    convert_options = pcsv.ConvertOptions()
    if columns:
        convert_options.include_columns = list(columns)
    reader = pcsv.open_csv(str(path), parse_options=parse_options, read_options=read_options, convert_options=convert_options)
    for batch in reader:
        offset = 0
        while offset < batch.num_rows:
            length = min(chunk_size, batch.num_rows - offset)
            sub_batch = batch.slice(offset, length)
            yield Chunk(sub_batch)
            offset += length

def write_csv(table: Table, path: str | Path, delimiter: str=',', include_header: bool=True) -> None:
    path = str(path)
    arrow_table = table.to_arrow()
    write_options = pcsv.WriteOptions(delimiter=delimiter, include_header=include_header)
    pcsv.write_csv(arrow_table, path, write_options=write_options)
