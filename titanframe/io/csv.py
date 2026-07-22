"""
TitanFrame CSV I/O
===================

Chunked CSV reader/writer with schema inference and streaming support.

Uses ``pyarrow.csv`` under the hood for high-performance parsing with
automatic type inference, configurable chunk sizes, and column selection.

Example::

    >>> from titanframe.io.csv import read_csv_to_table
    >>> table = read_csv_to_table("data.csv", chunk_size=100_000)
    >>> table.num_rows
    1000000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence

import pyarrow as pa
import pyarrow.csv as pcsv

from titanframe.core.chunk import Chunk
from titanframe.core.schema import Schema
from titanframe.core.table import Table


def infer_csv_schema(
    path: str | Path,
    sample_rows: int = 1000,
    delimiter: str = ",",
) -> Schema:
    """
    Infer the schema from a CSV file by reading a small sample.

    Args:
        path: Path to the CSV file.
        sample_rows: Number of rows to sample for type inference.
        delimiter: Column delimiter character.

    Returns:
        Inferred :class:`Schema`.
    """
    parse_options = pcsv.ParseOptions(delimiter=delimiter)
    read_options = pcsv.ReadOptions(block_size=1024 * 1024)

    reader = pcsv.open_csv(
        str(path),
        parse_options=parse_options,
        read_options=read_options,
    )

    first_batch = next(reader)
    arrow_schema = first_batch.schema
    return Schema.from_arrow(arrow_schema)


def read_csv_to_table(
    path: str | Path,
    chunk_size: int = 65536,
    columns: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    has_header: bool = True,
    skip_rows: int = 0,
    null_values: Optional[list[str]] = None,
    **kwargs: Any,
) -> Table:
    """
    Read a CSV file into a TitanFrame Table.

    Supports chunked reading for out-of-core processing.

    Args:
        path: Path to the CSV file.
        chunk_size: Number of rows per chunk (controls memory granularity).
        columns: Optional list of columns to read (projection pushdown).
        delimiter: Column delimiter.
        has_header: Whether the file has a header row.
        skip_rows: Number of rows to skip at the beginning.
        null_values: Additional strings to treat as null.

    Returns:
        A :class:`Table` with the CSV data.

    Example::

        >>> table = read_csv_to_table("sales.csv", chunk_size=100_000)
        >>> table = read_csv_to_table("data.csv", columns=["id", "value"])
    """
    path = str(path)

    parse_options = pcsv.ParseOptions(delimiter=delimiter)

    read_options = pcsv.ReadOptions(
        skip_rows=skip_rows,
        block_size=max(chunk_size * 256, 1024 * 1024),
    )
    if not has_header:
        read_options.autogenerate_column_names = True

    convert_options = pcsv.ConvertOptions()
    if columns:
        convert_options.include_columns = list(columns)
    if null_values:
        convert_options.null_values = null_values

    arrow_table = pcsv.read_csv(
        path,
        parse_options=parse_options,
        read_options=read_options,
        convert_options=convert_options,
    )

    return Table.from_arrow(arrow_table, chunk_size=chunk_size)


def read_csv_streaming(
    path: str | Path,
    chunk_size: int = 65536,
    columns: Optional[Sequence[str]] = None,
    delimiter: str = ",",
):
    """
    Stream a CSV file as an iterator of Chunks.

    This is the true out-of-core reader — each chunk is yielded
    individually, so only one chunk's worth of data is in memory at a time.

    Args:
        path: Path to the CSV file.
        chunk_size: Rows per yielded chunk.
        columns: Optional column projection.
        delimiter: Column delimiter.

    Yields:
        :class:`Chunk` instances, one at a time.
    """
    parse_options = pcsv.ParseOptions(delimiter=delimiter)
    read_options = pcsv.ReadOptions(block_size=chunk_size * 256)

    convert_options = pcsv.ConvertOptions()
    if columns:
        convert_options.include_columns = list(columns)

    reader = pcsv.open_csv(
        str(path),
        parse_options=parse_options,
        read_options=read_options,
        convert_options=convert_options,
    )

    for batch in reader:
        offset = 0
        while offset < batch.num_rows:
            length = min(chunk_size, batch.num_rows - offset)
            sub_batch = batch.slice(offset, length)
            yield Chunk(sub_batch)
            offset += length


def write_csv(
    table: Table,
    path: str | Path,
    delimiter: str = ",",
    include_header: bool = True,
) -> None:
    """
    Write a TitanFrame Table to a CSV file.

    Args:
        table: Table to write.
        path: Output file path.
        delimiter: Column delimiter.
        include_header: Whether to write a header row.
    """
    path = str(path)
    arrow_table = table.to_arrow()
    write_options = pcsv.WriteOptions(
        delimiter=delimiter,
        include_header=include_header,
    )
    pcsv.write_csv(arrow_table, path, write_options=write_options)
