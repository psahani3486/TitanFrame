"""
TitanFrame Top-Level Functions
================================

Module-level convenience functions like ``tf.concat()``, ``tf.read_csv()``,
``tf.scan_csv()``, etc. — the primary entry points for users.

Example::

    >>> import titanframe as tf
    >>> df = tf.read_csv("data.csv")
    >>> lf = tf.scan_csv("data.csv")
    >>> combined = tf.concat([df1, df2])
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from titanframe.api.dataframe import DataFrame
from titanframe.api.lazyframe import LazyFrame
from titanframe.core.table import Table
from titanframe.plan.logical.scan import Scan, ScanFormat



def read_csv(
    path: str,
    chunk_size: int = 65536,
    columns: Optional[list[str]] = None,
    **kwargs: Any,
) -> DataFrame:
    """
    Read a CSV file into an eager DataFrame.

    Args:
        path: Path to the CSV file.
        chunk_size: Number of rows per chunk.
        columns: Optional list of columns to read (projection).

    Returns:
        A :class:`DataFrame`.

    Example::

        >>> df = tf.read_csv("sales.csv")
        >>> df = tf.read_csv("sales.csv", columns=["region", "revenue"])
    """
    from titanframe.io.csv import read_csv_to_table
    table = read_csv_to_table(path, chunk_size=chunk_size, columns=columns, **kwargs)
    return DataFrame._from_table(table)


def scan_csv(
    path: str,
    chunk_size: int = 65536,
) -> LazyFrame:
    """
    Create a lazy scan of a CSV file.

    No data is read until ``.collect()`` is called.

    Args:
        path: Path to the CSV file.
        chunk_size: Number of rows per chunk when reading.

    Returns:
        A :class:`LazyFrame`.

    Example::

        >>> lf = tf.scan_csv("huge_dataset.csv")
        >>> result = lf.filter(col("x") > 10).collect()
    """
    from titanframe.io.csv import infer_csv_schema
    schema = infer_csv_schema(path)
    scan = Scan(path, ScanFormat.CSV, schema, chunk_size=chunk_size)
    return LazyFrame(scan)


def read_ipc(path: str) -> DataFrame:
    """
    Read an Arrow IPC file into an eager DataFrame.
    """
    table = Table.from_ipc_file(path)
    return DataFrame._from_table(table)

def read_parquet(path: str, **kwargs) -> LazyFrame:
    """
    Create a lazy scan of a Parquet file.
    """
    from titanframe.io.parquet import read_parquet as _read_parquet
    scan_node = _read_parquet(path, **kwargs)
    return LazyFrame(scan_node)

scan_parquet = read_parquet


def scan_ipc(path: str) -> LazyFrame:
    """
    Create a lazy scan of an Arrow IPC file.

    Args:
        path: Path to the Arrow IPC file.

    Returns:
        A :class:`LazyFrame`.
    """
    from titanframe.io.arrow_ipc import infer_ipc_schema
    schema = infer_ipc_schema(path)
    scan = Scan(path, ScanFormat.ARROW_IPC, schema)
    return LazyFrame(scan)



def from_dict(data: dict[str, list]) -> DataFrame:
    """
    Create a DataFrame from a Python dictionary.

    Example::

        >>> df = tf.from_dict({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    """
    return DataFrame(data)


def from_arrow(table: Any) -> DataFrame:
    """Create a DataFrame from a PyArrow Table."""
    return DataFrame(table)


def from_pandas(df: Any) -> DataFrame:
    """Create a DataFrame from a Pandas DataFrame."""
    return DataFrame(df)



def concat(
    dfs: Sequence[DataFrame],
    how: str = "vertical",
) -> DataFrame:
    """
    Concatenate multiple DataFrames.

    Args:
        dfs: List of DataFrames to concatenate.
        how: ``"vertical"`` (vstack) or ``"horizontal"`` (hstack).

    Returns:
        A combined :class:`DataFrame`.

    Example::

        >>> combined = tf.concat([df1, df2, df3])
    """
    if not dfs:
        raise ValueError("Cannot concatenate empty list of DataFrames")

    if how == "vertical":
        result = dfs[0]
        for df in dfs[1:]:
            result = result.vstack(df)
        return result
    elif how == "horizontal":
        import pyarrow as pa
        tables = [df.to_arrow() for df in dfs]
        all_columns: dict[str, Any] = {}
        for table in tables:
            for name in table.column_names:
                all_columns[name] = table.column(name).combine_chunks()
        return DataFrame(pa.table(all_columns))
    else:
        raise ValueError(f"Unknown concat mode: {how!r}. Use 'vertical' or 'horizontal'.")


def merge(
    left: DataFrame,
    right: DataFrame,
    on: str | list[str],
    how: str = "inner",
    suffix: str = "_right",
) -> DataFrame:
    """
    Merge two DataFrames (Pandas-style alias for join).

    Example::

        >>> result = tf.merge(orders, customers, on="customer_id", how="left")
    """
    return left.join(right, on=on, how=how, suffix=suffix)
