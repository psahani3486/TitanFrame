from __future__ import annotations
from typing import Any, Optional, Sequence
from titanframe.api.dataframe import DataFrame
from titanframe.api.lazyframe import LazyFrame
from titanframe.core.table import Table
from titanframe.plan.logical.scan import Scan, ScanFormat

def read_csv(path: str, chunk_size: int=65536, columns: Optional[list[str]]=None, **kwargs: Any) -> DataFrame:
    from titanframe.io.csv import read_csv_to_table
    table = read_csv_to_table(path, chunk_size=chunk_size, columns=columns, **kwargs)
    return DataFrame._from_table(table)

def scan_csv(path: str, chunk_size: int=65536) -> LazyFrame:
    from titanframe.io.csv import infer_csv_schema
    schema = infer_csv_schema(path)
    scan = Scan(path, ScanFormat.CSV, schema, chunk_size=chunk_size)
    return LazyFrame(scan)

def read_ipc(path: str) -> DataFrame:
    table = Table.from_ipc_file(path)
    return DataFrame._from_table(table)

def read_parquet(path: str, **kwargs) -> LazyFrame:
    from titanframe.io.parquet import read_parquet as _read_parquet
    scan_node = _read_parquet(path, **kwargs)
    return LazyFrame(scan_node)
scan_parquet = read_parquet

def scan_ipc(path: str) -> LazyFrame:
    from titanframe.io.arrow_ipc import infer_ipc_schema
    schema = infer_ipc_schema(path)
    scan = Scan(path, ScanFormat.ARROW_IPC, schema)
    return LazyFrame(scan)

def from_dict(data: dict[str, list]) -> DataFrame:
    return DataFrame(data)

def from_arrow(table: Any) -> DataFrame:
    return DataFrame(table)

def from_pandas(df: Any) -> DataFrame:
    return DataFrame(df)

def concat(dfs: Sequence[DataFrame], how: str='vertical') -> DataFrame:
    if not dfs:
        raise ValueError('Cannot concatenate empty list of DataFrames')
    if how == 'vertical':
        result = dfs[0]
        for df in dfs[1:]:
            result = result.vstack(df)
        return result
    elif how == 'horizontal':
        import pyarrow as pa
        tables = [df.to_arrow() for df in dfs]
        all_columns: dict[str, Any] = {}
        for table in tables:
            for name in table.column_names:
                all_columns[name] = table.column(name).combine_chunks()
        return DataFrame(pa.table(all_columns))
    else:
        raise ValueError(f"Unknown concat mode: {how!r}. Use 'vertical' or 'horizontal'.")

def merge(left: DataFrame, right: DataFrame, on: str | list[str], how: str='inner', suffix: str='_right') -> DataFrame:
    return left.join(right, on=on, how=how, suffix=suffix)
