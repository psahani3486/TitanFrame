"""
TitanFrame — A Pandas-like DataFrame library for out-of-core, GPU-accelerated computation.

Core public API:
    - DataFrame: Eager evaluation (Pandas-like)
    - LazyFrame: Deferred evaluation with query optimization
    - Series: Single column extraction
    - col(): Column reference expression
    - lit(): Literal value expression
    - read_csv(), scan_csv(): CSV I/O
    - read_ipc(), scan_ipc(): Arrow IPC I/O
    - concat(), merge(): DataFrame combination
    - from_dict(), from_arrow(), from_pandas(): Construction helpers
"""

from titanframe._version import __version__

# ---- Phase 1: Core types ----
from titanframe.core.dtypes import (
    DType,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Bool,
    Utf8,
    Binary,
    Date,
    Datetime,
    Duration,
    Null,
    promote,
    can_cast,
    from_arrow as dtype_from_arrow,
)
from titanframe.core.schema import Schema
from titanframe.core.chunk import Chunk
from titanframe.core.table import Table

# ---- Phase 2: Expression DSL ----
from titanframe.expr.column_expr import col
from titanframe.expr.literal_expr import lit

# ---- Phase 2: User-facing API ----
from titanframe.api.dataframe import DataFrame
from titanframe.api.lazyframe import LazyFrame
from titanframe.api.series import Series

# ---- Phase 2: Top-level functions ----
from titanframe.api.functions import (
    read_csv,
    scan_csv,
    read_ipc,
    scan_ipc,
    read_parquet,
    scan_parquet,
    from_dict,
    from_arrow,
    from_pandas,
    concat,
    merge,
)


# ---- Phase 2: Config ----
from titanframe.api.config import TitanFrameConfig, config

# ---- Phase 11: Telemetry Dashboard ----
from titanframe.telemetry.server import start_dashboard, stop_dashboard

__all__ = [
    "__version__",
    # Types
    "DType",
    "Int8", "Int16", "Int32", "Int64",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float32", "Float64",
    "Bool", "Utf8", "Binary",
    "Date", "Datetime", "Duration", "Null",
    "promote", "can_cast", "dtype_from_arrow",
    # Core
    "Schema", "Chunk", "Table",
    # Expressions
    "col", "lit",
    # API
    "DataFrame", "LazyFrame", "Series",
    # I/O
    "read_csv", "scan_csv", "read_ipc", "scan_ipc", "read_parquet",
    # Construction
    "from_dict", "from_arrow", "from_pandas",
    # Combining
    "concat", "merge",
    # Config
    "TitanFrameConfig", "config",
    # Telemetry
    "start_dashboard", "stop_dashboard"
]
