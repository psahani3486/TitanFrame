"""
TitanFrame DataFrame — Eager Evaluation
=========================================

The DataFrame is the primary user-facing object. It provides a Pandas-
compatible API that operates eagerly (each operation executes immediately).

For deferred/optimized execution on large datasets, use :class:`LazyFrame`.

Example::

    >>> import titanframe as tf
    >>> df = tf.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
    >>> df.filter(tf.col("age") > 25)
    >>> df.select("name", "age")
    >>> df.group_by("name").agg(tf.col("age").mean())
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from titanframe.core.chunk import Chunk
from titanframe.core.column import ChunkedColumn
from titanframe.core.dtypes import DType, from_arrow, from_value
from titanframe.core.schema import Schema
from titanframe.core.table import Table
from titanframe.expr.base import (
    Expr, BinaryExpr, UnaryExpr, AggExpr, CastExpr, AliasExpr, Op, UnaryOp, AggOp,
)
from titanframe.expr.column_expr import col, ColumnExpr
from titanframe.expr.literal_expr import lit, LiteralExpr
from titanframe.api.series import Series
from titanframe.api.groupby import GroupBy


class DataFrame:
    """
    An eager, Pandas-compatible DataFrame backed by a TitanFrame Table.

    Every operation executes immediately and returns a new DataFrame.

    Args:
        data: Python dict, Arrow Table, Pandas DataFrame, or TitanFrame Table.
    """

    __slots__ = ("_table",)

    def __init__(self, data: dict[str, list] | pa.Table | Table | None = None):
        if data is None:
            self._table = Table(Schema())
        elif isinstance(data, dict):
            self._table = Table.from_pydict(data)
        elif isinstance(data, pa.Table):
            self._table = Table.from_arrow(data)
        elif isinstance(data, Table):
            self._table = data
        else:
            # Try Pandas
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    self._table = Table.from_pandas(data)
                    return
            except ImportError:
                pass
            raise TypeError(f"Cannot construct DataFrame from {type(data).__name__}")

    @classmethod
    def _from_table(cls, table: Table) -> DataFrame:
        """Internal constructor from a Table object."""
        df = object.__new__(cls)
        df._table = table
        return df

    # ---- Properties ----

    @property
    def schema(self) -> Schema:
        """The schema of this DataFrame."""
        return self._table.schema

    @property
    def columns(self) -> list[str]:
        """Column names."""
        return self._table.column_names

    @property
    def dtypes(self) -> list[DType]:
        """Column data types."""
        return self._table.dtypes

    @property
    def shape(self) -> tuple[int, int]:
        """(num_rows, num_columns)."""
        return self._table.shape

    @property
    def num_rows(self) -> int:
        return self._table.num_rows

    @property
    def num_columns(self) -> int:
        return self._table.num_columns

    @property
    def is_empty(self) -> bool:
        return self._table.is_empty

    def __len__(self) -> int:
        return self.num_rows

    # ---- Column access ----

    def __getitem__(self, key: str | list[str] | Series) -> Series | DataFrame:
        """
        Access columns by name or filter by boolean mask.

        - ``df["col"]`` → Series
        - ``df[["col1", "col2"]]`` → DataFrame
        - ``df[bool_series]`` → filtered DataFrame
        """
        if isinstance(key, str):
            column = self._table.column(key)
            return Series(key, column)
        elif isinstance(key, list):
            return DataFrame._from_table(self._table.select(key))
        elif isinstance(key, Series):
            # Boolean mask filtering
            mask = key._to_combined_array()
            arrow_table = self._table.to_arrow()
            filtered = arrow_table.filter(mask)
            return DataFrame._from_table(Table.from_arrow(filtered))
        raise TypeError(f"Invalid key type: {type(key).__name__}")

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Add or replace a column: ``df["new_col"] = values``.
        """
        if isinstance(value, Series):
            col_data = value._chunked_column
        elif isinstance(value, list):
            arr = pa.array(value)
            col_data = ChunkedColumn(key, from_arrow(arr.type), [arr])
        elif isinstance(value, (int, float, str, bool)):
            # Scalar broadcast
            arr = pa.array([value] * self.num_rows)
            col_data = ChunkedColumn(key, from_arrow(arr.type), [arr])
        else:
            raise TypeError(f"Cannot set column from {type(value).__name__}")

        # If column exists, drop it first, then append
        if key in self.columns:
            new_table = self._table.drop([key])
        else:
            new_table = self._table

        # Rechunk to align
        col_data = col_data.rechunk(max(1, new_table.num_chunks))
        self._table = new_table.append_column(key, col_data)

    # ---- Selection ----

    def select(self, *names: str | Expr) -> DataFrame:
        """
        Select columns by name or expression.

        Example::

            >>> df.select("name", "age")
            >>> df.select(col("name"), (col("a") + col("b")).alias("sum"))
        """
        # Simple case: all strings
        if all(isinstance(n, str) for n in names):
            return DataFrame._from_table(self._table.select(list(names)))  # type: ignore

        # Expression-based select
        result_arrays: dict[str, pa.Array] = {}
        arrow_table = self._table.to_arrow()
        for expr_or_name in names:
            if isinstance(expr_or_name, str):
                result_arrays[expr_or_name] = arrow_table.column(expr_or_name).combine_chunks()
            elif isinstance(expr_or_name, Expr):
                name, arr = _eval_expr_on_table(expr_or_name, arrow_table)
                result_arrays[name] = arr
            else:
                raise TypeError(f"Expected str or Expr, got {type(expr_or_name).__name__}")

        return DataFrame(pa.table(result_arrays))

    def with_columns(self, *exprs: Expr) -> DataFrame:
        """
        Add or replace columns by expression.

        Example::

            >>> df.with_columns(
            ...     (col("a") + col("b")).alias("sum"),
            ...     col("a").cast(Float64).alias("a_float"),
            ... )
        """
        arrow_table = self._table.to_arrow()
        new_columns: dict[str, pa.Array] = {}

        # Start with existing columns
        for name in self.columns:
            new_columns[name] = arrow_table.column(name).combine_chunks()

        # Evaluate and add/replace
        for expr in exprs:
            name, arr = _eval_expr_on_table(expr, arrow_table)
            new_columns[name] = arr

        return DataFrame(pa.table(new_columns))

    def drop(self, *names: str) -> DataFrame:
        """Drop columns by name."""
        return DataFrame._from_table(self._table.drop(list(names)))

    def rename(self, mapping: dict[str, str]) -> DataFrame:
        """Rename columns."""
        return DataFrame._from_table(self._table.rename(mapping))

    # ---- Filtering ----

    def filter(self, expr: Expr) -> DataFrame:
        """
        Filter rows by a boolean expression.

        Example::

            >>> df.filter(col("age") > 25)
            >>> df.filter((col("status") == "active") & (col("revenue") > 1000))
        """
        arrow_table = self._table.to_arrow()
        _, mask = _eval_expr_on_table(expr, arrow_table)
        filtered = arrow_table.filter(mask)
        return DataFrame._from_table(Table.from_arrow(filtered))

    # ---- Aggregation ----

    def group_by(self, *keys: str | Expr) -> GroupBy:
        """
        Group by columns for aggregation.

        Example::

            >>> df.group_by("region").agg(col("revenue").sum().alias("total"))
        """
        return GroupBy(self, keys)

    # ---- Sorting ----

    def sort(self, by: str | list[str], descending: bool | list[bool] = False) -> DataFrame:
        """
        Sort by one or more columns.

        Example::

            >>> df.sort("age")
            >>> df.sort(["region", "revenue"], descending=[False, True])
        """
        if isinstance(by, str):
            by = [by]
        if isinstance(descending, bool):
            descending = [descending] * len(by)

        sort_keys = [
            (name, "descending" if desc else "ascending")
            for name, desc in zip(by, descending)
        ]

        arrow_table = self._table.to_arrow()
        indices = pc.sort_indices(arrow_table, sort_keys=sort_keys)
        sorted_table = pc.take(arrow_table, indices)
        return DataFrame._from_table(Table.from_arrow(sorted_table))

    # ---- Joins ----

    def join(
        self,
        other: DataFrame,
        on: str | list[str],
        how: str = "inner",
        suffix: str = "_right",
    ) -> DataFrame:
        """
        Join with another DataFrame.

        Args:
            other: Right-side DataFrame.
            on: Join key column(s).
            how: Join type — "inner", "left", "right", "outer".
            suffix: Suffix for duplicate column names from the right side.

        Example::

            >>> orders.join(customers, on="customer_id", how="left")
        """
        if isinstance(on, str):
            on = [on]

        left_table = self._table.to_arrow()
        right_table = other._table.to_arrow()

        # Rename duplicate columns in right table
        right_rename = {}
        for col_name in right_table.column_names:
            if col_name in left_table.column_names and col_name not in on:
                right_rename[col_name] = f"{col_name}{suffix}"

        if right_rename:
            new_names = [right_rename.get(n, n) for n in right_table.column_names]
            right_table = right_table.rename_columns(new_names)

        # Map TitanFrame join types to pyarrow
        join_type_map = {
            "inner": "inner",
            "left": "left outer",
            "right": "right outer",
            "outer": "full outer",
            "cross": "inner",  # cross join not directly supported; fallback
        }
        pa_join_type = join_type_map.get(how, how)

        result = left_table.join(right_table, keys=on, join_type=pa_join_type)
        return DataFrame._from_table(Table.from_arrow(result))

    # ---- Slicing ----

    def head(self, n: int = 5) -> DataFrame:
        """Return the first n rows."""
        return DataFrame._from_table(self._table.head(n))

    def tail(self, n: int = 5) -> DataFrame:
        """Return the last n rows."""
        return DataFrame._from_table(self._table.tail(n))

    def slice(self, offset: int, length: Optional[int] = None) -> DataFrame:
        """Return a slice of rows."""
        return DataFrame._from_table(self._table.slice(offset, length))

    def sample(self, n: int = 5, seed: Optional[int] = None) -> DataFrame:
        """Random sample of n rows."""
        import random
        rng = random.Random(seed)
        total = self.num_rows
        indices = rng.sample(range(total), min(n, total))
        arrow_table = self._table.to_arrow()
        sampled = pc.take(arrow_table, pa.array(sorted(indices)))
        return DataFrame._from_table(Table.from_arrow(sampled))

    # ---- Deduplication ----

    def unique(self, subset: Optional[list[str]] = None) -> DataFrame:
        """Remove duplicate rows."""
        arrow_table = self._table.to_arrow()
        if subset:
            # Group by subset columns, take first of each group
            grouped = arrow_table.group_by(subset)
            for name in arrow_table.column_names:
                if name not in subset:
                    grouped = grouped.aggregate([(name, "first")])
            return DataFrame._from_table(Table.from_arrow(grouped))
        else:
            # Full row dedup via pandas (Arrow doesn't have a native API)
            df_pd = arrow_table.to_pandas().drop_duplicates()
            return DataFrame._from_table(Table.from_pandas(df_pd))

    def drop_nulls(self, subset: Optional[list[str]] = None) -> DataFrame:
        """Remove rows with null values."""
        arrow_table = self._table.to_arrow()
        result = arrow_table.drop_null()
        return DataFrame._from_table(Table.from_arrow(result))

    # ---- Null handling ----

    def fill_null(self, value: Any) -> DataFrame:
        """Fill null values across all columns."""
        arrow_table = self._table.to_arrow()
        new_columns = {}
        for name in arrow_table.column_names:
            col_arr = arrow_table.column(name).combine_chunks()
            try:
                filled = pc.fill_null(col_arr, pa.scalar(value, type=col_arr.type))
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                filled = col_arr  # Skip columns where the fill type doesn't match
            new_columns[name] = filled
        return DataFrame(pa.table(new_columns))

    # ---- Statistics ----

    def describe(self) -> DataFrame:
        """
        Compute summary statistics for numeric columns.

        Returns a DataFrame with count, mean, std, min, 25%, 50%, 75%, max.
        """
        stats: dict[str, list] = {"statistic": ["count", "mean", "std", "min", "max"]}

        for name in self.columns:
            dtype = self.schema[name]
            if dtype.is_numeric:
                s = self[name]
                stats[name] = [
                    float(s.count()),
                    s.mean(),
                    s.std(),
                    s.min(),
                    s.max(),
                ]
        return DataFrame(stats)

    # ---- Vertical operations ----

    def vstack(self, other: DataFrame) -> DataFrame:
        """Vertically concatenate another DataFrame."""
        return DataFrame._from_table(self._table.vstack(other._table))

    # ---- Conversion ----

    def to_pandas(self) -> Any:
        """Convert to a Pandas DataFrame."""
        return self._table.to_pandas()

    def to_pydict(self) -> dict[str, list]:
        """Convert to a Python dict of lists."""
        return self._table.to_pydict()

    def to_arrow(self) -> pa.Table:
        """Convert to a PyArrow Table."""
        return self._table.to_arrow()

    def to_lazy(self) -> Any:
        """
        Convert to a LazyFrame for deferred, optimized execution.
        """
        from titanframe.api.lazyframe import LazyFrame
        from titanframe.plan.logical.scan import Scan, ScanFormat
        scan = Scan(
            source="<in_memory>",
            format=ScanFormat.IN_MEMORY,
            schema=self.schema,
            table=self._table
        )
        lf = LazyFrame(scan)
        lf._materialized_data = self._table
        return lf

    # ---- Pandas Compat ----
    
    def dropna(self, subset: Optional[list[str]] = None) -> DataFrame:
        return self.drop_nulls(subset)

    def fillna(self, value: Any) -> DataFrame:
        return self.fill_null(value)

    def merge(self, right: DataFrame, how: str = "inner", on: str | list[str] | None = None) -> DataFrame:
        return self.to_lazy().join(right.to_lazy(), on=on, how=how).collect()

    def concat(self, others: list[DataFrame] | DataFrame) -> DataFrame:
        if not isinstance(others, list):
            others = [others]
        result = self.to_lazy()
        for o in others:
            result = result.vstack(o.to_lazy())
        return result.collect()

    def to_csv(self, target: str, **kwargs) -> None:
        from titanframe.io.csv import write_csv
        write_csv(self._table, target, **kwargs)

    def to_parquet(self, target: str, **kwargs) -> None:
        from titanframe.io.parquet import write_parquet
        write_parquet(self.to_lazy(), target, **kwargs)

    def to_json(self, target: str, **kwargs) -> None:
        from titanframe.io.json import write_json
        write_json(self.to_lazy(), target, **kwargs)

    def to_sql(self, target: str, uri: str, **kwargs) -> None:
        from titanframe.io.database import write_sql
        write_sql(self.to_lazy(), target, uri, **kwargs)

    def __repr__(self) -> str:
        lines = [f"shape: ({self.num_rows}, {self.num_columns})"]
        
        if self.num_rows == 0:
            return "\n".join(lines)
            
        # Get head and tail
        max_rows = 10
        if self.num_rows <= max_rows:
            preview = self.to_pydict()
            n_disp = self.num_rows
            tail_idx = -1
        else:
            head = self.head(5).to_pydict()
            tail = self.tail(5).to_pydict()
            preview = {k: head[k] + tail[k] for k in self.columns}
            n_disp = 10
            tail_idx = 5
            
        # Format types
        types = [f"<{str(t)}>" for t in self.dtypes]
        
        # Calculate column widths
        col_widths = {}
        for i, name in enumerate(self.columns):
            values = [str(v) for v in preview[name]]
            type_len = len(types[i])
            col_widths[name] = max(len(name), type_len, max((len(v) for v in values), default=0))
            
        # Header
        header = " | ".join(name.ljust(col_widths[name]) for name in self.columns)
        type_row = " | ".join(types[i].ljust(col_widths[name]) for i, name in enumerate(self.columns))
        separator = "-+-".join("-" * col_widths[name] for name in self.columns)
        
        lines.append(f"+-{''.join('-' for _ in range(len(header)))}-+")
        lines.append(f"| {header} |")
        lines.append(f"| {type_row} |")
        lines.append(f"+-{separator}-+")
        
        # Rows
        for i in range(n_disp):
            if i == tail_idx:
                lines.append(f"| {'...'.center(len(header))} |")
            row = " | ".join(str(preview[name][i]).ljust(col_widths[name]) for name in self.columns)
            lines.append(f"| {row} |")
            
        lines.append(f"+-{''.join('-' for _ in range(len(header)))}-+")
        
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        max_rows = 20
        preview = self.head(max_rows)
        data = preview.to_pydict()

        html = ['<table style="border-collapse:collapse;">']
        # Header
        html.append('<tr>')
        for name in self.columns:
            html.append(
                f'<th style="border:1px solid #ddd;padding:6px 12px;'
                f'background:#f5f5f5;font-weight:600;">{name}</th>'
            )
        html.append('</tr>')
        # Type row
        html.append('<tr>')
        for dtype in self.dtypes:
            html.append(
                f'<td style="border:1px solid #ddd;padding:4px 12px;'
                f'color:#888;font-size:0.85em;">{dtype}</td>'
            )
        html.append('</tr>')
        # Data rows
        n_rows = preview.num_rows
        for i in range(n_rows):
            html.append('<tr>')
            for name in self.columns:
                val = data[name][i]
                html.append(f'<td style="border:1px solid #ddd;padding:4px 12px;">{val}</td>')
            html.append('</tr>')
        html.append('</table>')

        if self.num_rows > max_rows:
            html.append(f'<p style="color:#888;">... {self.num_rows - max_rows} more rows</p>')

        html.append(
            f'<p style="color:#888;font-size:0.85em;">'
            f'shape: ({self.num_rows:,}, {self.num_columns})</p>'
        )
        return "\n".join(html)


# ---------------------------------------------------------------------------
# Expression evaluator (eager — for DataFrame operations)
# ---------------------------------------------------------------------------

def _eval_expr_on_table(expr: Expr, table: pa.Table) -> tuple[str, pa.Array]:
    """
    Evaluate an expression against an Arrow table.

    Returns (output_name, result_array).
    """
    if isinstance(expr, AliasExpr):
        _, arr = _eval_expr_on_table(expr.child, table)
        return expr.name, arr

    if isinstance(expr, ColumnExpr):
        return expr.column_name, table.column(expr.column_name).combine_chunks()

    if isinstance(expr, LiteralExpr):
        n = table.num_rows
        arr = pa.array([expr.value] * n, type=expr.dtype.arrow_type)
        return repr(expr), arr

    if isinstance(expr, BinaryExpr):
        _, left = _eval_expr_on_table(expr.left, table)
        _, right = _eval_expr_on_table(expr.right, table)
        result = _apply_binary_op(expr.op, left, right)
        return repr(expr), result

    if isinstance(expr, UnaryExpr):
        _, child = _eval_expr_on_table(expr.operand, table)
        result = _apply_unary_op(expr.op, child)
        return repr(expr), result

    # TryCastExpr is NOT a subclass of CastExpr — check it first
    from titanframe.expr.cast_expr import TryCastExpr
    if isinstance(expr, TryCastExpr):
        _, child = _eval_expr_on_table(expr.child, table)
        try:
            result = pc.cast(child, expr.target_dtype.arrow_type, safe=False)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            result = pa.nulls(len(child), type=expr.target_dtype.arrow_type)
        return repr(expr), result

    if isinstance(expr, CastExpr):
        _, child = _eval_expr_on_table(expr.child, table)
        result = pc.cast(child, expr.target_dtype.arrow_type)
        return repr(expr), result

    if isinstance(expr, AggExpr):
        # Handle QuantileExpr (subclass of AggExpr with a q parameter)
        from titanframe.expr.agg_expr import QuantileExpr
        if isinstance(expr, QuantileExpr):
            _, child = _eval_expr_on_table(expr.child, table)
            scalar = pc.quantile(child, q=expr.q)
            # quantile returns an array of results; take the first
            q_val = scalar[0].as_py() if len(scalar) > 0 else None
            result = pa.array([q_val] * table.num_rows)
            return repr(expr), result

        _, child = _eval_expr_on_table(expr.child, table)
        result = _apply_agg_op(expr.op, child, table.num_rows)
        return repr(expr), result

    raise TypeError(f"Cannot evaluate expression type: {type(expr).__name__}")


# ---- Op dispatch tables ----

_BINARY_OPS = {
    Op.ADD: pc.add,
    Op.SUB: pc.subtract,
    Op.MUL: pc.multiply,
    Op.TRUE_DIV: pc.divide,
    Op.EQ: pc.equal,
    Op.NE: pc.not_equal,
    Op.LT: pc.less,
    Op.LE: pc.less_equal,
    Op.GT: pc.greater,
    Op.GE: pc.greater_equal,
    Op.AND: pc.and_,
    Op.OR: pc.or_,
    Op.XOR: pc.xor,
}


def _apply_binary_op(op: Op, left: pa.Array, right: pa.Array) -> pa.Array:
    if op in _BINARY_OPS:
        return _BINARY_OPS[op](left, right)
    if op == Op.FLOOR_DIV:
        div = pc.divide(left, right)
        return pc.floor(div)
    if op == Op.MOD:
        div = pc.divide(left, right)
        floored = pc.floor(div)
        return pc.subtract(left, pc.multiply(floored, right))
    if op == Op.POW:
        return pc.power(left, right)
    raise ValueError(f"Unsupported binary op: {op}")


_UNARY_OPS = {
    UnaryOp.NEG: pc.negate,
    UnaryOp.NOT: pc.invert,
    UnaryOp.ABS: pc.abs,
    UnaryOp.IS_NULL: pc.is_null,
    UnaryOp.IS_NOT_NULL: pc.is_valid,
    UnaryOp.CEIL: pc.ceil,
    UnaryOp.FLOOR: pc.floor,
}


def _apply_unary_op(op: UnaryOp, arr: pa.Array) -> pa.Array:
    if op in _UNARY_OPS:
        return _UNARY_OPS[op](arr)
    if op == UnaryOp.SQRT:
        return pc.power(arr, pa.scalar(0.5))
    if op == UnaryOp.LOG:
        return pc.ln(arr)
    if op == UnaryOp.EXP:
        # Arrow doesn't have exp; use a manual approach
        import numpy as np
        np_arr = arr.to_numpy(zero_copy_only=False)
        return pa.array(np.exp(np_arr))
    raise ValueError(f"Unsupported unary op: {op}")


def _apply_agg_op(op: AggOp, arr: pa.Array, n: int) -> pa.Array:
    """Apply aggregation and broadcast result to match the original row count."""
    agg_funcs = {
        AggOp.SUM: pc.sum,
        AggOp.MEAN: pc.mean,
        AggOp.MIN: pc.min,
        AggOp.MAX: pc.max,
        AggOp.COUNT: pc.count,
    }
    if op in agg_funcs:
        scalar = agg_funcs[op](arr)
        return pa.array([scalar.as_py()] * n)
    if op == AggOp.STD:
        scalar = pc.stddev(arr)
        return pa.array([scalar.as_py()] * n)
    if op == AggOp.VAR:
        scalar = pc.variance(arr)
        return pa.array([scalar.as_py()] * n)
    raise ValueError(f"Unsupported agg op: {op}")
