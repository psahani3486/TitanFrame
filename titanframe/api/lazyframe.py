"""
TitanFrame LazyFrame — Deferred Execution
============================================

A LazyFrame captures a computation as a logical plan (DAG) without
executing it. When ``.collect()`` is called, the plan is optimized
and then executed.

This is where TitanFrame's power lies: the optimizer can rewrite the
plan to push down predicates, prune columns, fuse operators, and
partition work across GPUs — all transparently to the user.

Example::

    >>> import titanframe as tf
    >>> lf = tf.scan_csv("huge_dataset.csv")
    >>> result = (
    ...     lf.filter(tf.col("revenue") > 1000)
    ...       .select("region", "revenue")
    ...       .group_by("region")
    ...       .agg(tf.col("revenue").sum().alias("total"))
    ...       .sort("total", descending=True)
    ...       .collect()
    ... )
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from titanframe.core.schema import Schema
from titanframe.core.table import Table
from titanframe.expr.base import Expr, SortExpr, SortOrder
from titanframe.expr.column_expr import col, ColumnExpr
from titanframe.plan.logical.node import LogicalPlan
from titanframe.plan.logical.scan import Scan, ScanFormat
from titanframe.plan.logical.projection import Projection
from titanframe.plan.logical.filter import Filter
from titanframe.plan.logical.aggregation import Aggregation
from titanframe.plan.logical.join import Join
from titanframe.plan.logical.sort import Sort
from titanframe.plan.logical.limit import Limit
from titanframe.plan.logical.distinct import Distinct
from titanframe.plan.logical.union import Union


class LazyFrame:
    """
    A deferred computation represented as a logical plan.

    No data is touched until ``.collect()`` is called. This allows
    the query optimizer to analyze the full computation graph and
    produce an efficient execution plan.

    Attributes:
        _plan: The root of the logical plan tree.
        _materialized_data: Optional pre-materialized data (for in-memory sources).
    """

    __slots__ = ("_plan", "_materialized_data")

    def __init__(self, plan: LogicalPlan):
        self._plan = plan
        self._materialized_data: Optional[Table] = None


    @property
    def schema(self) -> Schema:
        """The output schema of this LazyFrame (computed from the plan)."""
        return self._plan.output_schema()

    @property
    def columns(self) -> list[str]:
        """Column names in the output."""
        return self.schema.names

    @property
    def plan(self) -> LogicalPlan:
        """The underlying logical plan (for inspection/debugging)."""
        return self._plan


    def select(self, *names_or_exprs: str | Expr) -> LazyFrame:
        """
        Select columns by name or expression.

        Example::

            >>> lf.select("name", "age")
            >>> lf.select(col("name"), (col("a") + col("b")).alias("sum"))
        """
        exprs: list[Expr] = []
        for item in names_or_exprs:
            if isinstance(item, str):
                exprs.append(col(item))
            elif isinstance(item, Expr):
                exprs.append(item)
            else:
                raise TypeError(f"Expected str or Expr, got {type(item).__name__}")

        new_plan = Projection(self._plan, exprs)
        return LazyFrame(new_plan)

    def with_columns(self, *exprs: Expr) -> LazyFrame:
        """
        Add or replace columns by expression.

        Existing columns are preserved; new expressions are appended.
        """
        existing = [col(name) for name in self.columns]
        all_exprs = existing + list(exprs)
        new_plan = Projection(self._plan, all_exprs)
        return LazyFrame(new_plan)

    def drop(self, *names: str) -> LazyFrame:
        """Drop columns by name."""
        keep = [col(n) for n in self.columns if n not in set(names)]
        new_plan = Projection(self._plan, keep)
        return LazyFrame(new_plan)

    def rename(self, mapping: dict[str, str]) -> LazyFrame:
        """Rename columns."""
        exprs = [
            col(name).alias(mapping.get(name, name)) for name in self.columns
        ]
        new_plan = Projection(self._plan, exprs)
        return LazyFrame(new_plan)


    def filter(self, expr: Expr) -> LazyFrame:
        """
        Filter rows by a boolean expression.

        Example::

            >>> lf.filter(col("age") > 25)
            >>> lf.filter((col("status") == "active") & (col("revenue") > 1000))
        """
        new_plan = Filter(self._plan, expr)
        return LazyFrame(new_plan)


    def group_by(self, *keys: str | Expr) -> Any:
        """
        Group by columns for aggregation.

        Returns a :class:`LazyGroupBy` proxy.
        """
        from titanframe.api.groupby import LazyGroupBy
        return LazyGroupBy(self, keys)


    def sort(
        self,
        by: str | list[str] | Expr | list[Expr],
        descending: bool | list[bool] = False,
    ) -> LazyFrame:
        """
        Sort by one or more columns.

        Example::

            >>> lf.sort("age")
            >>> lf.sort(["region", "revenue"], descending=[False, True])
        """
        if isinstance(by, (str, Expr)):
            by = [by]
        if isinstance(descending, bool):
            descending = [descending] * len(by)

        sort_exprs: list[Expr] = []
        for b, d in zip(by, descending):
            if isinstance(b, str):
                expr = col(b)
            else:
                expr = b
            sort_exprs.append(expr.desc() if d else expr.asc())

        new_plan = Sort(self._plan, sort_exprs)
        return LazyFrame(new_plan)


    def join(
        self,
        other: LazyFrame,
        on: str | list[str],
        how: str = "inner",
        suffix: str = "_right",
    ) -> LazyFrame:
        """
        Join with another LazyFrame.

        Example::

            >>> orders_lf.join(customers_lf, on="customer_id", how="left")
        """
        if isinstance(on, str):
            on = [on]
        new_plan = Join(self._plan, other._plan, on, how, suffix)
        return LazyFrame(new_plan)


    def head(self, n: int = 5) -> LazyFrame:
        """Take the first n rows."""
        new_plan = Limit(self._plan, n)
        return LazyFrame(new_plan)

    def tail(self, n: int = 5) -> LazyFrame:
        """Take the last n rows."""
        new_plan = Limit(self._plan, n, offset=-1)
        return LazyFrame(new_plan)

    def limit(self, n: int) -> LazyFrame:
        """Limit to n rows."""
        return self.head(n)

    def slice(self, offset: int, length: int) -> LazyFrame:
        """Take a slice of rows."""
        new_plan = Limit(self._plan, length, offset=offset)
        return LazyFrame(new_plan)


    def unique(self, subset: Optional[list[str]] = None) -> LazyFrame:
        """Remove duplicate rows."""
        new_plan = Distinct(self._plan, subset)
        return LazyFrame(new_plan)

    def distinct(self) -> LazyFrame:
        """Remove duplicate rows (alias for unique)."""
        return self.unique()


    def vstack(self, other: LazyFrame) -> LazyFrame:
        """Vertically concatenate another LazyFrame."""
        new_plan = Union([self._plan, other._plan])
        return LazyFrame(new_plan)


    def collect(self) -> Any:
        """
        Execute the lazy computation and return an eager DataFrame.
        """
        from titanframe.api.dataframe import DataFrame
        from titanframe.plan.optimizer.driver import QueryOptimizer
        from titanframe.plan.optimizer.predicate_pushdown import PredicatePushdown
        from titanframe.plan.optimizer.projection_pushdown import ProjectionPushdown
        from titanframe.plan.optimizer.constant_folding import ConstantFolding
        from titanframe.plan.physical.planner import PhysicalPlanner
        from titanframe.engine.scheduler import DAGScheduler
        from titanframe.plan.physical.node import ExecutionContext
        
        rules = [PredicatePushdown(), ProjectionPushdown(), ConstantFolding()]
        optimizer = QueryOptimizer(rules)
        optimized_plan = optimizer.optimize(self._plan)
        
        planner = PhysicalPlanner()
        phys_plan = planner.plan(optimized_plan)
        
        ctx = ExecutionContext()
        scheduler = DAGScheduler()
        arrow_table = scheduler.execute(phys_plan, ctx)
        
        return DataFrame(arrow_table)
        
    def _get_physical_plan(self):
        """Internal helper for streaming I/O."""
        from titanframe.plan.optimizer.driver import QueryOptimizer
        from titanframe.plan.optimizer.predicate_pushdown import PredicatePushdown
        from titanframe.plan.optimizer.projection_pushdown import ProjectionPushdown
        from titanframe.plan.optimizer.constant_folding import ConstantFolding
        from titanframe.plan.physical.planner import PhysicalPlanner
        
        rules = [PredicatePushdown(), ProjectionPushdown(), ConstantFolding()]
        optimizer = QueryOptimizer(rules)
        optimized_plan = optimizer.optimize(self._plan)
        
        planner = PhysicalPlanner()
        return planner.plan(optimized_plan)

    def _execute(self, plan: LogicalPlan) -> Table:
        """
        Execute a logical plan directly (interpreter mode).

        This is a simple recursive evaluator used before the full physical
        planner is built. It walks the plan tree bottom-up and executes
        each node using pyarrow.compute.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        if isinstance(plan, Scan):
            return self._execute_scan(plan)

        if isinstance(plan, Projection):
            child_table = self._execute(plan.input)
            return self._execute_projection(plan, child_table)

        if isinstance(plan, Filter):
            child_table = self._execute(plan.input)
            return self._execute_filter(plan, child_table)

        if isinstance(plan, Sort):
            child_table = self._execute(plan.input)
            return self._execute_sort(plan, child_table)

        if isinstance(plan, Limit):
            child_table = self._execute(plan.input)
            return self._execute_limit(plan, child_table)

        if isinstance(plan, Distinct):
            child_table = self._execute(plan.input)
            return self._execute_distinct(plan, child_table)

        if isinstance(plan, Aggregation):
            child_table = self._execute(plan.input)
            return self._execute_aggregation(plan, child_table)

        if isinstance(plan, Join):
            left_table = self._execute(plan.left)
            right_table = self._execute(plan.right)
            return self._execute_join(plan, left_table, right_table)

        if isinstance(plan, Union):
            tables = [self._execute(child) for child in plan.inputs]
            result = tables[0]
            for t in tables[1:]:
                result = result.vstack(t)
            return result

        raise TypeError(f"Cannot execute plan node: {type(plan).__name__}")

    def _execute_scan(self, plan: Scan) -> Table:
        """Execute a Scan node."""
        if plan.format == ScanFormat.IN_MEMORY and self._materialized_data is not None:
            table = self._materialized_data
        elif plan.format == ScanFormat.CSV:
            from titanframe.io.csv import read_csv_to_table
            table = read_csv_to_table(plan.source, chunk_size=plan.chunk_size)
        elif plan.format == ScanFormat.ARROW_IPC:
            table = Table.from_ipc_file(plan.source)
        else:
            raise ValueError(f"Unsupported scan format: {plan.format}")

        if plan.projection:
            table = table.select(plan.projection)

        if plan.limit is not None:
            table = table.head(plan.limit)

        return table

    def _execute_projection(self, plan: Projection, child: Table) -> Table:
        """Execute a Projection node."""
        from titanframe.api.dataframe import _eval_expr_on_table

        arrow_table = child.to_arrow()
        result_columns: dict[str, Any] = {}

        for expr in plan.exprs:
            name, arr = _eval_expr_on_table(expr, arrow_table)
            result_columns[name] = arr

        import pyarrow as pa
        return Table.from_arrow(pa.table(result_columns))

    def _execute_filter(self, plan: Filter, child: Table) -> Table:
        """Execute a Filter node."""
        from titanframe.api.dataframe import _eval_expr_on_table
        import pyarrow as pa

        arrow_table = child.to_arrow()
        _, mask = _eval_expr_on_table(plan.predicate, arrow_table)
        filtered = arrow_table.filter(mask)
        return Table.from_arrow(filtered)

    def _execute_sort(self, plan: Sort, child: Table) -> Table:
        """Execute a Sort node."""
        import pyarrow.compute as pc

        arrow_table = child.to_arrow()
        sort_keys = []

        for expr in plan.sort_exprs:
            if isinstance(expr, SortExpr):
                col_name = _extract_sort_col(expr.child)
                order = "descending" if expr.order == SortOrder.DESC else "ascending"
                sort_keys.append((col_name, order))
            elif isinstance(expr, ColumnExpr):
                sort_keys.append((expr.column_name, "ascending"))
            else:
                col_name = _extract_sort_col(expr)
                sort_keys.append((col_name, "ascending"))

        indices = pc.sort_indices(arrow_table, sort_keys=sort_keys)
        sorted_table = pc.take(arrow_table, indices)
        return Table.from_arrow(sorted_table)

    def _execute_limit(self, plan: Limit, child: Table) -> Table:
        """Execute a Limit node."""
        if plan.offset == -1:
            return child.tail(plan.n)
        return child.slice(plan.offset, plan.n)

    def _execute_distinct(self, plan: Distinct, child: Table) -> Table:
        """Execute a Distinct node."""
        df_pd = child.to_pandas()
        if plan.subset:
            df_pd = df_pd.drop_duplicates(subset=plan.subset)
        else:
            df_pd = df_pd.drop_duplicates()
        return Table.from_pandas(df_pd)

    def _execute_aggregation(self, plan: Aggregation, child: Table) -> Table:
        """Execute an Aggregation node."""
        from titanframe.api.dataframe import DataFrame
        from titanframe.api.groupby import GroupBy

        df = DataFrame._from_table(child)
        key_names = [
            k.column_name if isinstance(k, ColumnExpr) else repr(k)
            for k in plan.group_keys
        ]
        gb = df.group_by(*key_names)
        result_df = gb.agg(*plan.agg_exprs)
        return result_df._table

    def _execute_join(self, plan: Join, left: Table, right: Table) -> Table:
        """Execute a Join node."""
        from titanframe.api.dataframe import DataFrame

        left_df = DataFrame._from_table(left)
        right_df = DataFrame._from_table(right)
        result_df = left_df.join(right_df, on=plan.on, how=plan.how, suffix=plan.suffix)
        return result_df._table


    def explain(self, optimized: bool = True) -> str:
        """
        Print the logical plan (optionally after optimization).
        """
        if optimized:
            from titanframe.plan.optimizer.driver import QueryOptimizer
            plan = QueryOptimizer().optimize(self._plan)
        else:
            plan = self._plan
        text = plan.explain()
        print(text)
        return text

    def show_graph(self) -> str:
        """
        Return a Mermaid diagram string of the logical plan.

        Can be rendered in Jupyter or any Mermaid-compatible viewer.
        """
        lines = ["graph TD"]
        nodes = self._plan.walk()
        for i, node in enumerate(nodes):
            label = f'{node.node_name()}: {node.node_description()}'
            label = label.replace('"', "'")
            lines.append(f'    N{i}["{label}"]')
        for i, node in enumerate(nodes):
            for child in node.children():
                j = nodes.index(child)
                lines.append(f"    N{i} --> N{j}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LazyFrame(schema={self.schema})\n"
            f"Plan:\n{self._plan.display(indent=1)}"
        )


def _extract_sort_col(expr: Expr) -> str:
    """Extract column name from a sort expression."""
    if isinstance(expr, ColumnExpr):
        return expr.column_name
    if isinstance(expr, SortExpr):
        return _extract_sort_col(expr.child)
    raise ValueError(f"Cannot extract sort column from {expr}")
