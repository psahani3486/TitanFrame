"""
Projection Node — SELECT / WithColumns
========================================

Represents column selection or computed column expressions.
Equivalent to SQL's ``SELECT expr1 AS name1, expr2 AS name2, ...``.

Example::

    >>> plan = Projection(scan_node, [col("name"), (col("a") + col("b")).alias("sum")])
"""

from __future__ import annotations

from typing import Sequence

from titanframe.core.schema import Schema
from titanframe.expr.base import Expr
from titanframe.plan.logical.node import LogicalPlan
from titanframe.plan.logical.utils import infer_expr_name, infer_expr_dtype


class Projection(LogicalPlan):
    """
    Column selection and/or computed column expressions.

    Equivalent to SQL's ``SELECT expr1 AS name1, expr2 AS name2, ...``.

    Attributes:
        input: The child plan node.
        exprs: List of expressions to compute.
    """

    __slots__ = ("input", "exprs")

    def __init__(self, input: LogicalPlan, exprs: Sequence[Expr]):
        self.input = input
        self.exprs = list(exprs)

    def output_schema(self) -> Schema:
        input_schema = self.input.output_schema()
        fields = []
        for expr in self.exprs:
            name = infer_expr_name(expr)
            dtype = infer_expr_dtype(expr, input_schema)
            fields.append((name, dtype))
        return Schema(fields)

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_name(self) -> str:
        return "Projection"

    def node_description(self) -> str:
        expr_strs = [infer_expr_name(e) for e in self.exprs]
        return f"columns=[{', '.join(expr_strs)}]"

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Projection(new_children[0], self.exprs)

    def with_exprs(self, exprs: Sequence[Expr]) -> Projection:
        """Return a new Projection with different expressions."""
        return Projection(self.input, exprs)
