"""
Column Expression — ``col("name")``
====================================

Creates a reference to a named column in the data. This is the most common
leaf node in expression trees.

Example::

    >>> from titanframe.expr.column_expr import col
    >>> expr = col("revenue") > 1000
    >>> expr.required_columns()
    {'revenue'}
"""

from __future__ import annotations

from titanframe.expr.base import Expr


class ColumnExpr(Expr):
    """
    A reference to a named column.

    Attributes:
        column_name: The name of the column being referenced.
    """

    __slots__ = ("column_name",)

    def __init__(self, column_name: str):
        self.column_name = column_name

    def children(self) -> list[Expr]:
        return []

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return ColumnExpr(self.column_name)

    def _collect_columns(self, out: set[str]) -> None:
        out.add(self.column_name)

    def display(self, indent: int = 0) -> str:
        return f"col({self.column_name!r})"

    def __hash__(self) -> int:
        return hash(("column", self.column_name))


def col(name: str) -> ColumnExpr:
    """
    Create a column reference expression.

    This is the primary entry point for building expressions::

        >>> from titanframe import col
        >>> expr = col("price") * col("quantity")

    Args:
        name: Column name to reference.

    Returns:
        A :class:`ColumnExpr` node.
    """
    return ColumnExpr(name)
