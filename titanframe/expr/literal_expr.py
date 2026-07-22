"""
Literal Expression — ``lit(value)``
====================================

Creates a constant value in an expression tree. The type is automatically
inferred from the Python value.

Example::

    >>> from titanframe.expr.literal_expr import lit
    >>> lit(42)
    LiteralExpr(42, Int64)
    >>> lit(3.14)
    LiteralExpr(3.14, Float64)
    >>> lit("hello")
    LiteralExpr('hello', Utf8)
"""

from __future__ import annotations

from typing import Any

from titanframe.core.dtypes import DType, from_value
from titanframe.expr.base import Expr


class LiteralExpr(Expr):
    """
    A constant value embedded in an expression tree.

    Attributes:
        value: The Python value.
        dtype: The inferred TitanFrame type.
    """

    __slots__ = ("value", "dtype")

    def __init__(self, value: Any, dtype: DType | None = None):
        self.value = value
        self.dtype = dtype if dtype is not None else from_value(value)

    def children(self) -> list[Expr]:
        return []

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return LiteralExpr(self.value, self.dtype)

    def display(self, indent: int = 0) -> str:
        return f"lit({self.value!r})"

    def __hash__(self) -> int:
        return hash(("literal", self.value, self.dtype))


def lit(value: Any, dtype: DType | None = None) -> LiteralExpr:
    """
    Create a literal value expression.

    The type is inferred from the Python value unless ``dtype`` is provided::

        >>> from titanframe import lit
        >>> expr = col("price") > lit(100)

    Args:
        value: Python scalar value.
        dtype: Optional explicit type override.

    Returns:
        A :class:`LiteralExpr` node.
    """
    return LiteralExpr(value, dtype)
