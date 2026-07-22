"""
Unary Expression — ``UnaryExpr``
==================================

Extended module for unary (single-operand) operations. Provides:
    - Arithmetic: ``Neg, Abs, Ceil, Floor, Sqrt, Log, Exp``
    - Logical: ``Not``
    - Null checks: ``IsNull, IsNotNull``

The core ``UnaryExpr`` class lives in :mod:`base` and is re-exported here.
This module adds additional math functions, type-inference helpers,
and convenience factory functions.

Example::

    >>> from titanframe.expr.unary_expr import neg, is_null, sqrt
    >>> from titanframe.expr.column_expr import col
    >>> expr = sqrt(col("variance"))
"""

from __future__ import annotations

from typing import Any

from titanframe.core.dtypes import DType, Bool, Float64
from titanframe.expr.base import (
    UnaryExpr,
    Expr,
    UnaryOp,
    _wrap,
)

__all__ = [
    "UnaryExpr",
    "UnaryOp",
    "neg",
    "not_",
    "is_null",
    "is_not_null",
    "abs_",
    "ceil",
    "floor",
    "sqrt",
    "log",
    "exp",
    "infer_unary_dtype",
    "is_null_check_op",
    "is_math_op",
]



_NULL_CHECK_OPS = frozenset({
    UnaryOp.IS_NULL, UnaryOp.IS_NOT_NULL,
})

_MATH_OPS = frozenset({
    UnaryOp.ABS, UnaryOp.CEIL, UnaryOp.FLOOR,
    UnaryOp.SQRT, UnaryOp.LOG, UnaryOp.EXP,
})

_PRESERVES_DTYPE_OPS = frozenset({
    UnaryOp.NEG, UnaryOp.ABS, UnaryOp.CEIL, UnaryOp.FLOOR,
})

_PRODUCES_FLOAT_OPS = frozenset({
    UnaryOp.SQRT, UnaryOp.LOG, UnaryOp.EXP,
})


def is_null_check_op(op: UnaryOp) -> bool:
    """Return ``True`` if the operator checks for nulls (IS_NULL / IS_NOT_NULL)."""
    return op in _NULL_CHECK_OPS


def is_math_op(op: UnaryOp) -> bool:
    """Return ``True`` if the operator is a mathematical function."""
    return op in _MATH_OPS



def infer_unary_dtype(op: UnaryOp, input_dtype: DType) -> DType:
    """
    Infer the output dtype of a unary operation.

    Rules:
        - ``IS_NULL``, ``IS_NOT_NULL``, ``NOT`` → ``Bool``
        - ``NEG``, ``ABS``, ``CEIL``, ``FLOOR`` → same as input
        - ``SQRT``, ``LOG``, ``EXP`` → ``Float64`` (always float output)

    Args:
        op: The unary operator.
        input_dtype: The operand's type.

    Returns:
        The inferred output DType.

    Raises:
        TypeError: If the operation is not valid for the given type.
    """
    if op in _NULL_CHECK_OPS:
        return Bool

    if op == UnaryOp.NOT:
        return Bool

    if op in _PRESERVES_DTYPE_OPS:
        return input_dtype

    if op in _PRODUCES_FLOAT_OPS:
        return Float64

    raise TypeError(f"Cannot infer dtype for unknown unary op: {op}")



def _make_unary(op: UnaryOp, operand: Any) -> UnaryExpr:
    """Create a UnaryExpr, wrapping raw Python values into LiteralExpr."""
    return UnaryExpr(op, _wrap(operand))


def neg(operand: Any) -> UnaryExpr:
    """
    Negate a value: ``-operand``.

    Example::

        >>> neg(col("x"))
    """
    return _make_unary(UnaryOp.NEG, operand)


def not_(operand: Any) -> UnaryExpr:
    """
    Logical NOT: ``~operand``.

    Example::

        >>> not_(col("is_active"))
    """
    return _make_unary(UnaryOp.NOT, operand)


def is_null(operand: Any) -> UnaryExpr:
    """
    Check if values are null.

    Example::

        >>> is_null(col("email"))
    """
    return _make_unary(UnaryOp.IS_NULL, operand)


def is_not_null(operand: Any) -> UnaryExpr:
    """
    Check if values are not null.

    Example::

        >>> is_not_null(col("email"))
    """
    return _make_unary(UnaryOp.IS_NOT_NULL, operand)


def abs_(operand: Any) -> UnaryExpr:
    """
    Absolute value: ``|operand|``.

    Example::

        >>> abs_(col("profit_loss"))
    """
    return _make_unary(UnaryOp.ABS, operand)


def ceil(operand: Any) -> UnaryExpr:
    """
    Ceiling function: round up to nearest integer.

    Example::

        >>> ceil(col("price"))
    """
    return _make_unary(UnaryOp.CEIL, operand)


def floor(operand: Any) -> UnaryExpr:
    """
    Floor function: round down to nearest integer.

    Example::

        >>> floor(col("price"))
    """
    return _make_unary(UnaryOp.FLOOR, operand)


def sqrt(operand: Any) -> UnaryExpr:
    """
    Square root.

    Example::

        >>> sqrt(col("variance"))
    """
    return _make_unary(UnaryOp.SQRT, operand)


def log(operand: Any) -> UnaryExpr:
    """
    Natural logarithm (ln).

    Example::

        >>> log(col("value"))
    """
    return _make_unary(UnaryOp.LOG, operand)


def exp(operand: Any) -> UnaryExpr:
    """
    Exponential function (e^x).

    Example::

        >>> exp(col("log_value"))
    """
    return _make_unary(UnaryOp.EXP, operand)
