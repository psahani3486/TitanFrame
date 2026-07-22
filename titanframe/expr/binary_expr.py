"""
Binary Expression — ``BinaryExpr``
====================================

Extended module for binary (two-operand) operations. Provides:
    - Arithmetic: ``Add, Sub, Mul, TrueDiv, FloorDiv, Mod, Pow``
    - Comparison: ``Eq, Ne, Lt, Le, Gt, Ge``
    - Logical: ``And, Or, Xor``

The core ``BinaryExpr`` class lives in :mod:`base` and is re-exported here.
This module adds type-inference helpers, operator classification utilities,
and convenience factory functions.

Example::

    >>> from titanframe.expr.binary_expr import add, gt, and_
    >>> expr = and_(gt(col("x"), lit(10)), gt(col("y"), lit(20)))
"""

from __future__ import annotations

from typing import Any

from titanframe.core.dtypes import DType, Bool, promote
from titanframe.expr.base import (
    BinaryExpr,
    Expr,
    Op,
    _wrap,
)

# Re-export for users importing from this module
__all__ = [
    "BinaryExpr",
    "Op",
    "add",
    "sub",
    "mul",
    "true_div",
    "floor_div",
    "mod",
    "pow_",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "and_",
    "or_",
    "xor",
    "infer_binary_dtype",
    "is_arithmetic_op",
    "is_comparison_op",
    "is_logical_op",
]


# ---------------------------------------------------------------------------
# Operator classification
# ---------------------------------------------------------------------------

_ARITHMETIC_OPS = frozenset({
    Op.ADD, Op.SUB, Op.MUL, Op.TRUE_DIV, Op.FLOOR_DIV, Op.MOD, Op.POW,
})

_COMPARISON_OPS = frozenset({
    Op.EQ, Op.NE, Op.LT, Op.LE, Op.GT, Op.GE,
})

_LOGICAL_OPS = frozenset({
    Op.AND, Op.OR, Op.XOR,
})


def is_arithmetic_op(op: Op) -> bool:
    """Return ``True`` if the operator performs arithmetic."""
    return op in _ARITHMETIC_OPS


def is_comparison_op(op: Op) -> bool:
    """Return ``True`` if the operator performs comparison."""
    return op in _COMPARISON_OPS


def is_logical_op(op: Op) -> bool:
    """Return ``True`` if the operator performs logical combination."""
    return op in _LOGICAL_OPS


# ---------------------------------------------------------------------------
# Type inference for binary operations
# ---------------------------------------------------------------------------

def infer_binary_dtype(op: Op, left_dtype: DType, right_dtype: DType) -> DType:
    """
    Infer the output dtype of a binary operation.

    Rules:
        - Arithmetic ops → promote(left, right)
        - Division always → Float64 (for true_div)
        - Comparison ops → Bool
        - Logical ops → Bool (inputs must be Bool)

    Args:
        op: The binary operator.
        left_dtype: Left operand's type.
        right_dtype: Right operand's type.

    Returns:
        The inferred output DType.

    Raises:
        TypeError: If the operation is not valid for the given types.
    """
    from titanframe.core.dtypes import Float64

    if is_comparison_op(op):
        return Bool

    if is_logical_op(op):
        return Bool

    if is_arithmetic_op(op):
        if op == Op.TRUE_DIV:
            # Division always produces float
            if left_dtype.is_integer and right_dtype.is_integer:
                return Float64
            return promote(left_dtype, right_dtype)

        return promote(left_dtype, right_dtype)

    raise TypeError(f"Cannot infer dtype for unknown op: {op}")


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def _make_binary(op: Op, left: Any, right: Any) -> BinaryExpr:
    """Create a BinaryExpr, wrapping raw Python values into LiteralExpr."""
    return BinaryExpr(op, _wrap(left), _wrap(right))


def add(left: Any, right: Any) -> BinaryExpr:
    """Create an addition expression: ``left + right``."""
    return _make_binary(Op.ADD, left, right)


def sub(left: Any, right: Any) -> BinaryExpr:
    """Create a subtraction expression: ``left - right``."""
    return _make_binary(Op.SUB, left, right)


def mul(left: Any, right: Any) -> BinaryExpr:
    """Create a multiplication expression: ``left * right``."""
    return _make_binary(Op.MUL, left, right)


def true_div(left: Any, right: Any) -> BinaryExpr:
    """Create a true division expression: ``left / right``."""
    return _make_binary(Op.TRUE_DIV, left, right)


def floor_div(left: Any, right: Any) -> BinaryExpr:
    """Create a floor division expression: ``left // right``."""
    return _make_binary(Op.FLOOR_DIV, left, right)


def mod(left: Any, right: Any) -> BinaryExpr:
    """Create a modulo expression: ``left % right``."""
    return _make_binary(Op.MOD, left, right)


def pow_(left: Any, right: Any) -> BinaryExpr:
    """Create a power expression: ``left ** right``."""
    return _make_binary(Op.POW, left, right)


def eq(left: Any, right: Any) -> BinaryExpr:
    """Create an equality comparison: ``left == right``."""
    return _make_binary(Op.EQ, left, right)


def ne(left: Any, right: Any) -> BinaryExpr:
    """Create a not-equal comparison: ``left != right``."""
    return _make_binary(Op.NE, left, right)


def lt(left: Any, right: Any) -> BinaryExpr:
    """Create a less-than comparison: ``left < right``."""
    return _make_binary(Op.LT, left, right)


def le(left: Any, right: Any) -> BinaryExpr:
    """Create a less-or-equal comparison: ``left <= right``."""
    return _make_binary(Op.LE, left, right)


def gt(left: Any, right: Any) -> BinaryExpr:
    """Create a greater-than comparison: ``left > right``."""
    return _make_binary(Op.GT, left, right)


def ge(left: Any, right: Any) -> BinaryExpr:
    """Create a greater-or-equal comparison: ``left >= right``."""
    return _make_binary(Op.GE, left, right)


def and_(left: Any, right: Any) -> BinaryExpr:
    """Create a logical AND: ``left & right``."""
    return _make_binary(Op.AND, left, right)


def or_(left: Any, right: Any) -> BinaryExpr:
    """Create a logical OR: ``left | right``."""
    return _make_binary(Op.OR, left, right)


def xor(left: Any, right: Any) -> BinaryExpr:
    """Create a logical XOR: ``left ^ right``."""
    return _make_binary(Op.XOR, left, right)
