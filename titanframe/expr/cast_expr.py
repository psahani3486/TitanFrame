"""
Cast Expression — ``CastExpr``
================================

Extended module for type casting operations. Provides:
    - Safe vs. unsafe cast validation
    - Convenience ``cast()`` function
    - Supported cast matrix queries
    - ``try_cast()`` expression variant (returns null on failure)

The core ``CastExpr`` class lives in :mod:`base` and is re-exported here.
This module adds validation logic, a TryCastExpr variant, and convenience
factory functions.

Example::

    >>> from titanframe.expr.cast_expr import cast, try_cast
    >>> from titanframe.expr.column_expr import col
    >>> from titanframe.core.dtypes import Float64
    >>> expr = cast(col("price"), Float64)
    >>> safe_expr = try_cast(col("maybe_number"), Float64)
"""

from __future__ import annotations

from typing import Any, Optional

from titanframe.core.dtypes import (
    DType,
    can_cast,
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    Utf8,
    Date, Datetime, Duration,
)
from titanframe.expr.base import (
    CastExpr,
    Expr,
    _wrap,
)

__all__ = [
    "CastExpr",
    "TryCastExpr",
    "cast",
    "try_cast",
    "validate_cast",
    "is_identity_cast",
    "NUMERIC_CAST_PAIRS",
    "TEMPORAL_CAST_PAIRS",
    "STRING_CAST_PAIRS",
]



class TryCastExpr(Expr):
    """
    Safe type cast: ``TRY_CAST(child AS dtype)``.

    Unlike :class:`CastExpr`, a ``TryCastExpr`` returns ``null`` for values
    that cannot be converted, rather than raising an error. This is useful
    for parsing messy string data into numeric types.

    Example::

        >>> try_cast(col("user_input"), Int64)
        # "123" → 123, "abc" → null, "" → null

    Attributes:
        child: The expression to cast.
        target_dtype: The target data type.
    """

    __slots__ = ("child", "target_dtype")

    def __init__(self, child: Expr, target_dtype: DType):
        self.child = child
        self.target_dtype = target_dtype

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return TryCastExpr(new_children[0], self.target_dtype)

    def display(self, indent: int = 0) -> str:
        return f"TryCastExpr({self.child.display()}, {self.target_dtype})"

    def __hash__(self) -> int:
        return hash(("try_cast", self.child, self.target_dtype))



class CastError(Exception):
    """Raised when a type cast is invalid or unsafe when safe mode is required."""
    pass


def validate_cast(
    source_dtype: DType,
    target_dtype: DType,
    safe: bool = True,
) -> None:
    """
    Validate that a cast from ``source_dtype`` to ``target_dtype`` is allowed.

    Args:
        source_dtype: The source data type.
        target_dtype: The target data type.
        safe: If ``True``, only allows lossless casts.

    Raises:
        CastError: If the cast is not allowed under the given safety mode.
    """
    if source_dtype == target_dtype:
        return

    if not can_cast(source_dtype, target_dtype, safe=safe):
        mode = "safe" if safe else "unsafe"
        raise CastError(
            f"Cannot {mode}-cast {source_dtype} → {target_dtype}. "
            f"{'Use safe=False or try_cast for potentially lossy conversions.' if safe else ''}"
        )


def is_identity_cast(source_dtype: DType, target_dtype: DType) -> bool:
    """
    Return ``True`` if the cast is a no-op (same source and target type).

    Identity casts can be eliminated by the query optimizer to avoid
    unnecessary computation.
    """
    return source_dtype == target_dtype



NUMERIC_CAST_PAIRS: list[tuple[DType, DType]] = [
    (Int8, Int16), (Int8, Int32), (Int8, Int64),
    (Int16, Int32), (Int16, Int64),
    (Int32, Int64),
    (UInt8, UInt16), (UInt8, UInt32), (UInt8, UInt64),
    (UInt16, UInt32), (UInt16, UInt64),
    (UInt32, UInt64),
    (Int8, Float32), (Int16, Float32), (Int8, Float64), (Int16, Float64),
    (Int32, Float64), (Int64, Float64),
    (UInt8, Float32), (UInt16, Float32), (UInt8, Float64), (UInt16, Float64),
    (UInt32, Float64), (UInt64, Float64),
    (Float32, Float64),
    (Bool, Int8), (Bool, Int32), (Bool, Int64), (Bool, Float64),
]

TEMPORAL_CAST_PAIRS: list[tuple[DType, DType]] = [
    (Date, Datetime),
    (Datetime, Date),
]

STRING_CAST_PAIRS: list[tuple[DType, DType]] = [
    (Utf8, Int64), (Utf8, Float64), (Utf8, Bool),
    (Utf8, Date), (Utf8, Datetime),
    (Int64, Utf8), (Float64, Utf8), (Bool, Utf8),
    (Date, Utf8), (Datetime, Utf8),
]



def cast(operand: Any, target_dtype: DType, safe: bool = True) -> CastExpr:
    """
    Create a type cast expression.

    Args:
        operand: Column expression or value to cast.
        target_dtype: Target data type.
        safe: If ``True`` (default), only lossless casts are allowed.

    Returns:
        A :class:`CastExpr`.

    Example::

        >>> cast(col("price"), Float64)
        >>> cast(col("count"), Int32, safe=False)
    """
    expr = _wrap(operand)
    return CastExpr(expr, target_dtype)


def try_cast(operand: Any, target_dtype: DType) -> TryCastExpr:
    """
    Create a safe cast expression that returns null on failure.

    Unlike :func:`cast`, this never raises an error for invalid values.
    Instead, values that cannot be converted are set to ``null``.

    Args:
        operand: Column expression or value to cast.
        target_dtype: Target data type.

    Returns:
        A :class:`TryCastExpr`.

    Example::

        >>> try_cast(col("user_input"), Int64)
        # "123" → 123, "abc" → null
    """
    return TryCastExpr(_wrap(operand), target_dtype)



def to_int64(operand: Any) -> CastExpr:
    """Shorthand for ``cast(operand, Int64)``."""
    return cast(operand, Int64)


def to_float64(operand: Any) -> CastExpr:
    """Shorthand for ``cast(operand, Float64)``."""
    return cast(operand, Float64)


def to_string(operand: Any) -> CastExpr:
    """Shorthand for ``cast(operand, Utf8)``."""
    return cast(operand, Utf8, safe=False)


def to_bool(operand: Any) -> CastExpr:
    """Shorthand for ``cast(operand, Bool)``."""
    return cast(operand, Bool, safe=False)


def to_date(operand: Any) -> CastExpr:
    """Shorthand for ``cast(operand, Date)``."""
    return cast(operand, Date, safe=False)


def to_datetime(operand: Any) -> CastExpr:
    """Shorthand for ``cast(operand, Datetime)``."""
    return cast(operand, Datetime, safe=False)
