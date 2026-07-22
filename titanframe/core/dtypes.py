"""
TitanFrame Type System
======================

Defines the canonical type system used throughout TitanFrame. Every type knows
its Apache Arrow equivalent, NumPy dtype, byte width, and participates in a
promotion matrix that governs mixed-type arithmetic.

Design Principles:
    - Types are singleton-like: ``Int32`` is a module-level instance, not a class.
    - Each DType is immutable and hashable.
    - The promotion matrix is symmetric and transitive.
    - Arrow is the source of truth for in-memory representation.

Example::

    >>> from titanframe.core.dtypes import Int32, Float64, promote
    >>> promote(Int32, Float64)
    Float64
    >>> Int32.arrow_type
    pyarrow.int32()
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

import numpy as np
import pyarrow as pa


# ---------------------------------------------------------------------------
# DType category — used for grouping types in the promotion matrix
# ---------------------------------------------------------------------------

class DTypeCategory(enum.Enum):
    """High-level category of a data type."""
    SIGNED_INTEGER = "signed_integer"
    UNSIGNED_INTEGER = "unsigned_integer"
    FLOATING = "floating"
    BOOLEAN = "boolean"
    STRING = "string"
    BINARY = "binary"
    TEMPORAL = "temporal"
    NULL = "null"
    NESTED = "nested"


# ---------------------------------------------------------------------------
# DType — the base type class
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DType:
    """
    Base data type descriptor.

    Each instance encapsulates the mapping between TitanFrame's logical type
    system, Apache Arrow's physical type, and NumPy's dtype.

    Attributes:
        name: Human-readable type name (e.g., ``"Int32"``).
        arrow_type: Corresponding ``pyarrow.DataType``.
        numpy_dtype: Corresponding ``numpy.dtype`` (``None`` for non-numeric types).
        byte_width: Number of bytes per element (``-1`` for variable-width).
        category: High-level grouping (integer, float, string, etc.).
        nullable: Whether this type supports null values (always ``True`` in Arrow).
        _rank: Internal rank for type promotion (higher wins).
    """

    name: str
    arrow_type: pa.DataType
    numpy_dtype: Optional[np.dtype]
    byte_width: int
    category: DTypeCategory
    nullable: bool = True
    _rank: int = 0

    # ---- Identity ----

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    # ---- Queries ----

    @property
    def is_numeric(self) -> bool:
        """Return ``True`` if this type supports arithmetic."""
        return self.category in (
            DTypeCategory.SIGNED_INTEGER,
            DTypeCategory.UNSIGNED_INTEGER,
            DTypeCategory.FLOATING,
        )

    @property
    def is_integer(self) -> bool:
        """Return ``True`` if this is a signed or unsigned integer type."""
        return self.category in (
            DTypeCategory.SIGNED_INTEGER,
            DTypeCategory.UNSIGNED_INTEGER,
        )

    @property
    def is_float(self) -> bool:
        """Return ``True`` if this is a floating-point type."""
        return self.category == DTypeCategory.FLOATING

    @property
    def is_boolean(self) -> bool:
        return self.category == DTypeCategory.BOOLEAN

    @property
    def is_string(self) -> bool:
        return self.category == DTypeCategory.STRING

    @property
    def is_temporal(self) -> bool:
        return self.category == DTypeCategory.TEMPORAL

    @property
    def is_null(self) -> bool:
        return self.category == DTypeCategory.NULL

    @property
    def is_nested(self) -> bool:
        return self.category == DTypeCategory.NESTED

    @property
    def is_variable_width(self) -> bool:
        """Return ``True`` if elements have variable byte width (strings, binary)."""
        return self.byte_width == -1


# ---------------------------------------------------------------------------
# Singleton type instances
# ---------------------------------------------------------------------------

# Signed integers (rank 10–13)
Int8 = DType("Int8", pa.int8(), np.dtype("int8"), 1, DTypeCategory.SIGNED_INTEGER, _rank=10)
Int16 = DType("Int16", pa.int16(), np.dtype("int16"), 2, DTypeCategory.SIGNED_INTEGER, _rank=11)
Int32 = DType("Int32", pa.int32(), np.dtype("int32"), 4, DTypeCategory.SIGNED_INTEGER, _rank=12)
Int64 = DType("Int64", pa.int64(), np.dtype("int64"), 8, DTypeCategory.SIGNED_INTEGER, _rank=13)

# Unsigned integers (rank 10–13)
UInt8 = DType("UInt8", pa.uint8(), np.dtype("uint8"), 1, DTypeCategory.UNSIGNED_INTEGER, _rank=10)
UInt16 = DType("UInt16", pa.uint16(), np.dtype("uint16"), 2, DTypeCategory.UNSIGNED_INTEGER, _rank=11)
UInt32 = DType("UInt32", pa.uint32(), np.dtype("uint32"), 4, DTypeCategory.UNSIGNED_INTEGER, _rank=12)
UInt64 = DType("UInt64", pa.uint64(), np.dtype("uint64"), 8, DTypeCategory.UNSIGNED_INTEGER, _rank=13)

# Floating point (rank 20–21)
Float32 = DType("Float32", pa.float32(), np.dtype("float32"), 4, DTypeCategory.FLOATING, _rank=20)
Float64 = DType("Float64", pa.float64(), np.dtype("float64"), 8, DTypeCategory.FLOATING, _rank=21)

# Boolean (rank 1)
Bool = DType("Bool", pa.bool_(), np.dtype("bool"), 1, DTypeCategory.BOOLEAN, _rank=1)

# String types (variable width)
Utf8 = DType("Utf8", pa.utf8(), None, -1, DTypeCategory.STRING, _rank=30)
Binary = DType("Binary", pa.binary(), None, -1, DTypeCategory.BINARY, _rank=31)

# Temporal types
Date = DType("Date", pa.date32(), None, 4, DTypeCategory.TEMPORAL, _rank=40)
Datetime = DType("Datetime", pa.timestamp("us"), None, 8, DTypeCategory.TEMPORAL, _rank=41)
Duration = DType("Duration", pa.duration("us"), None, 8, DTypeCategory.TEMPORAL, _rank=42)

# Null type (rank 0 — promotes to anything)
Null = DType("Null", pa.null(), None, 0, DTypeCategory.NULL, _rank=0)


# ---------------------------------------------------------------------------
# All types registry — for lookup and iteration
# ---------------------------------------------------------------------------

ALL_DTYPES: list[DType] = [
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    Bool,
    Utf8, Binary,
    Date, Datetime, Duration,
    Null,
]

_ARROW_TO_DTYPE: dict[pa.DataType, DType] = {}
_NAME_TO_DTYPE: dict[str, DType] = {}

for _dt in ALL_DTYPES:
    _ARROW_TO_DTYPE[_dt.arrow_type] = _dt
    _NAME_TO_DTYPE[_dt.name.lower()] = _dt


def from_arrow(arrow_type: pa.DataType) -> DType:
    """
    Convert an Arrow type to a TitanFrame DType.

    Falls back to heuristic matching for parameterized types (e.g.,
    ``timestamp('ns')`` maps to ``Datetime``).

    Raises:
        TypeError: If the Arrow type has no TitanFrame equivalent.
    """
    # Direct lookup
    if arrow_type in _ARROW_TO_DTYPE:
        return _ARROW_TO_DTYPE[arrow_type]

    # Parameterized type matching
    if pa.types.is_timestamp(arrow_type):
        return Datetime
    if pa.types.is_duration(arrow_type):
        return Duration
    if pa.types.is_date(arrow_type):
        return Date
    if pa.types.is_large_string(arrow_type) or pa.types.is_string(arrow_type):
        return Utf8
    if pa.types.is_large_binary(arrow_type) or pa.types.is_binary(arrow_type):
        return Binary
    if pa.types.is_boolean(arrow_type):
        return Bool
    if pa.types.is_null(arrow_type):
        return Null
    if pa.types.is_signed_integer(arrow_type):
        # Match by byte width
        width = arrow_type.bit_width // 8
        return {1: Int8, 2: Int16, 4: Int32, 8: Int64}[width]
    if pa.types.is_unsigned_integer(arrow_type):
        width = arrow_type.bit_width // 8
        return {1: UInt8, 2: UInt16, 4: UInt32, 8: UInt64}[width]
    if pa.types.is_floating(arrow_type):
        width = arrow_type.bit_width // 8
        return {4: Float32, 8: Float64}[width]

    raise TypeError(f"No TitanFrame DType for Arrow type: {arrow_type}")


def from_name(name: str) -> DType:
    """
    Look up a DType by its case-insensitive name.

    Raises:
        KeyError: If the name is not recognized.
    """
    key = name.lower()
    if key not in _NAME_TO_DTYPE:
        raise KeyError(f"Unknown DType name: {name!r}. Valid names: {list(_NAME_TO_DTYPE.keys())}")
    return _NAME_TO_DTYPE[key]


def from_python_type(python_type: type) -> DType:
    """
    Infer a TitanFrame DType from a Python built-in type.

    Mapping:
        int → Int64, float → Float64, str → Utf8, bool → Bool, None → Null

    Raises:
        TypeError: If the Python type has no mapping.
    """
    _PYTHON_TO_DTYPE: dict[type, DType] = {
        int: Int64,
        float: Float64,
        str: Utf8,
        bool: Bool,
        type(None): Null,
    }
    if python_type not in _PYTHON_TO_DTYPE:
        raise TypeError(f"Cannot infer DType from Python type: {python_type}")
    return _PYTHON_TO_DTYPE[python_type]


def from_value(value: Any) -> DType:
    """Infer a DType from a Python value."""
    if value is None:
        return Null
    return from_python_type(type(value))


# ---------------------------------------------------------------------------
# Type promotion
# ---------------------------------------------------------------------------

# Explicit promotion table for mixed-type arithmetic.
# The key is (left_dtype, right_dtype), value is the result dtype.
# This table is symmetric: (A, B) → C implies (B, A) → C.
_PROMOTION_TABLE: dict[tuple[DType, DType], DType] = {}


def _register_promotion(a: DType, b: DType, result: DType) -> None:
    """Register a bidirectional promotion rule."""
    _PROMOTION_TABLE[(a, b)] = result
    _PROMOTION_TABLE[(b, a)] = result


def _build_promotion_table() -> None:
    """
    Build the full type promotion matrix.

    Rules (following NumPy/Pandas conventions):
        1. Null promotes to anything → the other type.
        2. Bool promotes to any numeric → that numeric type.
        3. Integer + Integer → wider integer (signed wins over unsigned at same width).
        4. Integer + Float → Float (at least Float64 for Int64).
        5. Float + Float → wider float.
        6. Same type → same type.
    """
    numeric_types = [
        Int8, Int16, Int32, Int64,
        UInt8, UInt16, UInt32, UInt64,
        Float32, Float64,
    ]

    # Rule: same type → same type
    for dt in ALL_DTYPES:
        _PROMOTION_TABLE[(dt, dt)] = dt

    # Rule: Null + X → X
    for dt in ALL_DTYPES:
        _register_promotion(Null, dt, dt)

    # Rule: Bool + numeric → numeric
    for dt in numeric_types:
        _register_promotion(Bool, dt, dt)

    # Rule: numeric promotions
    signed = [Int8, Int16, Int32, Int64]
    unsigned = [UInt8, UInt16, UInt32, UInt64]
    floats = [Float32, Float64]

    # Signed + Signed → wider
    for i, a in enumerate(signed):
        for j, b in enumerate(signed):
            if i != j:
                result = signed[max(i, j)]
                _register_promotion(a, b, result)

    # Unsigned + Unsigned → wider
    for i, a in enumerate(unsigned):
        for j, b in enumerate(unsigned):
            if i != j:
                result = unsigned[max(i, j)]
                _register_promotion(a, b, result)

    # Signed + Unsigned → signed of next width (to avoid overflow)
    # e.g., Int32 + UInt32 → Int64, Int8 + UInt8 → Int16
    _cross_promote = {
        (Int8, UInt8): Int16,
        (Int8, UInt16): Int32,
        (Int8, UInt32): Int64,
        (Int8, UInt64): Int64,  # best effort
        (Int16, UInt8): Int16,
        (Int16, UInt16): Int32,
        (Int16, UInt32): Int64,
        (Int16, UInt64): Int64,
        (Int32, UInt8): Int32,
        (Int32, UInt16): Int32,
        (Int32, UInt32): Int64,
        (Int32, UInt64): Int64,
        (Int64, UInt8): Int64,
        (Int64, UInt16): Int64,
        (Int64, UInt32): Int64,
        (Int64, UInt64): Int64,  # may lose precision for huge uint64
    }
    for (a, b), result in _cross_promote.items():
        _register_promotion(a, b, result)

    # Integer + Float → Float
    for int_dt in signed + unsigned:
        _register_promotion(int_dt, Float32, Float32 if int_dt.byte_width <= 2 else Float64)
        _register_promotion(int_dt, Float64, Float64)

    # Float + Float → wider
    _register_promotion(Float32, Float64, Float64)


_build_promotion_table()


def promote(left: DType, right: DType) -> DType:
    """
    Determine the result type when combining two types in arithmetic.

    Uses an explicit promotion table that follows NumPy/Pandas conventions.

    Args:
        left: Left operand type.
        right: Right operand type.

    Returns:
        The promoted DType.

    Raises:
        TypeError: If the types cannot be promoted (e.g., Utf8 + Int32).

    Example::

        >>> promote(Int32, Float64)
        Float64
        >>> promote(Int8, UInt8)
        Int16
    """
    key = (left, right)
    if key in _PROMOTION_TABLE:
        return _PROMOTION_TABLE[key]
    raise TypeError(
        f"Cannot promote types {left} and {right}. "
        f"Arithmetic between {left.category.value} and {right.category.value} is not supported."
    )


def can_cast(source: DType, target: DType, safe: bool = True) -> bool:
    """
    Check if ``source`` can be cast to ``target``.

    Args:
        source: The source data type.
        target: The target data type.
        safe: If ``True``, only allows lossless casts. If ``False``, allows
              potentially lossy casts (e.g., Float64 → Int32).

    Returns:
        ``True`` if the cast is valid.
    """
    if source == target:
        return True
    if source == Null:
        return True  # Null can become anything

    if safe:
        # Safe: only widening numeric casts
        try:
            promoted = promote(source, target)
            return promoted == target
        except TypeError:
            return False
    else:
        # Unsafe: allow any numeric→numeric, string→numeric (parsed), etc.
        if source.is_numeric and target.is_numeric:
            return True
        if source.is_string and target.is_numeric:
            return True
        if source.is_numeric and target.is_string:
            return True
        if source.is_boolean and target.is_numeric:
            return True
        if source.is_numeric and target.is_boolean:
            return True
        return False
