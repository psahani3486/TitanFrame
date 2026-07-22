"""
Tests for TitanFrame Type System (dtypes.py)
=============================================

Comprehensive tests covering:
    - All 17 type instances and their properties
    - Arrow ↔ DType bidirectional conversion
    - The full type promotion matrix (numeric cross-products)
    - Safe and unsafe cast checking
    - Edge cases: Null promotion, Bool→numeric, signed×unsigned
"""

import pytest
import numpy as np
import pyarrow as pa

from titanframe.core.dtypes import (
    DType,
    DTypeCategory,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    Bool, Utf8, Binary,
    Date, Datetime, Duration, Null,
    ALL_DTYPES,
    from_arrow,
    from_name,
    from_python_type,
    from_value,
    promote,
    can_cast,
)


# ---------------------------------------------------------------------------
# Type instances
# ---------------------------------------------------------------------------

class TestDTypeProperties:
    """Test basic properties of each DType singleton."""

    def test_all_dtypes_count(self):
        assert len(ALL_DTYPES) == 17

    def test_all_dtypes_unique_names(self):
        names = [dt.name for dt in ALL_DTYPES]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("dtype, expected_category", [
        (Int8, DTypeCategory.SIGNED_INTEGER),
        (Int16, DTypeCategory.SIGNED_INTEGER),
        (Int32, DTypeCategory.SIGNED_INTEGER),
        (Int64, DTypeCategory.SIGNED_INTEGER),
        (UInt8, DTypeCategory.UNSIGNED_INTEGER),
        (UInt16, DTypeCategory.UNSIGNED_INTEGER),
        (UInt32, DTypeCategory.UNSIGNED_INTEGER),
        (UInt64, DTypeCategory.UNSIGNED_INTEGER),
        (Float32, DTypeCategory.FLOATING),
        (Float64, DTypeCategory.FLOATING),
        (Bool, DTypeCategory.BOOLEAN),
        (Utf8, DTypeCategory.STRING),
        (Binary, DTypeCategory.BINARY),
        (Date, DTypeCategory.TEMPORAL),
        (Datetime, DTypeCategory.TEMPORAL),
        (Duration, DTypeCategory.TEMPORAL),
        (Null, DTypeCategory.NULL),
    ])
    def test_category(self, dtype: DType, expected_category: DTypeCategory):
        assert dtype.category == expected_category

    @pytest.mark.parametrize("dtype", [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64])
    def test_is_numeric(self, dtype: DType):
        assert dtype.is_numeric

    @pytest.mark.parametrize("dtype", [Bool, Utf8, Binary, Date, Datetime, Duration, Null])
    def test_is_not_numeric(self, dtype: DType):
        assert not dtype.is_numeric

    def test_is_integer(self):
        assert Int32.is_integer
        assert UInt64.is_integer
        assert not Float64.is_integer

    def test_is_float(self):
        assert Float32.is_float
        assert Float64.is_float
        assert not Int32.is_float

    def test_is_boolean(self):
        assert Bool.is_boolean
        assert not Int32.is_boolean

    def test_is_string(self):
        assert Utf8.is_string
        assert not Int32.is_string

    def test_is_temporal(self):
        assert Date.is_temporal
        assert Datetime.is_temporal
        assert Duration.is_temporal
        assert not Int32.is_temporal

    def test_is_null(self):
        assert Null.is_null
        assert not Int32.is_null

    def test_is_variable_width(self):
        assert Utf8.is_variable_width
        assert Binary.is_variable_width
        assert not Int32.is_variable_width
        assert not Float64.is_variable_width

    @pytest.mark.parametrize("dtype, expected_width", [
        (Int8, 1), (Int16, 2), (Int32, 4), (Int64, 8),
        (UInt8, 1), (UInt16, 2), (UInt32, 4), (UInt64, 8),
        (Float32, 4), (Float64, 8),
        (Bool, 1),
        (Utf8, -1), (Binary, -1),
    ])
    def test_byte_width(self, dtype: DType, expected_width: int):
        assert dtype.byte_width == expected_width

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_repr_is_name(self, dtype: DType):
        assert repr(dtype) == dtype.name
        assert str(dtype) == dtype.name


# ---------------------------------------------------------------------------
# Arrow conversion
# ---------------------------------------------------------------------------

class TestArrowConversion:
    """Test bidirectional Arrow type conversion."""

    @pytest.mark.parametrize("dtype, arrow_type", [
        (Int8, pa.int8()),
        (Int16, pa.int16()),
        (Int32, pa.int32()),
        (Int64, pa.int64()),
        (UInt8, pa.uint8()),
        (UInt16, pa.uint16()),
        (UInt32, pa.uint32()),
        (UInt64, pa.uint64()),
        (Float32, pa.float32()),
        (Float64, pa.float64()),
        (Bool, pa.bool_()),
        (Utf8, pa.utf8()),
        (Binary, pa.binary()),
        (Null, pa.null()),
    ])
    def test_round_trip(self, dtype: DType, arrow_type: pa.DataType):
        """DType → arrow_type → from_arrow → same DType."""
        assert dtype.arrow_type == arrow_type
        assert from_arrow(arrow_type) == dtype

    def test_from_arrow_timestamp_ns(self):
        """Parameterized timestamps should map to Datetime."""
        assert from_arrow(pa.timestamp("ns")) == Datetime
        assert from_arrow(pa.timestamp("s")) == Datetime

    def test_from_arrow_duration(self):
        assert from_arrow(pa.duration("ns")) == Duration

    def test_from_arrow_date64(self):
        assert from_arrow(pa.date64()) == Date

    def test_from_arrow_large_utf8(self):
        assert from_arrow(pa.large_utf8()) == Utf8

    def test_from_arrow_large_binary(self):
        assert from_arrow(pa.large_binary()) == Binary

    def test_from_arrow_unknown_raises(self):
        with pytest.raises(TypeError, match="No TitanFrame DType"):
            from_arrow(pa.map_(pa.utf8(), pa.int32()))


# ---------------------------------------------------------------------------
# Name and Python type conversion
# ---------------------------------------------------------------------------

class TestNameConversion:
    """Test lookup by name."""

    @pytest.mark.parametrize("name", ["int32", "INT32", "Int32"])
    def test_case_insensitive(self, name: str):
        assert from_name(name) == Int32

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="Unknown DType"):
            from_name("complex128")


class TestPythonTypeConversion:

    def test_int(self):
        assert from_python_type(int) == Int64

    def test_float(self):
        assert from_python_type(float) == Float64

    def test_str(self):
        assert from_python_type(str) == Utf8

    def test_bool(self):
        assert from_python_type(bool) == Bool

    def test_none(self):
        assert from_python_type(type(None)) == Null

    def test_unsupported_raises(self):
        with pytest.raises(TypeError, match="Cannot infer"):
            from_python_type(list)


class TestValueConversion:

    def test_int_value(self):
        assert from_value(42) == Int64

    def test_float_value(self):
        assert from_value(3.14) == Float64

    def test_str_value(self):
        assert from_value("hello") == Utf8

    def test_bool_value(self):
        assert from_value(True) == Bool

    def test_none_value(self):
        assert from_value(None) == Null


# ---------------------------------------------------------------------------
# Type promotion matrix
# ---------------------------------------------------------------------------

class TestPromotion:
    """Test the full type promotion matrix."""

    # Same type → same type
    @pytest.mark.parametrize("dtype", [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64, Bool, Null])
    def test_same_type(self, dtype: DType):
        assert promote(dtype, dtype) == dtype

    # Null + X → X
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_null_promotion(self, dtype: DType):
        assert promote(Null, dtype) == dtype
        assert promote(dtype, Null) == dtype

    # Bool + numeric → numeric
    @pytest.mark.parametrize("dtype", [Int8, Int32, Int64, Float32, Float64])
    def test_bool_to_numeric(self, dtype: DType):
        assert promote(Bool, dtype) == dtype

    # Signed integer widening
    def test_int8_int16(self):
        assert promote(Int8, Int16) == Int16

    def test_int8_int32(self):
        assert promote(Int8, Int32) == Int32

    def test_int8_int64(self):
        assert promote(Int8, Int64) == Int64

    def test_int16_int32(self):
        assert promote(Int16, Int32) == Int32

    def test_int32_int64(self):
        assert promote(Int32, Int64) == Int64

    # Unsigned integer widening
    def test_uint8_uint16(self):
        assert promote(UInt8, UInt16) == UInt16

    def test_uint32_uint64(self):
        assert promote(UInt32, UInt64) == UInt64

    # Signed × Unsigned → wider signed (critical for correctness!)
    def test_int8_uint8(self):
        """Int8 + UInt8 → Int16 to prevent overflow."""
        assert promote(Int8, UInt8) == Int16

    def test_int16_uint16(self):
        """Int16 + UInt16 → Int32."""
        assert promote(Int16, UInt16) == Int32

    def test_int32_uint32(self):
        """Int32 + UInt32 → Int64."""
        assert promote(Int32, UInt32) == Int64

    def test_int8_uint32(self):
        assert promote(Int8, UInt32) == Int64

    # Integer + Float → Float
    def test_int32_float64(self):
        assert promote(Int32, Float64) == Float64

    def test_int64_float32(self):
        """Int64 + Float32 → Float64 (Float32 can't hold Int64 precision)."""
        assert promote(Int64, Float32) == Float64

    def test_int8_float32(self):
        """Int8 + Float32 → Float32 (small enough)."""
        assert promote(Int8, Float32) == Float32

    # Float widening
    def test_float32_float64(self):
        assert promote(Float32, Float64) == Float64

    # Symmetry check
    def test_promotion_is_symmetric(self):
        """promote(A, B) == promote(B, A) for all numeric pairs."""
        numeric = [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64]
        for a in numeric:
            for b in numeric:
                assert promote(a, b) == promote(b, a), f"Asymmetry: promote({a}, {b})"

    # Invalid promotions
    def test_string_int_raises(self):
        with pytest.raises(TypeError, match="Cannot promote"):
            promote(Utf8, Int32)

    def test_date_float_raises(self):
        with pytest.raises(TypeError, match="Cannot promote"):
            promote(Date, Float64)


# ---------------------------------------------------------------------------
# Cast checking
# ---------------------------------------------------------------------------

class TestCanCast:
    """Test safe and unsafe casting rules."""

    def test_same_type_always_castable(self):
        for dt in ALL_DTYPES:
            assert can_cast(dt, dt, safe=True)
            assert can_cast(dt, dt, safe=False)

    def test_null_casts_to_anything(self):
        for dt in ALL_DTYPES:
            assert can_cast(Null, dt, safe=True)

    # Safe: widening only
    def test_safe_int32_to_int64(self):
        assert can_cast(Int32, Int64, safe=True)

    def test_safe_int64_to_int32_fails(self):
        """Narrowing is not safe."""
        assert not can_cast(Int64, Int32, safe=True)

    def test_safe_float32_to_float64(self):
        assert can_cast(Float32, Float64, safe=True)

    def test_safe_int32_to_float64(self):
        assert can_cast(Int32, Float64, safe=True)

    def test_safe_string_to_int_fails(self):
        assert not can_cast(Utf8, Int32, safe=True)

    # Unsafe: allow lossy
    def test_unsafe_int64_to_int32(self):
        assert can_cast(Int64, Int32, safe=False)

    def test_unsafe_float64_to_int32(self):
        assert can_cast(Float64, Int32, safe=False)

    def test_unsafe_string_to_int(self):
        assert can_cast(Utf8, Int32, safe=False)

    def test_unsafe_int_to_string(self):
        assert can_cast(Int32, Utf8, safe=False)

    def test_unsafe_bool_to_int(self):
        assert can_cast(Bool, Int32, safe=False)
