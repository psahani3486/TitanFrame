"""
Tests for TitanFrame Phase 2 Extended Expression Modules
=========================================================

Tests covering the four new expression modules:
    - binary_expr.py: factory functions, type inference, operator classification
    - unary_expr.py: factory functions, type inference, operator classification
    - agg_expr.py: factory functions, QuantileExpr, partial agg, type inference
    - cast_expr.py: CastExpr, TryCastExpr, validation, shorthands

Also tests the expr/__init__.py comprehensive exports and updated
DataFrame evaluator for TryCastExpr and QuantileExpr.
"""

import pytest

from titanframe.core.dtypes import (
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64, Bool, Utf8, Date, Datetime, Null,
)
from titanframe.expr.base import (
    Expr, BinaryExpr, UnaryExpr, AggExpr, CastExpr, AliasExpr,
    Op, UnaryOp, AggOp,
)
from titanframe.expr.column_expr import col, ColumnExpr
from titanframe.expr.literal_expr import lit, LiteralExpr


# ===========================================================================
# Binary Expression Module
# ===========================================================================

class TestBinaryExprModule:
    """Tests for titanframe.expr.binary_expr."""

    def test_import(self):
        from titanframe.expr.binary_expr import (
            add, sub, mul, true_div, floor_div, mod, pow_,
            eq, ne, lt, le, gt, ge,
            and_, or_, xor,
            infer_binary_dtype,
            is_arithmetic_op, is_comparison_op, is_logical_op,
        )

    # ---- Factory functions ----

    def test_add_factory(self):
        from titanframe.expr.binary_expr import add
        expr = add(col("a"), col("b"))
        assert isinstance(expr, BinaryExpr)
        assert expr.op == Op.ADD

    def test_sub_factory(self):
        from titanframe.expr.binary_expr import sub
        expr = sub(col("a"), 10)
        assert isinstance(expr, BinaryExpr)
        assert expr.op == Op.SUB
        assert isinstance(expr.right, LiteralExpr)

    def test_mul_factory(self):
        from titanframe.expr.binary_expr import mul
        expr = mul(col("x"), 2.0)
        assert expr.op == Op.MUL

    def test_true_div_factory(self):
        from titanframe.expr.binary_expr import true_div
        expr = true_div(col("a"), col("b"))
        assert expr.op == Op.TRUE_DIV

    def test_floor_div_factory(self):
        from titanframe.expr.binary_expr import floor_div
        expr = floor_div(col("a"), 2)
        assert expr.op == Op.FLOOR_DIV

    def test_mod_factory(self):
        from titanframe.expr.binary_expr import mod
        expr = mod(col("a"), 3)
        assert expr.op == Op.MOD

    def test_pow_factory(self):
        from titanframe.expr.binary_expr import pow_
        expr = pow_(col("a"), 2)
        assert expr.op == Op.POW

    def test_comparison_factories(self):
        from titanframe.expr.binary_expr import eq, ne, lt, le, gt, ge
        assert eq(col("a"), 1).op == Op.EQ
        assert ne(col("a"), 1).op == Op.NE
        assert lt(col("a"), 1).op == Op.LT
        assert le(col("a"), 1).op == Op.LE
        assert gt(col("a"), 1).op == Op.GT
        assert ge(col("a"), 1).op == Op.GE

    def test_logical_factories(self):
        from titanframe.expr.binary_expr import and_, or_, xor
        assert and_(col("a"), col("b")).op == Op.AND
        assert or_(col("a"), col("b")).op == Op.OR
        assert xor(col("a"), col("b")).op == Op.XOR

    def test_auto_wrap_literals(self):
        from titanframe.expr.binary_expr import add
        expr = add(42, col("a"))
        assert isinstance(expr.left, LiteralExpr)
        assert expr.left.value == 42

    # ---- Operator classification ----

    def test_is_arithmetic_op(self):
        from titanframe.expr.binary_expr import is_arithmetic_op
        assert is_arithmetic_op(Op.ADD) is True
        assert is_arithmetic_op(Op.SUB) is True
        assert is_arithmetic_op(Op.MUL) is True
        assert is_arithmetic_op(Op.POW) is True
        assert is_arithmetic_op(Op.EQ) is False
        assert is_arithmetic_op(Op.AND) is False

    def test_is_comparison_op(self):
        from titanframe.expr.binary_expr import is_comparison_op
        assert is_comparison_op(Op.EQ) is True
        assert is_comparison_op(Op.GT) is True
        assert is_comparison_op(Op.ADD) is False
        assert is_comparison_op(Op.AND) is False

    def test_is_logical_op(self):
        from titanframe.expr.binary_expr import is_logical_op
        assert is_logical_op(Op.AND) is True
        assert is_logical_op(Op.OR) is True
        assert is_logical_op(Op.XOR) is True
        assert is_logical_op(Op.ADD) is False
        assert is_logical_op(Op.EQ) is False

    # ---- Type inference ----

    def test_infer_arithmetic_same_type(self):
        from titanframe.expr.binary_expr import infer_binary_dtype
        assert infer_binary_dtype(Op.ADD, Int64, Int64) == Int64

    def test_infer_arithmetic_promotion(self):
        from titanframe.expr.binary_expr import infer_binary_dtype
        assert infer_binary_dtype(Op.ADD, Int32, Float64) == Float64

    def test_infer_true_div_int_produces_float(self):
        from titanframe.expr.binary_expr import infer_binary_dtype
        assert infer_binary_dtype(Op.TRUE_DIV, Int64, Int64) == Float64

    def test_infer_comparison_produces_bool(self):
        from titanframe.expr.binary_expr import infer_binary_dtype
        assert infer_binary_dtype(Op.GT, Int64, Int32) == Bool
        assert infer_binary_dtype(Op.EQ, Float64, Float32) == Bool

    def test_infer_logical_produces_bool(self):
        from titanframe.expr.binary_expr import infer_binary_dtype
        assert infer_binary_dtype(Op.AND, Bool, Bool) == Bool
        assert infer_binary_dtype(Op.OR, Bool, Bool) == Bool


# ===========================================================================
# Unary Expression Module
# ===========================================================================

class TestUnaryExprModule:
    """Tests for titanframe.expr.unary_expr."""

    def test_import(self):
        from titanframe.expr.unary_expr import (
            neg, not_, abs_, ceil, floor, sqrt, log, exp,
            is_null, is_not_null,
            infer_unary_dtype,
            is_null_check_op, is_math_op,
        )

    # ---- Factory functions ----

    def test_neg_factory(self):
        from titanframe.expr.unary_expr import neg
        expr = neg(col("x"))
        assert isinstance(expr, UnaryExpr)
        assert expr.op == UnaryOp.NEG

    def test_not_factory(self):
        from titanframe.expr.unary_expr import not_
        expr = not_(col("flag"))
        assert expr.op == UnaryOp.NOT

    def test_is_null_factory(self):
        from titanframe.expr.unary_expr import is_null
        expr = is_null(col("email"))
        assert expr.op == UnaryOp.IS_NULL

    def test_is_not_null_factory(self):
        from titanframe.expr.unary_expr import is_not_null
        expr = is_not_null(col("email"))
        assert expr.op == UnaryOp.IS_NOT_NULL

    def test_abs_factory(self):
        from titanframe.expr.unary_expr import abs_
        expr = abs_(col("delta"))
        assert expr.op == UnaryOp.ABS

    def test_ceil_factory(self):
        from titanframe.expr.unary_expr import ceil
        expr = ceil(col("price"))
        assert expr.op == UnaryOp.CEIL

    def test_floor_factory(self):
        from titanframe.expr.unary_expr import floor
        expr = floor(col("price"))
        assert expr.op == UnaryOp.FLOOR

    def test_sqrt_factory(self):
        from titanframe.expr.unary_expr import sqrt
        expr = sqrt(col("variance"))
        assert expr.op == UnaryOp.SQRT

    def test_log_factory(self):
        from titanframe.expr.unary_expr import log
        expr = log(col("value"))
        assert expr.op == UnaryOp.LOG

    def test_exp_factory(self):
        from titanframe.expr.unary_expr import exp
        expr = exp(col("log_value"))
        assert expr.op == UnaryOp.EXP

    def test_auto_wrap_literal(self):
        from titanframe.expr.unary_expr import neg
        expr = neg(42)
        assert isinstance(expr.operand, LiteralExpr)

    # ---- Operator classification ----

    def test_is_null_check_op(self):
        from titanframe.expr.unary_expr import is_null_check_op
        assert is_null_check_op(UnaryOp.IS_NULL) is True
        assert is_null_check_op(UnaryOp.IS_NOT_NULL) is True
        assert is_null_check_op(UnaryOp.NEG) is False

    def test_is_math_op(self):
        from titanframe.expr.unary_expr import is_math_op
        assert is_math_op(UnaryOp.SQRT) is True
        assert is_math_op(UnaryOp.LOG) is True
        assert is_math_op(UnaryOp.EXP) is True
        assert is_math_op(UnaryOp.NEG) is False
        assert is_math_op(UnaryOp.IS_NULL) is False

    # ---- Type inference ----

    def test_infer_null_check_produces_bool(self):
        from titanframe.expr.unary_expr import infer_unary_dtype
        assert infer_unary_dtype(UnaryOp.IS_NULL, Int64) == Bool
        assert infer_unary_dtype(UnaryOp.IS_NOT_NULL, Utf8) == Bool

    def test_infer_not_produces_bool(self):
        from titanframe.expr.unary_expr import infer_unary_dtype
        assert infer_unary_dtype(UnaryOp.NOT, Bool) == Bool

    def test_infer_neg_preserves_type(self):
        from titanframe.expr.unary_expr import infer_unary_dtype
        assert infer_unary_dtype(UnaryOp.NEG, Int64) == Int64
        assert infer_unary_dtype(UnaryOp.NEG, Float64) == Float64

    def test_infer_abs_preserves_type(self):
        from titanframe.expr.unary_expr import infer_unary_dtype
        assert infer_unary_dtype(UnaryOp.ABS, Int32) == Int32

    def test_infer_sqrt_produces_float(self):
        from titanframe.expr.unary_expr import infer_unary_dtype
        assert infer_unary_dtype(UnaryOp.SQRT, Int64) == Float64

    def test_infer_log_produces_float(self):
        from titanframe.expr.unary_expr import infer_unary_dtype
        assert infer_unary_dtype(UnaryOp.LOG, Int32) == Float64

    def test_infer_exp_produces_float(self):
        from titanframe.expr.unary_expr import infer_unary_dtype
        assert infer_unary_dtype(UnaryOp.EXP, Float32) == Float64


# ===========================================================================
# Aggregation Expression Module
# ===========================================================================

class TestAggExprModule:
    """Tests for titanframe.expr.agg_expr."""

    def test_import(self):
        from titanframe.expr.agg_expr import (
            sum_, mean, min_, max_, count, count_distinct,
            first, last, std, var, median, quantile,
            any_, all_,
            QuantileExpr, PartialAggResult,
            infer_agg_dtype, partial_agg_state_fields, is_reducible_op,
        )

    # ---- Factory functions ----

    def test_sum_factory(self):
        from titanframe.expr.agg_expr import sum_
        expr = sum_(col("revenue"))
        assert isinstance(expr, AggExpr)
        assert expr.op == AggOp.SUM

    def test_mean_factory(self):
        from titanframe.expr.agg_expr import mean
        expr = mean(col("score"))
        assert expr.op == AggOp.MEAN

    def test_min_factory(self):
        from titanframe.expr.agg_expr import min_
        expr = min_(col("x"))
        assert expr.op == AggOp.MIN

    def test_max_factory(self):
        from titanframe.expr.agg_expr import max_
        expr = max_(col("x"))
        assert expr.op == AggOp.MAX

    def test_count_factory(self):
        from titanframe.expr.agg_expr import count
        expr = count(col("id"))
        assert expr.op == AggOp.COUNT

    def test_count_distinct_factory(self):
        from titanframe.expr.agg_expr import count_distinct
        expr = count_distinct(col("customer_id"))
        assert expr.op == AggOp.COUNT_DISTINCT

    def test_first_factory(self):
        from titanframe.expr.agg_expr import first
        expr = first(col("name"))
        assert expr.op == AggOp.FIRST

    def test_last_factory(self):
        from titanframe.expr.agg_expr import last
        expr = last(col("ts"))
        assert expr.op == AggOp.LAST

    def test_std_factory(self):
        from titanframe.expr.agg_expr import std
        expr = std(col("score"))
        assert expr.op == AggOp.STD

    def test_var_factory(self):
        from titanframe.expr.agg_expr import var
        expr = var(col("score"))
        assert expr.op == AggOp.VAR

    def test_median_factory(self):
        from titanframe.expr.agg_expr import median
        expr = median(col("income"))
        assert expr.op == AggOp.MEDIAN

    def test_any_factory(self):
        from titanframe.expr.agg_expr import any_
        expr = any_(col("flag"))
        assert expr.op == AggOp.ANY

    def test_all_factory(self):
        from titanframe.expr.agg_expr import all_
        expr = all_(col("valid"))
        assert expr.op == AggOp.ALL

    # ---- QuantileExpr ----

    def test_quantile_creates_quantile_expr(self):
        from titanframe.expr.agg_expr import quantile, QuantileExpr
        expr = quantile(col("latency"), q=0.95)
        assert isinstance(expr, QuantileExpr)
        assert expr.q == 0.95
        assert expr.op == AggOp.QUANTILE

    def test_quantile_default_is_median(self):
        from titanframe.expr.agg_expr import quantile
        expr = quantile(col("x"))
        assert expr.q == 0.5

    def test_quantile_invalid_q_raises(self):
        from titanframe.expr.agg_expr import quantile
        with pytest.raises(ValueError, match="Quantile q must be"):
            quantile(col("x"), q=1.5)
        with pytest.raises(ValueError, match="Quantile q must be"):
            quantile(col("x"), q=-0.1)

    def test_quantile_is_agg_subclass(self):
        from titanframe.expr.agg_expr import quantile
        expr = quantile(col("x"), q=0.75)
        assert isinstance(expr, AggExpr)  # Inherits from AggExpr

    def test_quantile_display(self):
        from titanframe.expr.agg_expr import quantile
        expr = quantile(col("x"), q=0.9)
        assert "0.9" in expr.display()

    def test_quantile_hash(self):
        from titanframe.expr.agg_expr import quantile
        a = quantile(col("x"), q=0.5)
        b = quantile(col("x"), q=0.5)
        assert hash(a) == hash(b)

    def test_quantile_different_q_different_hash(self):
        from titanframe.expr.agg_expr import quantile
        a = quantile(col("x"), q=0.5)
        b = quantile(col("x"), q=0.95)
        assert hash(a) != hash(b)

    def test_quantile_with_children(self):
        from titanframe.expr.agg_expr import quantile
        expr = quantile(col("x"), q=0.75)
        new_expr = expr._with_children([col("y")])
        assert isinstance(new_expr, type(expr))
        assert new_expr.q == 0.75

    # ---- Type inference ----

    def test_infer_count_is_int64(self):
        from titanframe.expr.agg_expr import infer_agg_dtype
        assert infer_agg_dtype(AggOp.COUNT, Float64) == Int64
        assert infer_agg_dtype(AggOp.COUNT_DISTINCT, Utf8) == Int64

    def test_infer_any_all_is_bool(self):
        from titanframe.expr.agg_expr import infer_agg_dtype
        assert infer_agg_dtype(AggOp.ANY, Bool) == Bool
        assert infer_agg_dtype(AggOp.ALL, Bool) == Bool

    def test_infer_mean_is_float(self):
        from titanframe.expr.agg_expr import infer_agg_dtype
        assert infer_agg_dtype(AggOp.MEAN, Int64) == Float64
        assert infer_agg_dtype(AggOp.STD, Int32) == Float64
        assert infer_agg_dtype(AggOp.VAR, Float32) == Float64

    def test_infer_sum_int_promotes(self):
        from titanframe.expr.agg_expr import infer_agg_dtype
        assert infer_agg_dtype(AggOp.SUM, Int8) == Int64
        assert infer_agg_dtype(AggOp.SUM, Int32) == Int64

    def test_infer_sum_float_preserves(self):
        from titanframe.expr.agg_expr import infer_agg_dtype
        assert infer_agg_dtype(AggOp.SUM, Float64) == Float64

    def test_infer_min_max_preserves(self):
        from titanframe.expr.agg_expr import infer_agg_dtype
        assert infer_agg_dtype(AggOp.MIN, Int32) == Int32
        assert infer_agg_dtype(AggOp.MAX, Float32) == Float32
        assert infer_agg_dtype(AggOp.FIRST, Utf8) == Utf8

    # ---- Partial aggregation ----

    def test_is_reducible_sum(self):
        from titanframe.expr.agg_expr import is_reducible_op
        assert is_reducible_op(AggOp.SUM) is True
        assert is_reducible_op(AggOp.COUNT) is True
        assert is_reducible_op(AggOp.MIN) is True
        assert is_reducible_op(AggOp.MEAN) is True

    def test_is_not_reducible_median(self):
        from titanframe.expr.agg_expr import is_reducible_op
        assert is_reducible_op(AggOp.MEDIAN) is False
        assert is_reducible_op(AggOp.QUANTILE) is False

    def test_partial_agg_state_fields(self):
        from titanframe.expr.agg_expr import partial_agg_state_fields
        assert "count" in partial_agg_state_fields(AggOp.SUM)
        assert "sum_value" in partial_agg_state_fields(AggOp.MEAN)
        assert "sum_sq_value" in partial_agg_state_fields(AggOp.STD)
        assert "min_value" in partial_agg_state_fields(AggOp.MIN)

    def test_partial_agg_result_defaults(self):
        from titanframe.expr.agg_expr import PartialAggResult
        p = PartialAggResult(op=AggOp.SUM)
        assert p.count == 0
        assert p.sum_value == 0.0
        assert p.min_value is None
        assert p.any_value is False
        assert p.all_value is True


# ===========================================================================
# Cast Expression Module
# ===========================================================================

class TestCastExprModule:
    """Tests for titanframe.expr.cast_expr."""

    def test_import(self):
        from titanframe.expr.cast_expr import (
            CastExpr, TryCastExpr,
            cast, try_cast, validate_cast,
            is_identity_cast, CastError,
            NUMERIC_CAST_PAIRS, TEMPORAL_CAST_PAIRS, STRING_CAST_PAIRS,
            to_int64, to_float64, to_string, to_bool, to_date, to_datetime,
        )

    # ---- Factory functions ----

    def test_cast_factory(self):
        from titanframe.expr.cast_expr import cast
        expr = cast(col("x"), Float64)
        assert isinstance(expr, CastExpr)
        assert expr.target_dtype == Float64

    def test_cast_auto_wraps_literal(self):
        from titanframe.expr.cast_expr import cast
        expr = cast(42, Utf8)
        assert isinstance(expr.child, LiteralExpr)

    def test_try_cast_factory(self):
        from titanframe.expr.cast_expr import try_cast, TryCastExpr
        expr = try_cast(col("input"), Int64)
        assert isinstance(expr, TryCastExpr)
        assert expr.target_dtype == Int64

    # ---- TryCastExpr ----

    def test_try_cast_children(self):
        from titanframe.expr.cast_expr import try_cast
        expr = try_cast(col("x"), Float64)
        children = expr.children()
        assert len(children) == 1
        assert isinstance(children[0], ColumnExpr)

    def test_try_cast_display(self):
        from titanframe.expr.cast_expr import try_cast
        expr = try_cast(col("x"), Int64)
        assert "TryCastExpr" in expr.display()

    def test_try_cast_hash(self):
        from titanframe.expr.cast_expr import try_cast
        a = try_cast(col("x"), Int64)
        b = try_cast(col("x"), Int64)
        assert hash(a) == hash(b)

    def test_try_cast_different_dtype_different_hash(self):
        from titanframe.expr.cast_expr import try_cast
        a = try_cast(col("x"), Int64)
        b = try_cast(col("x"), Float64)
        assert hash(a) != hash(b)

    def test_try_cast_with_children(self):
        from titanframe.expr.cast_expr import try_cast, TryCastExpr
        expr = try_cast(col("x"), Float64)
        new_expr = expr._with_children([col("y")])
        assert isinstance(new_expr, TryCastExpr)
        assert new_expr.target_dtype == Float64

    # ---- Validation ----

    def test_validate_identity_cast(self):
        from titanframe.expr.cast_expr import validate_cast
        validate_cast(Int64, Int64)  # Should not raise

    def test_validate_safe_widening(self):
        from titanframe.expr.cast_expr import validate_cast
        validate_cast(Int32, Int64)  # Should not raise
        validate_cast(Float32, Float64)  # Should not raise

    def test_validate_unsafe_narrowing_raises(self):
        from titanframe.expr.cast_expr import validate_cast, CastError
        with pytest.raises(CastError):
            validate_cast(Float64, Int32, safe=True)

    def test_validate_unsafe_allowed(self):
        from titanframe.expr.cast_expr import validate_cast
        validate_cast(Float64, Int32, safe=False)  # Should not raise

    def test_is_identity_cast(self):
        from titanframe.expr.cast_expr import is_identity_cast
        assert is_identity_cast(Int64, Int64) is True
        assert is_identity_cast(Int64, Float64) is False

    # ---- Shorthand functions ----

    def test_to_int64(self):
        from titanframe.expr.cast_expr import to_int64
        expr = to_int64(col("x"))
        assert isinstance(expr, CastExpr)
        assert expr.target_dtype == Int64

    def test_to_float64(self):
        from titanframe.expr.cast_expr import to_float64
        expr = to_float64(col("x"))
        assert expr.target_dtype == Float64

    def test_to_string(self):
        from titanframe.expr.cast_expr import to_string
        expr = to_string(col("x"))
        assert expr.target_dtype == Utf8

    def test_to_bool(self):
        from titanframe.expr.cast_expr import to_bool
        expr = to_bool(col("x"))
        assert expr.target_dtype == Bool

    # ---- Cast matrices ----

    def test_numeric_cast_pairs_non_empty(self):
        from titanframe.expr.cast_expr import NUMERIC_CAST_PAIRS
        assert len(NUMERIC_CAST_PAIRS) > 0

    def test_temporal_cast_pairs(self):
        from titanframe.expr.cast_expr import TEMPORAL_CAST_PAIRS
        assert (Date, Datetime) in TEMPORAL_CAST_PAIRS

    def test_string_cast_pairs(self):
        from titanframe.expr.cast_expr import STRING_CAST_PAIRS
        assert (Utf8, Int64) in STRING_CAST_PAIRS
        assert (Int64, Utf8) in STRING_CAST_PAIRS


# ===========================================================================
# Expr __init__.py exports
# ===========================================================================

class TestExprModuleExports:
    """Tests that the expr package __init__.py exports everything correctly."""

    def test_base_types_exported(self):
        from titanframe.expr import (
            Expr, BinaryExpr, UnaryExpr, AggExpr, CastExpr,
            AliasExpr, SortExpr,
            Op, UnaryOp, AggOp, SortOrder,
        )

    def test_leaf_types_exported(self):
        from titanframe.expr import col, ColumnExpr, lit, LiteralExpr

    def test_binary_factories_exported(self):
        from titanframe.expr import (
            add, sub, mul, true_div, floor_div, mod, pow_,
            eq, ne, lt, le, gt, ge,
            and_, or_, xor,
        )

    def test_unary_factories_exported(self):
        from titanframe.expr import (
            neg, not_, abs_, ceil, floor, sqrt, log, exp,
            is_null, is_not_null,
        )

    def test_agg_factories_exported(self):
        from titanframe.expr import (
            sum_, mean, min_, max_, count, count_distinct,
            first, last, std, var, median, quantile,
            any_, all_, QuantileExpr,
        )

    def test_cast_factories_exported(self):
        from titanframe.expr import (
            cast, try_cast, TryCastExpr, validate_cast,
            to_int64, to_float64, to_string, to_bool,
        )

    def test_type_inference_exported(self):
        from titanframe.expr import (
            infer_binary_dtype, infer_unary_dtype, infer_agg_dtype,
        )


# ===========================================================================
# DataFrame evaluator integration
# ===========================================================================

class TestEvaluatorIntegration:
    """Tests that the DataFrame evaluator handles new expression types."""

    def test_quantile_evaluation(self):
        import titanframe as tf
        from titanframe.expr.agg_expr import quantile

        df = tf.DataFrame({"x": [10, 20, 30, 40, 50]})
        result = df.select(quantile(col("x"), q=0.5).alias("median"))
        values = result.to_pydict()["median"]
        assert all(v == pytest.approx(30.0) for v in values)

    def test_quantile_95th(self):
        import titanframe as tf
        from titanframe.expr.agg_expr import quantile

        df = tf.DataFrame({"x": list(range(1, 101))})
        result = df.select(quantile(col("x"), q=0.95).alias("p95"))
        values = result.to_pydict()["p95"]
        # 95th percentile of 1-100 should be around 95
        assert values[0] == pytest.approx(95.0, abs=1.0)

    def test_try_cast_valid_values(self):
        import titanframe as tf
        from titanframe.expr.cast_expr import try_cast

        df = tf.DataFrame({"x": [1, 2, 3]})
        result = df.select(try_cast(col("x"), tf.Float64).alias("x_float"))
        values = result.to_pydict()["x_float"]
        assert values == [1.0, 2.0, 3.0]

    def test_cast_int_to_float(self):
        import titanframe as tf

        df = tf.DataFrame({"x": [1, 2, 3]})
        result = df.select(col("x").cast(tf.Float64).alias("x_float"))
        values = result.to_pydict()["x_float"]
        assert values == [1.0, 2.0, 3.0]

    def test_lazy_quantile(self):
        """Test that QuantileExpr works through eager select with alias chaining."""
        import titanframe as tf
        from titanframe.expr.agg_expr import quantile

        df = tf.DataFrame({"x": [10, 20, 30, 40, 50]})
        # Test that quantile works when chained with alias in eager mode
        result = df.select(
            quantile(col("x"), q=0.25).alias("q25"),
            quantile(col("x"), q=0.75).alias("q75"),
        )
        vals = result.to_pydict()
        assert all(v == pytest.approx(20.0) for v in vals["q25"])
        assert all(v == pytest.approx(40.0) for v in vals["q75"])


# ===========================================================================
# Package-level imports
# ===========================================================================

class TestPackageLevelImports:
    """Tests that titanframe package exposes the full Phase 2 API."""

    def test_dataframe_exposed(self):
        import titanframe as tf
        assert hasattr(tf, "DataFrame")

    def test_lazyframe_exposed(self):
        import titanframe as tf
        assert hasattr(tf, "LazyFrame")

    def test_series_exposed(self):
        import titanframe as tf
        assert hasattr(tf, "Series")

    def test_io_functions_exposed(self):
        import titanframe as tf
        assert hasattr(tf, "read_csv")
        assert hasattr(tf, "scan_csv")
        assert hasattr(tf, "read_ipc")
        assert hasattr(tf, "scan_ipc")

    def test_construction_functions_exposed(self):
        import titanframe as tf
        assert hasattr(tf, "from_dict")
        assert hasattr(tf, "from_arrow")
        assert hasattr(tf, "from_pandas")

    def test_combining_functions_exposed(self):
        import titanframe as tf
        assert hasattr(tf, "concat")
        assert hasattr(tf, "merge")

    def test_config_exposed(self):
        import titanframe as tf
        assert hasattr(tf, "config")
        assert hasattr(tf, "TitanFrameConfig")

    def test_type_utilities_exposed(self):
        import titanframe as tf
        assert hasattr(tf, "promote")
        assert hasattr(tf, "can_cast")
