"""
Tests for TitanFrame Expressions (expr/)
=========================================

Tests covering:
    - Expression tree construction via operator overloads
    - Column reference and literal expressions
    - Binary, unary, aggregation, cast, alias expressions
    - Required columns collection (for projection pushdown)
    - Tree walking and mapping
    - Expression hashing (for CSE)
"""

import pytest

from titanframe.core.dtypes import Int64, Float64, Utf8
from titanframe.expr.base import (
    Expr, BinaryExpr, UnaryExpr, AggExpr, CastExpr, AliasExpr, SortExpr,
    Op, UnaryOp, AggOp, SortOrder,
)
from titanframe.expr.column_expr import col, ColumnExpr
from titanframe.expr.literal_expr import lit, LiteralExpr



class TestColumnExpr:

    def test_creates_column_ref(self):
        expr = col("revenue")
        assert isinstance(expr, ColumnExpr)
        assert expr.column_name == "revenue"

    def test_no_children(self):
        expr = col("x")
        assert expr.children() == []

    def test_required_columns(self):
        expr = col("x")
        assert expr.required_columns() == {"x"}

    def test_display(self):
        expr = col("price")
        assert "price" in expr.display()

    def test_hash(self):
        a = col("x")
        b = col("x")
        assert hash(a) == hash(b)

    def test_hash_different(self):
        a = col("x")
        b = col("y")
        assert hash(a) != hash(b)



class TestLiteralExpr:

    def test_int_literal(self):
        expr = lit(42)
        assert isinstance(expr, LiteralExpr)
        assert expr.value == 42
        assert expr.dtype == Int64

    def test_float_literal(self):
        expr = lit(3.14)
        assert expr.dtype == Float64

    def test_string_literal(self):
        expr = lit("hello")
        assert expr.dtype == Utf8

    def test_none_literal(self):
        from titanframe.core.dtypes import Null
        expr = lit(None)
        assert expr.dtype == Null

    def test_explicit_dtype(self):
        expr = lit(42, Float64)
        assert expr.dtype == Float64

    def test_no_children(self):
        expr = lit(1)
        assert expr.children() == []

    def test_required_columns_empty(self):
        expr = lit(42)
        assert expr.required_columns() == set()



class TestBinaryExprFromOperators:

    def test_add(self):
        expr = col("a") + col("b")
        assert isinstance(expr, BinaryExpr)
        assert expr.op == Op.ADD

    def test_add_literal(self):
        expr = col("a") + 10
        assert isinstance(expr, BinaryExpr)
        assert isinstance(expr.right, LiteralExpr)
        assert expr.right.value == 10

    def test_radd(self):
        expr = 10 + col("a")
        assert isinstance(expr, BinaryExpr)
        assert isinstance(expr.left, LiteralExpr)
        assert expr.left.value == 10

    def test_sub(self):
        expr = col("a") - col("b")
        assert expr.op == Op.SUB

    def test_mul(self):
        expr = col("a") * 2
        assert expr.op == Op.MUL

    def test_truediv(self):
        expr = col("a") / col("b")
        assert expr.op == Op.TRUE_DIV

    def test_floordiv(self):
        expr = col("a") // 2
        assert expr.op == Op.FLOOR_DIV

    def test_mod(self):
        expr = col("a") % 3
        assert expr.op == Op.MOD

    def test_pow(self):
        expr = col("a") ** 2
        assert expr.op == Op.POW

    def test_eq(self):
        expr = col("a") == col("b")
        assert expr.op == Op.EQ

    def test_ne(self):
        expr = col("a") != 5
        assert expr.op == Op.NE

    def test_lt(self):
        expr = col("a") < 100
        assert expr.op == Op.LT

    def test_le(self):
        expr = col("a") <= 100
        assert expr.op == Op.LE

    def test_gt(self):
        expr = col("a") > 0
        assert expr.op == Op.GT

    def test_ge(self):
        expr = col("a") >= 0
        assert expr.op == Op.GE

    def test_and(self):
        expr = (col("a") > 0) & (col("b") < 10)
        assert expr.op == Op.AND

    def test_or(self):
        expr = (col("a") > 0) | (col("b") < 10)
        assert expr.op == Op.OR

    def test_xor(self):
        expr = col("a") ^ col("b")
        assert expr.op == Op.XOR



class TestUnaryExpr:

    def test_neg(self):
        expr = -col("x")
        assert isinstance(expr, UnaryExpr)
        assert expr.op == UnaryOp.NEG

    def test_not(self):
        expr = ~col("flag")
        assert expr.op == UnaryOp.NOT

    def test_abs(self):
        expr = col("x").abs()
        assert expr.op == UnaryOp.ABS

    def test_is_null(self):
        expr = col("x").is_null()
        assert expr.op == UnaryOp.IS_NULL

    def test_is_not_null(self):
        expr = col("x").is_not_null()
        assert expr.op == UnaryOp.IS_NOT_NULL

    def test_sqrt(self):
        expr = col("x").sqrt()
        assert expr.op == UnaryOp.SQRT

    def test_log(self):
        expr = col("x").log()
        assert expr.op == UnaryOp.LOG

    def test_exp(self):
        expr = col("x").exp()
        assert expr.op == UnaryOp.EXP



class TestAggExpr:

    def test_sum(self):
        expr = col("revenue").sum()
        assert isinstance(expr, AggExpr)
        assert expr.op == AggOp.SUM

    def test_mean(self):
        expr = col("x").mean()
        assert expr.op == AggOp.MEAN

    def test_min(self):
        expr = col("x").min()
        assert expr.op == AggOp.MIN

    def test_max(self):
        expr = col("x").max()
        assert expr.op == AggOp.MAX

    def test_count(self):
        expr = col("x").count()
        assert expr.op == AggOp.COUNT

    def test_std(self):
        expr = col("x").std()
        assert expr.op == AggOp.STD

    def test_var(self):
        expr = col("x").var()
        assert expr.op == AggOp.VAR

    def test_median(self):
        expr = col("x").median()
        assert expr.op == AggOp.MEDIAN



class TestCastExpr:

    def test_cast(self):
        expr = col("x").cast(Float64)
        assert isinstance(expr, CastExpr)
        assert expr.target_dtype == Float64

    def test_cast_children(self):
        expr = col("x").cast(Float64)
        children = expr.children()
        assert len(children) == 1
        assert isinstance(children[0], ColumnExpr)


class TestAliasExpr:

    def test_alias(self):
        expr = col("x").sum().alias("total")
        assert isinstance(expr, AliasExpr)
        assert expr.name == "total"

    def test_alias_preserves_child(self):
        inner = col("x").sum()
        expr = inner.alias("total")
        assert expr.child is inner



class TestSortExpr:

    def test_asc(self):
        expr = col("x").asc()
        assert isinstance(expr, SortExpr)
        assert expr.order == SortOrder.ASC

    def test_desc(self):
        expr = col("x").desc()
        assert expr.order == SortOrder.DESC



class TestRequiredColumns:

    def test_single_column(self):
        assert col("x").required_columns() == {"x"}

    def test_binary_expr(self):
        expr = col("a") + col("b")
        assert expr.required_columns() == {"a", "b"}

    def test_complex_expr(self):
        expr = ((col("a") * col("b")) + lit(1)) > col("c")
        assert expr.required_columns() == {"a", "b", "c"}

    def test_agg_expr(self):
        expr = col("revenue").sum()
        assert expr.required_columns() == {"revenue"}

    def test_nested_agg(self):
        expr = (col("a") + col("b")).sum().alias("total")
        assert expr.required_columns() == {"a", "b"}



class TestTreeWalking:

    def test_walk_simple(self):
        expr = col("a") + col("b")
        nodes = expr.walk()
        assert len(nodes) == 3

    def test_walk_complex(self):
        expr = (col("a") + lit(1)).sum().alias("total")
        nodes = expr.walk()
        types = [type(n).__name__ for n in nodes]
        assert "AliasExpr" in types
        assert "AggExpr" in types
        assert "BinaryExpr" in types
        assert "ColumnExpr" in types
        assert "LiteralExpr" in types

    def test_map_identity(self):
        expr = col("a") + col("b")
        mapped = expr.transform(lambda x: x)
        assert isinstance(mapped, BinaryExpr)

    def test_with_children(self):
        expr = col("a") + col("b")
        new_expr = expr._with_children([col("x"), col("y")])
        assert isinstance(new_expr, BinaryExpr)
        assert new_expr.left.column_name == "x"
        assert new_expr.right.column_name == "y"



class TestExprHashing:

    def test_same_expr_same_hash(self):
        a = col("x") + lit(1)
        b = col("x") + lit(1)
        assert hash(a) == hash(b)

    def test_different_expr_different_hash(self):
        a = col("x") + lit(1)
        b = col("x") + lit(2)
        assert hash(a) != hash(b)

    def test_aggregate_hashing(self):
        a = col("x").sum()
        b = col("x").sum()
        assert hash(a) == hash(b)

    def test_alias_changes_hash(self):
        a = col("x").sum().alias("a")
        b = col("x").sum().alias("b")
        assert hash(a) != hash(b)
