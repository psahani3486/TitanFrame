"""
Logical Plan Utilities
========================

Shared helper functions for schema inference used by logical plan nodes.
"""

from __future__ import annotations

from titanframe.core.dtypes import DType, Float64, Int64, Bool, promote
from titanframe.core.schema import Schema
from titanframe.expr.base import (
    Expr, AliasExpr, AggExpr, CastExpr, BinaryExpr, UnaryExpr,
    Op, UnaryOp, AggOp,
)
from titanframe.expr.column_expr import ColumnExpr
from titanframe.expr.literal_expr import LiteralExpr


def infer_expr_name(expr: Expr) -> str:
    """
    Infer an output column name from an expression.

    Used by Projection and Aggregation nodes to determine column names
    when the user doesn't provide an explicit alias.

    Rules:
        - ``AliasExpr`` → the alias name
        - ``ColumnExpr`` → the column name
        - ``AggExpr`` → ``"{op}_{child_name}"`` (e.g., ``"sum_revenue"``)
        - Fallback → ``repr(expr)``
    """
    if isinstance(expr, AliasExpr):
        return expr.name
    if isinstance(expr, ColumnExpr):
        return expr.column_name
    if isinstance(expr, AggExpr):
        inner_name = infer_expr_name(expr.child)
        return f"{expr.op.value}_{inner_name}"
    # Fallback
    return repr(expr)


def infer_expr_dtype(expr: Expr, input_schema: Schema) -> DType:
    """
    Infer the output dtype of an expression given an input schema.

    This is a simplified inference used during logical planning.
    Full type resolution happens during physical planning.

    Args:
        expr: The expression to infer the type for.
        input_schema: The schema of the input data.

    Returns:
        The inferred output DType.
    """
    from titanframe.core.dtypes import from_value

    if isinstance(expr, AliasExpr):
        return infer_expr_dtype(expr.child, input_schema)
    if isinstance(expr, ColumnExpr):
        return input_schema[expr.column_name]
    if isinstance(expr, LiteralExpr):
        return expr.dtype
    if isinstance(expr, CastExpr):
        return expr.target_dtype
    if isinstance(expr, BinaryExpr):
        left_dt = infer_expr_dtype(expr.left, input_schema)
        right_dt = infer_expr_dtype(expr.right, input_schema)
        if expr.op in (Op.EQ, Op.NE, Op.LT, Op.LE, Op.GT, Op.GE, Op.AND, Op.OR, Op.XOR):
            return Bool
        try:
            return promote(left_dt, right_dt)
        except TypeError:
            return Float64  # Fallback
    if isinstance(expr, UnaryExpr):
        if expr.op in (UnaryOp.IS_NULL, UnaryOp.IS_NOT_NULL, UnaryOp.NOT):
            return Bool
        return infer_expr_dtype(expr.operand, input_schema)
    if isinstance(expr, AggExpr):
        child_dt = infer_expr_dtype(expr.child, input_schema)
        if expr.op in (AggOp.COUNT, AggOp.COUNT_DISTINCT):
            return Int64
        if expr.op in (AggOp.MEAN, AggOp.STD, AggOp.VAR, AggOp.MEDIAN):
            return Float64
        if expr.op in (AggOp.ANY, AggOp.ALL):
            return Bool
        return child_dt  # SUM, MIN, MAX, FIRST, LAST preserve type

    # Check TryCastExpr
    from titanframe.expr.cast_expr import TryCastExpr
    if isinstance(expr, TryCastExpr):
        return expr.target_dtype

    return Float64  # Safe fallback
