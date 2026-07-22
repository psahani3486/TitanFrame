"""
Expression DSL for building computation graphs.
=================================================

This package provides the expression tree system that captures user intent
without executing it. Every operation builds a symbolic tree node — the
compiler's intermediate representation (IR).

Submodules:
    - :mod:`base` — ``Expr`` ABC, operator enums, core node classes
    - :mod:`column_expr` — ``col("name")`` column references
    - :mod:`literal_expr` — ``lit(42)`` constant values
    - :mod:`binary_expr` — Arithmetic, comparison, and logical ops
    - :mod:`unary_expr` — Negation, math functions, null checks
    - :mod:`agg_expr` — Aggregation operations with partial agg support
    - :mod:`cast_expr` — Type casting with safe/unsafe modes
    - :mod:`string_expr` — String accessor (``.str``)
    - :mod:`datetime_expr` — Datetime accessor (``.dt``)
    - :mod:`window_expr` — Window functions (``.over()``)
    - :mod:`udf_expr` — User-defined functions
"""

from titanframe.expr.base import (
    Expr,
    BinaryExpr,
    UnaryExpr,
    AggExpr,
    CastExpr,
    AliasExpr,
    SortExpr,
    Op,
    UnaryOp,
    AggOp,
    SortOrder,
)

from titanframe.expr.column_expr import col, ColumnExpr
from titanframe.expr.literal_expr import lit, LiteralExpr

from titanframe.expr.binary_expr import (
    add, sub, mul, true_div, floor_div, mod, pow_,
    eq, ne, lt, le, gt, ge,
    and_, or_, xor,
    infer_binary_dtype,
)
from titanframe.expr.unary_expr import (
    neg, not_, abs_, ceil, floor, sqrt, log, exp,
    is_null, is_not_null,
    infer_unary_dtype,
)
from titanframe.expr.agg_expr import (
    sum_, mean, min_, max_, count, count_distinct,
    first, last, std, var, median, quantile,
    any_, all_,
    QuantileExpr,
    infer_agg_dtype,
)
from titanframe.expr.cast_expr import (
    cast, try_cast,
    TryCastExpr,
    validate_cast,
    to_int64, to_float64, to_string, to_bool,
)

__all__ = [
    "Expr", "BinaryExpr", "UnaryExpr", "AggExpr", "CastExpr",
    "AliasExpr", "SortExpr",
    "Op", "UnaryOp", "AggOp", "SortOrder",
    "col", "ColumnExpr", "lit", "LiteralExpr",
    "add", "sub", "mul", "true_div", "floor_div", "mod", "pow_",
    "eq", "ne", "lt", "le", "gt", "ge",
    "and_", "or_", "xor",
    "neg", "not_", "abs_", "ceil", "floor", "sqrt", "log", "exp",
    "is_null", "is_not_null",
    "sum_", "mean", "min_", "max_", "count", "count_distinct",
    "first", "last", "std", "var", "median", "quantile",
    "any_", "all_", "QuantileExpr",
    "cast", "try_cast", "TryCastExpr", "validate_cast",
    "to_int64", "to_float64", "to_string", "to_bool",
    "infer_binary_dtype", "infer_unary_dtype", "infer_agg_dtype",
]
