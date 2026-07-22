from __future__ import annotations
from typing import Any
from titanframe.core.dtypes import DType, Bool, Float64
from titanframe.expr.base import UnaryExpr, Expr, UnaryOp, _wrap
__all__ = ['UnaryExpr', 'UnaryOp', 'neg', 'not_', 'is_null', 'is_not_null', 'abs_', 'ceil', 'floor', 'sqrt', 'log', 'exp', 'infer_unary_dtype', 'is_null_check_op', 'is_math_op']
_NULL_CHECK_OPS = frozenset({UnaryOp.IS_NULL, UnaryOp.IS_NOT_NULL})
_MATH_OPS = frozenset({UnaryOp.ABS, UnaryOp.CEIL, UnaryOp.FLOOR, UnaryOp.SQRT, UnaryOp.LOG, UnaryOp.EXP})
_PRESERVES_DTYPE_OPS = frozenset({UnaryOp.NEG, UnaryOp.ABS, UnaryOp.CEIL, UnaryOp.FLOOR})
_PRODUCES_FLOAT_OPS = frozenset({UnaryOp.SQRT, UnaryOp.LOG, UnaryOp.EXP})

def is_null_check_op(op: UnaryOp) -> bool:
    return op in _NULL_CHECK_OPS

def is_math_op(op: UnaryOp) -> bool:
    return op in _MATH_OPS

def infer_unary_dtype(op: UnaryOp, input_dtype: DType) -> DType:
    if op in _NULL_CHECK_OPS:
        return Bool
    if op == UnaryOp.NOT:
        return Bool
    if op in _PRESERVES_DTYPE_OPS:
        return input_dtype
    if op in _PRODUCES_FLOAT_OPS:
        return Float64
    raise TypeError(f'Cannot infer dtype for unknown unary op: {op}')

def _make_unary(op: UnaryOp, operand: Any) -> UnaryExpr:
    return UnaryExpr(op, _wrap(operand))

def neg(operand: Any) -> UnaryExpr:
    return _make_unary(UnaryOp.NEG, operand)

def not_(operand: Any) -> UnaryExpr:
    return _make_unary(UnaryOp.NOT, operand)

def is_null(operand: Any) -> UnaryExpr:
    return _make_unary(UnaryOp.IS_NULL, operand)

def is_not_null(operand: Any) -> UnaryExpr:
    return _make_unary(UnaryOp.IS_NOT_NULL, operand)

def abs_(operand: Any) -> UnaryExpr:
    return _make_unary(UnaryOp.ABS, operand)

def ceil(operand: Any) -> UnaryExpr:
    return _make_unary(UnaryOp.CEIL, operand)

def floor(operand: Any) -> UnaryExpr:
    return _make_unary(UnaryOp.FLOOR, operand)

def sqrt(operand: Any) -> UnaryExpr:
    return _make_unary(UnaryOp.SQRT, operand)

def log(operand: Any) -> UnaryExpr:
    return _make_unary(UnaryOp.LOG, operand)

def exp(operand: Any) -> UnaryExpr:
    return _make_unary(UnaryOp.EXP, operand)
