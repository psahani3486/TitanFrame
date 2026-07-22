from __future__ import annotations
from typing import Any, Optional
from titanframe.core.dtypes import DType, can_cast, Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64, Utf8, Date, Datetime, Duration
from titanframe.expr.base import CastExpr, Expr, _wrap
__all__ = ['CastExpr', 'TryCastExpr', 'cast', 'try_cast', 'validate_cast', 'is_identity_cast', 'NUMERIC_CAST_PAIRS', 'TEMPORAL_CAST_PAIRS', 'STRING_CAST_PAIRS']

class TryCastExpr(Expr):
    __slots__ = ('child', 'target_dtype')

    def __init__(self, child: Expr, target_dtype: DType):
        self.child = child
        self.target_dtype = target_dtype

    def children(self) -> list[Expr]:
        return [self.child]

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return TryCastExpr(new_children[0], self.target_dtype)

    def display(self, indent: int=0) -> str:
        return f'TryCastExpr({self.child.display()}, {self.target_dtype})'

    def __hash__(self) -> int:
        return hash(('try_cast', self.child, self.target_dtype))

class CastError(Exception):
    pass

def validate_cast(source_dtype: DType, target_dtype: DType, safe: bool=True) -> None:
    if source_dtype == target_dtype:
        return
    if not can_cast(source_dtype, target_dtype, safe=safe):
        mode = 'safe' if safe else 'unsafe'
        raise CastError(f"Cannot {mode}-cast {source_dtype} → {target_dtype}. {('Use safe=False or try_cast for potentially lossy conversions.' if safe else '')}")

def is_identity_cast(source_dtype: DType, target_dtype: DType) -> bool:
    return source_dtype == target_dtype
NUMERIC_CAST_PAIRS: list[tuple[DType, DType]] = [(Int8, Int16), (Int8, Int32), (Int8, Int64), (Int16, Int32), (Int16, Int64), (Int32, Int64), (UInt8, UInt16), (UInt8, UInt32), (UInt8, UInt64), (UInt16, UInt32), (UInt16, UInt64), (UInt32, UInt64), (Int8, Float32), (Int16, Float32), (Int8, Float64), (Int16, Float64), (Int32, Float64), (Int64, Float64), (UInt8, Float32), (UInt16, Float32), (UInt8, Float64), (UInt16, Float64), (UInt32, Float64), (UInt64, Float64), (Float32, Float64), (Bool, Int8), (Bool, Int32), (Bool, Int64), (Bool, Float64)]
TEMPORAL_CAST_PAIRS: list[tuple[DType, DType]] = [(Date, Datetime), (Datetime, Date)]
STRING_CAST_PAIRS: list[tuple[DType, DType]] = [(Utf8, Int64), (Utf8, Float64), (Utf8, Bool), (Utf8, Date), (Utf8, Datetime), (Int64, Utf8), (Float64, Utf8), (Bool, Utf8), (Date, Utf8), (Datetime, Utf8)]

def cast(operand: Any, target_dtype: DType, safe: bool=True) -> CastExpr:
    expr = _wrap(operand)
    return CastExpr(expr, target_dtype)

def try_cast(operand: Any, target_dtype: DType) -> TryCastExpr:
    return TryCastExpr(_wrap(operand), target_dtype)

def to_int64(operand: Any) -> CastExpr:
    return cast(operand, Int64)

def to_float64(operand: Any) -> CastExpr:
    return cast(operand, Float64)

def to_string(operand: Any) -> CastExpr:
    return cast(operand, Utf8, safe=False)

def to_bool(operand: Any) -> CastExpr:
    return cast(operand, Bool, safe=False)

def to_date(operand: Any) -> CastExpr:
    return cast(operand, Date, safe=False)

def to_datetime(operand: Any) -> CastExpr:
    return cast(operand, Datetime, safe=False)
