from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from titanframe.core.dtypes import DType, Bool, Float64, Int64
from titanframe.expr.base import AggExpr, Expr, AggOp, _wrap
__all__ = ['AggExpr', 'AggOp', 'QuantileExpr', 'PartialAggResult', 'sum_', 'mean', 'min_', 'max_', 'count', 'count_distinct', 'first', 'last', 'std', 'var', 'median', 'quantile', 'any_', 'all_', 'infer_agg_dtype', 'partial_agg_state_fields', 'is_reducible_op']

class QuantileExpr(AggExpr):
    __slots__ = ('q',)

    def __init__(self, child: Expr, q: float=0.5):
        super().__init__(AggOp.QUANTILE, child)
        if not 0.0 <= q <= 1.0:
            raise ValueError(f'Quantile q must be in [0, 1], got {q}')
        self.q = q

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return QuantileExpr(new_children[0], self.q)

    def display(self, indent: int=0) -> str:
        return f'QuantileExpr({self.child.display()}, q={self.q})'

    def __hash__(self) -> int:
        return hash(('quantile', self.child, self.q))

def infer_agg_dtype(op: AggOp, input_dtype: DType) -> DType:
    if op in (AggOp.COUNT, AggOp.COUNT_DISTINCT):
        return Int64
    if op in (AggOp.ANY, AggOp.ALL):
        return Bool
    if op in (AggOp.MEAN, AggOp.STD, AggOp.VAR, AggOp.MEDIAN, AggOp.QUANTILE):
        return Float64
    if op == AggOp.SUM:
        if input_dtype.is_integer:
            return Int64
        return input_dtype
    if op in (AggOp.MIN, AggOp.MAX, AggOp.FIRST, AggOp.LAST):
        return input_dtype
    raise TypeError(f'Cannot infer dtype for unknown agg op: {op}')

@dataclass
class PartialAggResult:
    op: AggOp
    count: int = 0
    sum_value: float = 0.0
    sum_sq_value: float = 0.0
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    first_value: Optional[Any] = None
    last_value: Optional[Any] = None
    any_value: bool = False
    all_value: bool = True

def partial_agg_state_fields(op: AggOp) -> list[str]:
    _fields_map: dict[AggOp, list[str]] = {AggOp.SUM: ['count', 'sum_value'], AggOp.MEAN: ['count', 'sum_value'], AggOp.STD: ['count', 'sum_value', 'sum_sq_value'], AggOp.VAR: ['count', 'sum_value', 'sum_sq_value'], AggOp.MIN: ['min_value'], AggOp.MAX: ['max_value'], AggOp.COUNT: ['count'], AggOp.COUNT_DISTINCT: ['count'], AggOp.FIRST: ['first_value', 'count'], AggOp.LAST: ['last_value'], AggOp.ANY: ['any_value'], AggOp.ALL: ['all_value'], AggOp.MEDIAN: ['count', 'sum_value'], AggOp.QUANTILE: ['count', 'sum_value']}
    return _fields_map.get(op, ['count', 'sum_value'])

def is_reducible_op(op: AggOp) -> bool:
    _non_reducible = frozenset({AggOp.MEDIAN, AggOp.QUANTILE})
    return op not in _non_reducible

def _make_agg(op: AggOp, operand: Any) -> AggExpr:
    return AggExpr(op, _wrap(operand))

def sum_(operand: Any) -> AggExpr:
    return _make_agg(AggOp.SUM, operand)

def mean(operand: Any) -> AggExpr:
    return _make_agg(AggOp.MEAN, operand)

def min_(operand: Any) -> AggExpr:
    return _make_agg(AggOp.MIN, operand)

def max_(operand: Any) -> AggExpr:
    return _make_agg(AggOp.MAX, operand)

def count(operand: Any) -> AggExpr:
    return _make_agg(AggOp.COUNT, operand)

def count_distinct(operand: Any) -> AggExpr:
    return _make_agg(AggOp.COUNT_DISTINCT, operand)

def first(operand: Any) -> AggExpr:
    return _make_agg(AggOp.FIRST, operand)

def last(operand: Any) -> AggExpr:
    return _make_agg(AggOp.LAST, operand)

def std(operand: Any) -> AggExpr:
    return _make_agg(AggOp.STD, operand)

def var(operand: Any) -> AggExpr:
    return _make_agg(AggOp.VAR, operand)

def median(operand: Any) -> AggExpr:
    return _make_agg(AggOp.MEDIAN, operand)

def quantile(operand: Any, q: float=0.5) -> QuantileExpr:
    return QuantileExpr(_wrap(operand), q)

def any_(operand: Any) -> AggExpr:
    return _make_agg(AggOp.ANY, operand)

def all_(operand: Any) -> AggExpr:
    return _make_agg(AggOp.ALL, operand)
