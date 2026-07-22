from __future__ import annotations
from typing import Any
from titanframe.core.dtypes import DType, from_value
from titanframe.expr.base import Expr

class LiteralExpr(Expr):
    __slots__ = ('value', 'dtype')

    def __init__(self, value: Any, dtype: DType | None=None):
        self.value = value
        self.dtype = dtype if dtype is not None else from_value(value)

    def children(self) -> list[Expr]:
        return []

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return LiteralExpr(self.value, self.dtype)

    def display(self, indent: int=0) -> str:
        return f'lit({self.value!r})'

    def __hash__(self) -> int:
        return hash(('literal', self.value, self.dtype))

def lit(value: Any, dtype: DType | None=None) -> LiteralExpr:
    return LiteralExpr(value, dtype)
