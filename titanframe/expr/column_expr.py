from __future__ import annotations
from titanframe.expr.base import Expr

class ColumnExpr(Expr):
    __slots__ = ('column_name',)

    def __init__(self, column_name: str):
        self.column_name = column_name

    def children(self) -> list[Expr]:
        return []

    def _with_children(self, new_children: list[Expr]) -> Expr:
        return ColumnExpr(self.column_name)

    def _collect_columns(self, out: set[str]) -> None:
        out.add(self.column_name)

    def display(self, indent: int=0) -> str:
        return f'col({self.column_name!r})'

    def __hash__(self) -> int:
        return hash(('column', self.column_name))

def col(name: str) -> ColumnExpr:
    return ColumnExpr(name)
