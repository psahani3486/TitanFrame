from __future__ import annotations
from typing import Sequence
from titanframe.core.schema import Schema
from titanframe.expr.base import Expr
from titanframe.plan.logical.node import LogicalPlan

class Sort(LogicalPlan):
    __slots__ = ('input', 'sort_exprs')

    def __init__(self, input: LogicalPlan, sort_exprs: Sequence[Expr]):
        self.input = input
        self.sort_exprs = list(sort_exprs)

    def output_schema(self) -> Schema:
        return self.input.output_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_name(self) -> str:
        return 'Sort'

    def node_description(self) -> str:
        return f"by=[{', '.join((repr(e) for e in self.sort_exprs))}]"

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Sort(new_children[0], self.sort_exprs)
