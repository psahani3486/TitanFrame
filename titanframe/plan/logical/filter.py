from __future__ import annotations
from titanframe.core.schema import Schema
from titanframe.expr.base import Expr
from titanframe.plan.logical.node import LogicalPlan

class Filter(LogicalPlan):
    __slots__ = ('input', 'predicate')

    def __init__(self, input: LogicalPlan, predicate: Expr):
        self.input = input
        self.predicate = predicate

    def output_schema(self) -> Schema:
        return self.input.output_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_name(self) -> str:
        return 'Filter'

    def node_description(self) -> str:
        return f'predicate={self.predicate}'

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Filter(new_children[0], self.predicate)

    def with_predicate(self, predicate: Expr) -> Filter:
        return Filter(self.input, predicate)
