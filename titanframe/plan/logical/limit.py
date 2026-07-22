from __future__ import annotations
from typing import Optional
from titanframe.core.schema import Schema
from titanframe.plan.logical.node import LogicalPlan

class Limit(LogicalPlan):
    __slots__ = ('input', 'limit', 'offset')

    def __init__(self, input: LogicalPlan, limit: Optional[int], offset: int=0):
        self.input = input
        self.limit = limit
        self.offset = offset

    def output_schema(self) -> Schema:
        return self.input.output_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_name(self) -> str:
        return 'Limit'

    def node_description(self) -> str:
        desc = []
        if self.limit is not None:
            desc.append(f'limit={self.limit}')
        if self.offset > 0:
            desc.append(f'offset={self.offset}')
        return ', '.join(desc) if desc else 'all'

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Limit(new_children[0], self.limit, self.offset)
