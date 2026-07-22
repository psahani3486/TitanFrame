from __future__ import annotations
from enum import Enum
from typing import Optional, Sequence
from titanframe.core.schema import Schema
from titanframe.plan.logical.node import LogicalPlan

class JoinType(Enum):
    INNER = 'inner'
    LEFT = 'left'
    RIGHT = 'right'
    OUTER = 'outer'
    CROSS = 'cross'
    SEMI = 'semi'
    ANTI = 'anti'

class Join(LogicalPlan):
    __slots__ = ('left', 'right', 'on', 'how', 'suffix')

    def __init__(self, left: LogicalPlan, right: LogicalPlan, on: Sequence[str], how: str='inner', suffix: str='_right'):
        self.left = left
        self.right = right
        self.on = list(on)
        self.how = how
        self.suffix = suffix

    def output_schema(self) -> Schema:
        left_schema = self.left.output_schema()
        right_schema = self.right.output_schema().drop(self.on)
        return left_schema.merge(right_schema, suffix=self.suffix)

    def children(self) -> list[LogicalPlan]:
        return [self.left, self.right]

    def node_name(self) -> str:
        return 'Join'

    def node_description(self) -> str:
        return f'on={self.on}, how={self.how!r}'

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Join(new_children[0], new_children[1], self.on, self.how, self.suffix)

    def swap_sides(self) -> Join:
        swap_how = {'inner': 'inner', 'left': 'right', 'right': 'left', 'outer': 'outer', 'cross': 'cross'}
        return Join(self.right, self.left, self.on, swap_how.get(self.how, self.how), self.suffix)
