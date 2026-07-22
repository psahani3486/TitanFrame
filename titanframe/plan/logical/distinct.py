"""
Distinct Node — Unique Rows
===========================
"""

from __future__ import annotations

from typing import Optional, Sequence

from titanframe.core.schema import Schema
from titanframe.plan.logical.node import LogicalPlan


class Distinct(LogicalPlan):
    """
    Remove duplicate rows.
    
    If subset is provided, distinct is computed using only those columns.
    """
    __slots__ = ("input", "subset", "keep")

    def __init__(self, input: LogicalPlan, subset: Optional[Sequence[str]] = None, keep: str = "first"):
        self.input = input
        self.subset = list(subset) if subset else None
        self.keep = keep

    def output_schema(self) -> Schema:
        return self.input.output_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_name(self) -> str:
        return "Distinct"

    def node_description(self) -> str:
        desc = f"keep={self.keep}"
        if self.subset:
            desc += f", subset={self.subset}"
        return desc

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Distinct(new_children[0], self.subset, self.keep)
