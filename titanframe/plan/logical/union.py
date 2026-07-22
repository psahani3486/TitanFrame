"""
Union Node — Vertical Concatenation
===================================
"""

from __future__ import annotations

from titanframe.core.schema import Schema
from titanframe.plan.logical.node import LogicalPlan


class Union(LogicalPlan):
    """
    Vertically concatenate two or more datasets.
    
    All inputs must have compatible schemas. The output schema is the
    schema of the first input.
    """
    __slots__ = ("inputs",)

    def __init__(self, inputs: list[LogicalPlan]):
        if len(inputs) < 2:
            raise ValueError("Union requires at least two inputs.")
        self.inputs = inputs
        # Ensure all inputs have the same schema structure
        base_schema = self.inputs[0].output_schema()
        for i, plan in enumerate(self.inputs[1:], 1):
            base_schema.assert_compatible(plan.output_schema())

    def output_schema(self) -> Schema:
        return self.inputs[0].output_schema()

    def children(self) -> list[LogicalPlan]:
        return self.inputs

    def node_name(self) -> str:
        return "Union"

    def node_description(self) -> str:
        return f"inputs={len(self.inputs)}"

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Union(new_children)
