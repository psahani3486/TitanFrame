"""
Filter Node — WHERE predicate
================================

Applies a boolean predicate to filter rows. The output schema is
identical to the input schema (only rows change, not columns).

Example::

    >>> plan = Filter(scan_node, col("age") > 25)
"""

from __future__ import annotations

from titanframe.core.schema import Schema
from titanframe.expr.base import Expr
from titanframe.plan.logical.node import LogicalPlan


class Filter(LogicalPlan):
    """
    Row filtering — ``WHERE predicate``.

    The predicate must evaluate to a boolean. The output schema is
    identical to the input schema (only rows change).

    Attributes:
        input: The child plan node.
        predicate: Boolean expression for filtering.
    """

    __slots__ = ("input", "predicate")

    def __init__(self, input: LogicalPlan, predicate: Expr):
        self.input = input
        self.predicate = predicate

    def output_schema(self) -> Schema:
        return self.input.output_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_name(self) -> str:
        return "Filter"

    def node_description(self) -> str:
        return f"predicate={self.predicate}"

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Filter(new_children[0], self.predicate)

    def with_predicate(self, predicate: Expr) -> Filter:
        """Return a new Filter with a different predicate."""
        return Filter(self.input, predicate)
