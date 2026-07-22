"""
Logical Plan Node — Base Class
================================

The LogicalPlan is the heart of TitanFrame's query compiler. Every lazy
operation creates a new LogicalPlan node, forming a Directed Acyclic Graph
(DAG) of relational algebra operations.

Key design:
    - **Immutable**: nodes are never mutated; tree-rewriting creates new nodes.
    - **Typed**: every node knows its output Schema.
    - **Visitor-ready**: supports the visitor pattern for optimizer rules.
    - **Displayable**: can pretty-print the full plan tree for debugging.

The plan is purely declarative — it describes *what* to compute.
The PhysicalPlanner decides *how* to compute it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence

from titanframe.core.schema import Schema
from titanframe.expr.base import Expr


class LogicalPlan(ABC):
    """
    Abstract base class for all logical plan nodes.

    Each node represents a relational algebra operation (Scan, Filter,
    Project, Join, Aggregate, etc.) and holds references to its input
    plan(s) as children.
    """

    __slots__ = ()

    @abstractmethod
    def output_schema(self) -> Schema:
        """
        Compute the output schema of this plan node.

        This is called recursively: each node derives its schema from its
        children's schemas and the operation it performs.
        """
        ...

    @abstractmethod
    def children(self) -> list[LogicalPlan]:
        """Return the input plan node(s)."""
        ...

    @abstractmethod
    def node_name(self) -> str:
        """Human-readable name of this node type (e.g., 'Filter', 'Scan')."""
        ...

    @abstractmethod
    def node_description(self) -> str:
        """
        Short description of this node's configuration.

        For Filter: the predicate expression.
        For Scan: the file path.
        For Projection: the column list.
        """
        ...

    @abstractmethod
    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        """
        Return a copy of this node with replaced children.

        Used by optimizer rules for tree rewriting.
        """
        ...


    def accept(self, visitor: PlanVisitor) -> Any:
        """
        Accept a visitor for plan traversal.

        The visitor's ``visit_*`` method is called based on the node type.
        Falls back to ``visit_generic`` if no specific method exists.
        """
        method_name = f"visit_{self.node_name().lower()}"
        visit_fn = getattr(visitor, method_name, None)
        if visit_fn is not None:
            return visit_fn(self)
        return visitor.visit_generic(self)

    def map_children(self, fn: Callable[[LogicalPlan], LogicalPlan]) -> LogicalPlan:
        """
        Apply ``fn`` to each child and return a new node with the results.

        This is the fundamental operation for bottom-up tree rewriting.
        """
        new_children = [fn(child) for child in self.children()]
        return self.with_children(new_children)

    def walk(self) -> list[LogicalPlan]:
        """Return all nodes in pre-order DFS."""
        result: list[LogicalPlan] = [self]
        for child in self.children():
            result.extend(child.walk())
        return result

    def walk_bottom_up(self) -> list[LogicalPlan]:
        """Return all nodes in post-order DFS (children before parents)."""
        result: list[LogicalPlan] = []
        for child in self.children():
            result.extend(child.walk_bottom_up())
        result.append(self)
        return result

    def depth(self) -> int:
        """Maximum depth of the plan tree."""
        if not self.children():
            return 1
        return 1 + max(child.depth() for child in self.children())

    def node_count(self) -> int:
        """Total number of nodes in the plan tree."""
        return len(self.walk())


    def display(self, indent: int = 0) -> str:
        """Pretty-print the plan tree."""
        prefix = "  " * indent
        lines = [f"{prefix}{self.node_name()}: {self.node_description()}"]
        lines.append(f"{prefix}  -> schema: {self.output_schema()}")
        for child in self.children():
            lines.append(child.display(indent + 1))
        return "\n".join(lines)

    def explain(self) -> str:
        """
        Return a human-readable explanation of the full plan.

        This is what the user sees when they call ``lazy_frame.explain()``.
        """
        return self.display()

    def __repr__(self) -> str:
        return f"{self.node_name()}({self.node_description()})"


class PlanVisitor(ABC):
    """
    Visitor pattern for plan traversal.

    Override ``visit_scan``, ``visit_filter``, etc. for specific node types.
    The ``visit_generic`` method is the fallback.
    """

    def visit_generic(self, node: LogicalPlan) -> Any:
        """Default handler for unhandled node types."""
        results = []
        for child in node.children():
            results.append(node.accept(self))
        return results
