from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence
from titanframe.core.schema import Schema
from titanframe.expr.base import Expr

class LogicalPlan(ABC):
    __slots__ = ()

    @abstractmethod
    def output_schema(self) -> Schema:
        ...

    @abstractmethod
    def children(self) -> list[LogicalPlan]:
        ...

    @abstractmethod
    def node_name(self) -> str:
        ...

    @abstractmethod
    def node_description(self) -> str:
        ...

    @abstractmethod
    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        ...

    def accept(self, visitor: PlanVisitor) -> Any:
        method_name = f'visit_{self.node_name().lower()}'
        visit_fn = getattr(visitor, method_name, None)
        if visit_fn is not None:
            return visit_fn(self)
        return visitor.visit_generic(self)

    def map_children(self, fn: Callable[[LogicalPlan], LogicalPlan]) -> LogicalPlan:
        new_children = [fn(child) for child in self.children()]
        return self.with_children(new_children)

    def walk(self) -> list[LogicalPlan]:
        result: list[LogicalPlan] = [self]
        for child in self.children():
            result.extend(child.walk())
        return result

    def walk_bottom_up(self) -> list[LogicalPlan]:
        result: list[LogicalPlan] = []
        for child in self.children():
            result.extend(child.walk_bottom_up())
        result.append(self)
        return result

    def depth(self) -> int:
        if not self.children():
            return 1
        return 1 + max((child.depth() for child in self.children()))

    def node_count(self) -> int:
        return len(self.walk())

    def display(self, indent: int=0) -> str:
        prefix = '  ' * indent
        lines = [f'{prefix}{self.node_name()}: {self.node_description()}']
        lines.append(f'{prefix}  -> schema: {self.output_schema()}')
        for child in self.children():
            lines.append(child.display(indent + 1))
        return '\n'.join(lines)

    def explain(self) -> str:
        return self.display()

    def __repr__(self) -> str:
        return f'{self.node_name()}({self.node_description()})'

class PlanVisitor(ABC):

    def visit_generic(self, node: LogicalPlan) -> Any:
        results = []
        for child in node.children():
            results.append(node.accept(self))
        return results
