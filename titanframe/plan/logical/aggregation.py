from __future__ import annotations
from typing import Sequence
from titanframe.core.schema import Schema
from titanframe.expr.base import Expr
from titanframe.plan.logical.node import LogicalPlan
from titanframe.plan.logical.utils import infer_expr_name, infer_expr_dtype

class Aggregation(LogicalPlan):
    __slots__ = ('input', 'group_keys', 'agg_exprs')

    def __init__(self, input: LogicalPlan, group_keys: Sequence[Expr], agg_exprs: Sequence[Expr]):
        self.input = input
        self.group_keys = list(group_keys)
        self.agg_exprs = list(agg_exprs)

    def output_schema(self) -> Schema:
        input_schema = self.input.output_schema()
        fields = []
        for key in self.group_keys:
            name = infer_expr_name(key)
            dtype = infer_expr_dtype(key, input_schema)
            fields.append((name, dtype))
        for agg in self.agg_exprs:
            name = infer_expr_name(agg)
            dtype = infer_expr_dtype(agg, input_schema)
            fields.append((name, dtype))
        return Schema(fields)

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_name(self) -> str:
        return 'Aggregation'

    def node_description(self) -> str:
        keys = [infer_expr_name(k) for k in self.group_keys]
        aggs = [infer_expr_name(a) for a in self.agg_exprs]
        return f"keys=[{', '.join(keys)}], aggs=[{', '.join(aggs)}]"

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Aggregation(new_children[0], self.group_keys, self.agg_exprs)
