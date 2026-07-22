from __future__ import annotations
from typing import Sequence
from titanframe.core.dtypes import Float64, Int64, Utf8, Bool
from titanframe.core.schema import Schema
from titanframe.expr.base import Expr, AliasExpr, AggExpr, CastExpr, BinaryExpr, UnaryExpr
from titanframe.expr.column_expr import ColumnExpr
from titanframe.expr.literal_expr import LiteralExpr
from titanframe.plan.logical.node import LogicalPlan

def _infer_expr_name(expr: Expr) -> str:
    if isinstance(expr, AliasExpr):
        return expr.name
    if isinstance(expr, ColumnExpr):
        return expr.column_name
    if isinstance(expr, AggExpr):
        inner_name = _infer_expr_name(expr.child)
        return f'{expr.op.value}_{inner_name}'
    return repr(expr)

def _infer_expr_dtype(expr: Expr, input_schema: Schema):
    from titanframe.core.dtypes import promote, from_value
    if isinstance(expr, AliasExpr):
        return _infer_expr_dtype(expr.child, input_schema)
    if isinstance(expr, ColumnExpr):
        return input_schema[expr.column_name]
    if isinstance(expr, LiteralExpr):
        return expr.dtype
    if isinstance(expr, CastExpr):
        return expr.target_dtype
    if isinstance(expr, BinaryExpr):
        left_dt = _infer_expr_dtype(expr.left, input_schema)
        right_dt = _infer_expr_dtype(expr.right, input_schema)
        from titanframe.expr.base import Op
        if expr.op in (Op.EQ, Op.NE, Op.LT, Op.LE, Op.GT, Op.GE, Op.AND, Op.OR, Op.XOR):
            return Bool
        try:
            return promote(left_dt, right_dt)
        except TypeError:
            return Float64
    if isinstance(expr, UnaryExpr):
        from titanframe.expr.base import UnaryOp
        if expr.op in (UnaryOp.IS_NULL, UnaryOp.IS_NOT_NULL, UnaryOp.NOT):
            return Bool
        return _infer_expr_dtype(expr.operand, input_schema)
    if isinstance(expr, AggExpr):
        from titanframe.expr.base import AggOp
        child_dt = _infer_expr_dtype(expr.child, input_schema)
        if expr.op in (AggOp.COUNT, AggOp.COUNT_DISTINCT):
            return Int64
        if expr.op in (AggOp.MEAN, AggOp.STD, AggOp.VAR, AggOp.MEDIAN):
            return Float64
        if expr.op in (AggOp.ANY, AggOp.ALL):
            return Bool
        return child_dt
    return Float64

class Projection(LogicalPlan):
    __slots__ = ('input', 'exprs')

    def __init__(self, input: LogicalPlan, exprs: Sequence[Expr]):
        self.input = input
        self.exprs = list(exprs)

    def output_schema(self) -> Schema:
        input_schema = self.input.output_schema()
        fields = []
        for expr in self.exprs:
            name = _infer_expr_name(expr)
            dtype = _infer_expr_dtype(expr, input_schema)
            fields.append((name, dtype))
        return Schema(fields)

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_name(self) -> str:
        return 'Projection'

    def node_description(self) -> str:
        expr_strs = [_infer_expr_name(e) for e in self.exprs]
        return f"columns=[{', '.join(expr_strs)}]"

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Projection(new_children[0], self.exprs)

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
            name = _infer_expr_name(key)
            dtype = _infer_expr_dtype(key, input_schema)
            fields.append((name, dtype))
        for agg in self.agg_exprs:
            name = _infer_expr_name(agg)
            dtype = _infer_expr_dtype(agg, input_schema)
            fields.append((name, dtype))
        return Schema(fields)

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_name(self) -> str:
        return 'Aggregation'

    def node_description(self) -> str:
        keys = [_infer_expr_name(k) for k in self.group_keys]
        aggs = [_infer_expr_name(a) for a in self.agg_exprs]
        return f"keys=[{', '.join(keys)}], aggs=[{', '.join(aggs)}]"

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Aggregation(new_children[0], self.group_keys, self.agg_exprs)

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

class Limit(LogicalPlan):
    __slots__ = ('input', 'n', 'offset')

    def __init__(self, input: LogicalPlan, n: int, offset: int=0):
        self.input = input
        self.n = n
        self.offset = offset

    def output_schema(self) -> Schema:
        return self.input.output_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_name(self) -> str:
        return 'Limit'

    def node_description(self) -> str:
        if self.offset > 0:
            return f'n={self.n}, offset={self.offset}'
        return f'n={self.n}'

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Limit(new_children[0], self.n, self.offset)

class Distinct(LogicalPlan):
    __slots__ = ('input', 'subset')

    def __init__(self, input: LogicalPlan, subset: Sequence[str] | None=None):
        self.input = input
        self.subset = list(subset) if subset else None

    def output_schema(self) -> Schema:
        return self.input.output_schema()

    def children(self) -> list[LogicalPlan]:
        return [self.input]

    def node_name(self) -> str:
        return 'Distinct'

    def node_description(self) -> str:
        if self.subset:
            return f'subset={self.subset}'
        return 'all columns'

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Distinct(new_children[0], self.subset)

class Union(LogicalPlan):
    __slots__ = ('inputs',)

    def __init__(self, inputs: Sequence[LogicalPlan]):
        self.inputs = list(inputs)
        if len(self.inputs) < 2:
            raise ValueError('Union requires at least 2 inputs')

    def output_schema(self) -> Schema:
        return self.inputs[0].output_schema()

    def children(self) -> list[LogicalPlan]:
        return list(self.inputs)

    def node_name(self) -> str:
        return 'Union'

    def node_description(self) -> str:
        return f'inputs={len(self.inputs)}'

    def with_children(self, new_children: list[LogicalPlan]) -> LogicalPlan:
        return Union(new_children)
