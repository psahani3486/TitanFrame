from titanframe.plan.logical.node import LogicalPlan
from titanframe.plan.logical.filter import Filter
from titanframe.plan.logical.scan import Scan
from titanframe.plan.logical.projection import Projection
from titanframe.plan.optimizer.rule import OptimizationRule
from titanframe.expr.base import Expr, BinaryExpr, Op

class PredicatePushdown(OptimizationRule):

    @property
    def name(self) -> str:
        return 'predicate_pushdown'

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        plan = plan.map_children(self.apply)
        if isinstance(plan, Filter):
            child = plan.input
            if isinstance(child, Projection):
                is_simple_proj = all((hasattr(e, 'column_name') for e in child.exprs))
                if is_simple_proj:
                    new_filter = Filter(child.input, plan.predicate)
                    pushed = self.apply(new_filter)
                    return Projection(pushed, child.exprs)
            if isinstance(child, Scan):
                new_pred = plan.predicate
                if child.predicate is not None:
                    new_pred = BinaryExpr(Op.AND, child.predicate, plan.predicate)
                return Scan(source=child.source, format=child.format, schema=child._schema, projection=child.projection, predicate=new_pred, limit=child.limit, chunk_size=child.chunk_size)
        return plan
