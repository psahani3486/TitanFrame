"""
Constant Folding Rule
=====================

Evaluates constant expressions at compile time.
"""

from titanframe.plan.logical.node import LogicalPlan
from titanframe.plan.logical.filter import Filter
from titanframe.plan.logical.projection import Projection
from titanframe.plan.logical.aggregation import Aggregation
from titanframe.plan.optimizer.rule import OptimizationRule
from titanframe.expr.base import Expr, BinaryExpr, UnaryExpr, Op
from titanframe.expr.literal_expr import LiteralExpr

def fold_expr_node(expr: Expr) -> Expr:
    if isinstance(expr, BinaryExpr):
        if isinstance(expr.left, LiteralExpr) and isinstance(expr.right, LiteralExpr):
            v1 = expr.left.value
            v2 = expr.right.value
            if v1 is None or v2 is None:
                return expr
            try:
                if expr.op == Op.ADD: return LiteralExpr(v1 + v2)
                if expr.op == Op.SUB: return LiteralExpr(v1 - v2)
                if expr.op == Op.MUL: return LiteralExpr(v1 * v2)
                if expr.op == Op.TRUE_DIV: return LiteralExpr(v1 / v2)
                if expr.op == Op.FLOOR_DIV: return LiteralExpr(v1 // v2)
                if expr.op == Op.EQ: return LiteralExpr(v1 == v2)
                if expr.op == Op.NE: return LiteralExpr(v1 != v2)
                if expr.op == Op.GT: return LiteralExpr(v1 > v2)
                if expr.op == Op.GE: return LiteralExpr(v1 >= v2)
                if expr.op == Op.LT: return LiteralExpr(v1 < v2)
                if expr.op == Op.LE: return LiteralExpr(v1 <= v2)
                if expr.op == Op.AND: return LiteralExpr(v1 and v2)
                if expr.op == Op.OR: return LiteralExpr(v1 or v2)
            except BaseException:
                pass
    return expr

def fold_expr(expr: Expr) -> Expr:
    # Use transform (bottom-up rewrite) which we implemented in Phase 2
    return expr.transform(fold_expr_node)

class ConstantFolding(OptimizationRule):
    @property
    def name(self) -> str:
        return "constant_folding"

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        plan = plan.map_children(self.apply)
        
        if isinstance(plan, Projection):
            new_exprs = [fold_expr(e) for e in plan.exprs]
            return Projection(plan.input, new_exprs)
            
        if isinstance(plan, Filter):
            new_pred = fold_expr(plan.predicate)
            return Filter(plan.input, new_pred)
            
        if isinstance(plan, Aggregation):
            new_keys = [fold_expr(k) for k in plan.group_keys]
            new_aggs = [fold_expr(a) for a in plan.agg_exprs]
            return Aggregation(plan.input, new_keys, new_aggs)
            
        return plan
