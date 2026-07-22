"""
Operator Fusion Rule
====================

Merges adjacent compatible nodes, like Filter -> Filter.
"""

from titanframe.plan.logical.node import LogicalPlan
from titanframe.plan.logical.filter import Filter
from titanframe.plan.optimizer.rule import OptimizationRule
from titanframe.expr.base import BinaryExpr, Op

class OperatorFusion(OptimizationRule):
    @property
    def name(self) -> str:
        return "operator_fusion"

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        plan = plan.map_children(self.apply)
        
        if isinstance(plan, Filter):
            child = plan.input
            if isinstance(child, Filter):
                new_pred = BinaryExpr(Op.AND, child.predicate, plan.predicate)
                return Filter(child.input, new_pred)
                
        return plan
