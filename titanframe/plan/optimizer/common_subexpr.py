"""
Common Subexpression Elimination
================================
"""

from titanframe.plan.logical.node import LogicalPlan
from titanframe.plan.optimizer.rule import OptimizationRule

class CommonSubexprElimination(OptimizationRule):
    @property
    def name(self) -> str:
        return "common_subexpr_elimination"

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        return plan.map_children(self.apply)
