"""
Join Reordering
===============
"""

from titanframe.plan.logical.node import LogicalPlan
from titanframe.plan.optimizer.rule import OptimizationRule

class JoinReorder(OptimizationRule):
    @property
    def name(self) -> str:
        return "join_reorder"

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        return plan.map_children(self.apply)
