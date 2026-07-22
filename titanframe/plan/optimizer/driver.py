"""
Query Optimizer Driver
======================
"""

from typing import List

from titanframe.plan.logical.node import LogicalPlan
from titanframe.plan.optimizer.rule import OptimizationRule


class QueryOptimizer:
    """
    Applies a series of optimization rules to a logical plan.
    """

    def __init__(self, rules: List[OptimizationRule] = None, max_passes: int = 10):
        if rules is None:
            from titanframe.plan.optimizer.predicate_pushdown import PredicatePushdown
            from titanframe.plan.optimizer.projection_pushdown import ProjectionPushdown
            from titanframe.plan.optimizer.constant_folding import ConstantFolding
            from titanframe.plan.optimizer.slice_pushdown import SlicePushdown
            from titanframe.plan.optimizer.common_subexpr import CommonSubexprElimination
            from titanframe.plan.optimizer.fusion import OperatorFusion
            rules = [
                PredicatePushdown(),
                ProjectionPushdown(),
                ConstantFolding(),
                SlicePushdown(),
                CommonSubexprElimination(),
                OperatorFusion()
            ]
        self.rules = rules
        self.max_passes = max_passes

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """
        Optimize the logical plan by applying rules until fixed-point or max_passes.
        """
        current_plan = plan
        for _ in range(self.max_passes):
            new_plan = current_plan
            for rule in self.rules:
                new_plan = rule.apply(new_plan)
                
            # Use .explain() which prints the full tree, instead of repr() which only prints the root
            if new_plan.explain() == current_plan.explain():
                break
            current_plan = new_plan
            
        return current_plan
