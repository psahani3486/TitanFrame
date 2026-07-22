"""
Optimization Rule Base Class
============================
"""

from abc import ABC, abstractmethod

from titanframe.plan.logical.node import LogicalPlan


class OptimizationRule(ABC):
    """
    A single rewrite rule applied to the logical plan.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the optimization rule."""
        pass

    @abstractmethod
    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        """
        Apply this rule to the plan and return the rewritten plan.
        
        The returned plan may be the same as the input if no optimizations
        were possible.
        """
        pass
