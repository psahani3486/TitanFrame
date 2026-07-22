from abc import ABC, abstractmethod
from titanframe.plan.logical.node import LogicalPlan

class OptimizationRule(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        pass
