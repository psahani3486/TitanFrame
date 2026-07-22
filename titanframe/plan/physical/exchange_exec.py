"""
ExchangeExec Node
=================

Data repartitioning (stubbed).
"""
from typing import Iterator, List
from titanframe.plan.physical.node import PhysicalPlan, ExecutionContext, Chunk

class ExchangeExec(PhysicalPlan):
    def __init__(self, input: PhysicalPlan, partition_type: str):
        self.input = input
        self.partition_type = partition_type
        
    def children(self) -> List[PhysicalPlan]:
        return [self.input]
        
    def execute(self, context: ExecutionContext) -> Iterator[Chunk]:
        for chunk in self.input.execute(context):
            yield chunk
