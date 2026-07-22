"""
Union Node
==========
Concatenates chunks from multiple physical plans.
"""
from typing import Iterator
from titanframe.plan.physical.node import PhysicalPlan, ExecutionContext, Chunk

class UnionExec(PhysicalPlan):
    def __init__(self, inputs: list[PhysicalPlan]):
        self.inputs = inputs

    def execute(self, context: ExecutionContext) -> Iterator[Chunk]:
        for plan in self.inputs:
            for chunk in plan.execute(context):
                yield chunk
