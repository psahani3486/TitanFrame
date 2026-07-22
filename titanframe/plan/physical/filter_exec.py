"""
FilterExec Node
===============

Filters rows based on a boolean predicate.
"""
from typing import Iterator, List
import pyarrow.compute as pc

from titanframe.plan.physical.node import PhysicalPlan, ExecutionContext, Chunk
from titanframe.plan.physical.evaluator import ExprEvaluator
from titanframe.expr.base import Expr

class FilterExec(PhysicalPlan):
    def __init__(self, input: PhysicalPlan, predicate: Expr):
        self.input = input
        self.predicate = predicate
        self.evaluator = ExprEvaluator()
        
    def children(self) -> List[PhysicalPlan]:
        return [self.input]
        
    def execute(self, context: ExecutionContext) -> Iterator[Chunk]:
        for chunk in self.input.execute(context):
            mask = self.evaluator.eval(self.predicate, chunk)
            
            # Use pc.filter to drop rows where mask is False or null
            filtered_batch = pc.filter(chunk.data, mask)
            if filtered_batch.num_rows > 0:
                yield Chunk(filtered_batch)
