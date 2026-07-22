"""
ProjectExec Node
================

Evaluates expressions and yields chunks with new columns.
"""
from typing import Iterator, List
import pyarrow as pa

from titanframe.plan.physical.node import PhysicalPlan, ExecutionContext, Chunk
from titanframe.plan.physical.evaluator import ExprEvaluator
from titanframe.expr.base import Expr

class ProjectExec(PhysicalPlan):
    def __init__(self, input: PhysicalPlan, exprs: List[Expr], output_names: List[str]):
        self.input = input
        self.exprs = exprs
        self.output_names = output_names
        self.evaluator = ExprEvaluator()
        
    def children(self) -> List[PhysicalPlan]:
        return [self.input]
        
    def execute(self, context: ExecutionContext) -> Iterator[Chunk]:
        for chunk in self.input.execute(context):
            arrays = []
            for expr in self.exprs:
                res = self.evaluator.eval(expr, chunk)
                if isinstance(res, pa.Scalar):
                    res = pa.array([res.as_py()] * chunk.num_rows, type=res.type)
                arrays.append(res)
                
            batch = pa.RecordBatch.from_arrays(arrays, names=self.output_names)
            yield Chunk(batch)
