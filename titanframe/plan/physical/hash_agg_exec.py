"""
HashAggExec Node
================

Aggregates data based on grouping keys using in-memory PyArrow grouping.
"""
from typing import Iterator, List
import pyarrow as pa
import pyarrow.compute as pc

from titanframe.plan.physical.node import PhysicalPlan, ExecutionContext, Chunk
from titanframe.expr.base import Expr, AggOp, AliasExpr
from titanframe.expr.agg_expr import AggExpr
from titanframe.expr.column_expr import ColumnExpr

class HashAggExec(PhysicalPlan):
    def __init__(self, input: PhysicalPlan, group_by: List[Expr], aggs: List[Expr], output_names: List[str]):
        self.input = input
        self.group_by = group_by
        self.aggs = aggs
        self.output_names = output_names
        
    def children(self) -> List[PhysicalPlan]:
        return [self.input]
        
    def execute(self, context: ExecutionContext) -> Iterator[Chunk]:
        buffer_list = []
        for chunk in self.input.execute(context):
            if context.memory_manager:
                buffer_list.append(context.memory_manager.register(chunk.data))
            else:
                buffer_list.append(chunk.data)
                
        if not buffer_list:
            return
            
        if context.memory_manager:
            batches = [b.get_data() for b in buffer_list]
        else:
            batches = buffer_list
            
        table = pa.Table.from_batches(batches)
        
        if context.memory_manager:
            for b in buffer_list:
                context.memory_manager.free(b)
        
        keys = []
        for g in self.group_by:
            if hasattr(g, "expr"): g = g.expr
            if isinstance(g, ColumnExpr):
                keys.append(g.column_name)
            else:
                raise NotImplementedError("Complex group by expressions must be projected first")
                
        agg_tuples = []
        for a in self.aggs:
            if isinstance(a, AliasExpr):
                a = a.child
            if isinstance(a, AggExpr):
                target_col = None
                if isinstance(a.child, ColumnExpr):
                    target_col = a.child.column_name
                else:
                    raise NotImplementedError("Complex agg expressions must be projected first")
                    
                op_map = {
                    AggOp.SUM: "hash_sum",
                    AggOp.MIN: "hash_min",
                    AggOp.MAX: "hash_max",
                    AggOp.COUNT: "hash_count",
                    AggOp.MEAN: "hash_mean",
                }
                if a.op not in op_map:
                    raise NotImplementedError(f"Agg op {a.op} not supported")
                    
                agg_tuples.append((target_col, op_map[a.op]))
            else:
                raise ValueError(f"Expected AggExpr, got {type(a)}")
                
        if not keys:
            result_arrays = []
            for target_col, op in agg_tuples:
                col_data = table.column(target_col)
                if op == "hash_sum": res = pc.sum(col_data)
                elif op == "hash_min": res = pc.min(col_data)
                elif op == "hash_max": res = pc.max(col_data)
                elif op == "hash_count": res = pc.count(col_data)
                elif op == "hash_mean": res = pc.mean(col_data)
                
                result_arrays.append(pa.array([res.as_py()]))
                
            res_batch = pa.RecordBatch.from_arrays(result_arrays, names=self.output_names)
            yield Chunk(res_batch)
            return

        res_table = table.group_by(keys).aggregate(agg_tuples)
        res_batch = res_table.to_batches()[0]
        
        renamed_batch = pa.RecordBatch.from_arrays(res_batch.columns, names=self.output_names)
        yield Chunk(renamed_batch)
