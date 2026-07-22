"""
SortMergeExec Node
==================

Sorts data according to provided keys.
"""
from typing import Iterator, List
import pyarrow as pa
import pyarrow.compute as pc

from titanframe.plan.physical.node import PhysicalPlan, ExecutionContext, Chunk
from titanframe.expr.base import Expr
from titanframe.expr.column_expr import ColumnExpr

class SortMergeExec(PhysicalPlan):
    def __init__(self, input: PhysicalPlan, sort_keys: List[Expr]):
        self.input = input
        self.sort_keys = sort_keys
        
    def children(self) -> List[PhysicalPlan]:
        return [self.input]
        
    def execute(self, context: ExecutionContext) -> Iterator[Chunk]:
        buffer_list = []
        for chunk in self.input.execute(context):
            if context.memory_manager: buffer_list.append(context.memory_manager.register(chunk.data))
            else: buffer_list.append(chunk.data)
            
        if not buffer_list:
            return
            
        if context.memory_manager:
            table = pa.Table.from_batches([b.get_data() for b in buffer_list])
            for b in buffer_list: context.memory_manager.free(b)
        else:
            table = pa.Table.from_batches(buffer_list)
        
        sort_opts = []
        for k in self.sort_keys:
            order = "ascending"
            col_name = None
            
            from titanframe.expr.base import SortExpr, SortOrder
            from titanframe.expr.column_expr import ColumnExpr
            
            if isinstance(k, SortExpr):
                if k.order == SortOrder.DESC:
                    order = "descending"
                inner = k.child
                if isinstance(inner, ColumnExpr):
                    col_name = inner.column_name
                else:
                    raise NotImplementedError("Only column sorting is supported")
            elif isinstance(k, ColumnExpr):
                col_name = k.column_name
            else:
                raise NotImplementedError("Only column sorting is supported")
                
            sort_opts.append((col_name, order))
                
        indices = pc.sort_indices(table, sort_keys=sort_opts)
        sorted_table = table.take(indices)
        
        for batch in sorted_table.to_batches(max_chunksize=context.batch_size):
            yield Chunk(batch)
