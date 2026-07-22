"""
HashJoinExec Node
=================

Joins two tables using an in-memory hash join (via PyArrow Table.join).
"""
from typing import Iterator, List
import pyarrow as pa

from titanframe.plan.physical.node import PhysicalPlan, ExecutionContext, Chunk

class HashJoinExec(PhysicalPlan):
    def __init__(self, left: PhysicalPlan, right: PhysicalPlan, keys: List[str], how: str, suffix: str):
        self.left = left
        self.right = right
        self.keys = keys
        self.how = how
        self.suffix = suffix
        
    def children(self) -> List[PhysicalPlan]:
        return [self.left, self.right]
        
    def execute(self, context: ExecutionContext) -> Iterator[Chunk]:
        left_buffers = []
        for chunk in self.left.execute(context):
            if context.memory_manager: left_buffers.append(context.memory_manager.register(chunk.data))
            else: left_buffers.append(chunk.data)
            
        if not left_buffers:
            return
            
        right_buffers = []
        for chunk in self.right.execute(context):
            if context.memory_manager: right_buffers.append(context.memory_manager.register(chunk.data))
            else: right_buffers.append(chunk.data)
            
        if context.memory_manager:
            left_table = pa.Table.from_batches([b.get_data() for b in left_buffers])
            for b in left_buffers: context.memory_manager.free(b)
        else:
            left_table = pa.Table.from_batches(left_buffers)
            
        if not right_buffers:
            if self.how == "inner": return
            raise ValueError("Empty right side of outer join not fully implemented")
            
        if context.memory_manager:
            right_table = pa.Table.from_batches([b.get_data() for b in right_buffers])
            for b in right_buffers: context.memory_manager.free(b)
        else:
            right_table = pa.Table.from_batches(right_buffers)
        
        pa_join_type = self.how
        if pa_join_type == "outer": pa_join_type = "full outer"
        
        res_table = left_table.join(
            right_table, 
            keys=self.keys, 
            join_type=pa_join_type, 
            right_suffix=self.suffix
        )
        
        for batch in res_table.to_batches(max_chunksize=context.batch_size):
            yield Chunk(batch)
