"""
SinkExec Node
=============

Writes chunks to a sink (e.g. Parquet file).
"""
from typing import Iterator, List
import pyarrow.parquet as pq
from titanframe.plan.physical.node import PhysicalPlan, ExecutionContext, Chunk
from titanframe.plan.logical.sink import SinkFormat

class SinkExec(PhysicalPlan):
    def __init__(self, input: PhysicalPlan, target: str, format: SinkFormat):
        self.input = input
        self.target = target
        self.format = format
        
    def children(self) -> List[PhysicalPlan]:
        return [self.input]
        
    def execute(self, context: ExecutionContext) -> Iterator[Chunk]:
        writer = None
        for chunk in self.input.execute(context):
            if writer is None:
                if self.format == SinkFormat.PARQUET:
                    writer = pq.ParquetWriter(self.target, chunk.schema)
                else:
                    raise NotImplementedError(f"SinkFormat {self.format} not implemented")
                    
            writer.write_batch(chunk.data)
            yield chunk
            
        if writer is not None:
            writer.close()
