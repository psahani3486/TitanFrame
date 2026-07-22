"""
Streaming Pipeline Execution
=============================

Executes operators in a streaming pipeline model.
"""

from typing import Iterator, List
from titanframe.plan.physical.node import PhysicalPlan, ExecutionContext, Chunk

class StreamingPipeline:
    """
    Executes a pipeline of operators in a streaming fashion.
    """
    
    def __init__(self, root: PhysicalPlan, context: ExecutionContext):
        self.root = root
        self.context = context

    def execute(self) -> Iterator[Chunk]:
        """Stream chunks from the root execution node."""
        yield from self.root.execute(self.context)

    def collect(self) -> List[Chunk]:
        """Collect all chunks into a list."""
        return list(self.execute())
