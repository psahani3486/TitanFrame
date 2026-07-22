from typing import Iterator, List
from titanframe.plan.physical.node import PhysicalPlan, ExecutionContext, Chunk

class StreamingPipeline:

    def __init__(self, root: PhysicalPlan, context: ExecutionContext):
        self.root = root
        self.context = context

    def execute(self) -> Iterator[Chunk]:
        yield from self.root.execute(self.context)

    def collect(self) -> List[Chunk]:
        return list(self.execute())
