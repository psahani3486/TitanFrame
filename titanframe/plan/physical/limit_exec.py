from typing import Iterator, List
from titanframe.plan.physical.node import PhysicalPlan, ExecutionContext, Chunk

class LimitExec(PhysicalPlan):

    def __init__(self, input: PhysicalPlan, limit: int | None, offset: int=0):
        self.input = input
        self.limit = limit
        self.offset = offset

    def children(self) -> List[PhysicalPlan]:
        return [self.input]

    def execute(self, context: ExecutionContext) -> Iterator[Chunk]:
        rows_skipped = 0
        rows_yielded = 0
        for chunk in self.input.execute(context):
            batch = chunk.data
            if rows_skipped < self.offset:
                to_skip = min(self.offset - rows_skipped, batch.num_rows)
                rows_skipped += to_skip
                if to_skip == batch.num_rows:
                    continue
                batch = batch.slice(to_skip, batch.num_rows - to_skip)
            if self.limit is not None:
                remaining = self.limit - rows_yielded
                if remaining <= 0:
                    break
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
            rows_yielded += batch.num_rows
            yield Chunk(batch)
            if self.limit is not None and rows_yielded >= self.limit:
                break
