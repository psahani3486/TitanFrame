from abc import ABC, abstractmethod
from typing import Iterator, List
import pyarrow as pa

class Chunk:
    __slots__ = ('data',)

    def __init__(self, data: pa.RecordBatch):
        self.data = data

    @property
    def num_rows(self) -> int:
        return self.data.num_rows

    def column(self, name: str) -> pa.Array:
        return self.data.column(name)

    @property
    def schema(self) -> pa.Schema:
        return self.data.schema

class ExecutionContext:

    def __init__(self, batch_size: int=65536, memory_manager=None):
        self.batch_size = batch_size
        self.memory_manager = memory_manager

class PhysicalPlan(ABC):

    @abstractmethod
    def execute(self, context: ExecutionContext) -> Iterator[Chunk]:
        pass

    def children(self) -> List['PhysicalPlan']:
        return []
