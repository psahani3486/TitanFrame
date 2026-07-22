"""
Physical Plan Base Classes
==========================

Defines the physical plan interface, chunks, and execution context.
"""
from abc import ABC, abstractmethod
from typing import Iterator, List
import pyarrow as pa

class Chunk:
    """
    A lightweight wrapper around pyarrow.RecordBatch representing a chunk of data.
    """
    __slots__ = ("data",)
    
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
    """
    State for a running query.
    Tracks memory budgets, thread pools, and active sessions.
    """
    def __init__(self, batch_size: int = 65536, memory_manager=None):
        self.batch_size = batch_size
        self.memory_manager = memory_manager


class PhysicalPlan(ABC):
    """
    Base class for executable physical plan nodes.
    
    The engine operates on a volcano/pull-based iterator model where each node
    yields Chunks of data to its parent.
    """
    
    @abstractmethod
    def execute(self, context: ExecutionContext) -> Iterator[Chunk]:
        """Pull-based execution: yield chunks one at a time."""
        pass

    def children(self) -> List["PhysicalPlan"]:
        return []
