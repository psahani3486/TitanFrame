import pytest
import pyarrow as pa
from titanframe.memory.manager import MemoryManager

def test_memory_manager_budget():
    mm = MemoryManager(budget_bytes=1000)
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], names=['a'])
    buf = mm.register(batch)
    assert buf is not None
    assert mm.current_usage > 0
    mm.free(buf)
    assert mm.current_usage == 0
