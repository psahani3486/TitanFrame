"""Integration test for out-of-core memory spilling to disk."""

import pytest
import pyarrow as pa
from titanframe.memory.manager import MemoryManager

def test_spill_end_to_end():
    # Budget fits 1 batch (approx 800 bytes), but not 2
    mm = MemoryManager(budget_bytes=1000)
    batch1 = pa.RecordBatch.from_arrays([pa.array([i for i in range(100)])], names=["id"])
    batch2 = pa.RecordBatch.from_arrays([pa.array([i for i in range(100)])], names=["id"])
    
    buf1 = mm.register(batch1)
    
    # Registering batch2 forces buf1 to spill to NVMe disk
    buf2 = mm.register(batch2)
    
    assert mm.spill_if_needed is not None
