import pytest
import pyarrow as pa
from titanframe.memory.manager import MemoryManager
from titanframe.memory.tier import Tier
from titanframe.memory.buffer import DeviceBuffer

def test_memory_manager_spills():
    # Create a small budget of 200 bytes
    manager = MemoryManager(budget_bytes=200)
    
    # Create some mock record batches
    # Each batch of 10 ints is ~80 bytes usually
    batch1 = pa.record_batch([pa.array([1] * 10, type=pa.int64())], names=["a"])
    batch2 = pa.record_batch([pa.array([2] * 10, type=pa.int64())], names=["a"])
    batch3 = pa.record_batch([pa.array([3] * 10, type=pa.int64())], names=["a"])
    
    buf1 = manager.register(batch1)
    assert buf1.tier == Tier.RAM
    
    buf2 = manager.register(batch2)
    assert buf2.tier == Tier.RAM
    
    # Registering batch3 should trigger a spill because 3 batches * 80 bytes = 240 bytes > 200 budget
    buf3 = manager.register(batch3)
    
    # Check that buf1 spilled (because it is LRU)
    assert buf1.tier == Tier.DISK
    assert buf2.tier == Tier.RAM
    assert buf3.tier == Tier.RAM
    
    # Reload buf1
    # This should cause buf2 to spill
    data1 = buf1.get_data()
    assert buf1.tier == Tier.RAM
    assert buf2.tier == Tier.DISK
    assert buf3.tier == Tier.RAM
    
    # Validate data
    assert data1.column("a").to_pylist() == [1] * 10
    
    manager.cleanup()

def test_pinning_prevents_spill():
    manager = MemoryManager(budget_bytes=200)
    
    batch1 = pa.record_batch([pa.array([1] * 10, type=pa.int64())], names=["a"])
    batch2 = pa.record_batch([pa.array([2] * 10, type=pa.int64())], names=["a"])
    batch3 = pa.record_batch([pa.array([3] * 10, type=pa.int64())], names=["a"])
    
    buf1 = manager.register(batch1)
    buf1.pin() # Pinned, cannot be spilled!
    
    buf2 = manager.register(batch2)
    
    # Registering batch3 triggers spill. Since buf1 is pinned, buf2 should spill.
    buf3 = manager.register(batch3)
    
    assert buf1.tier == Tier.RAM
    assert buf2.tier == Tier.DISK
    assert buf3.tier == Tier.RAM
    
    buf1.unpin()
    
    manager.cleanup()
