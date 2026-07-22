import pytest
import pyarrow as pa
from titanframe.memory.manager import MemoryManager
from titanframe.memory.tier import Tier
from titanframe.memory.buffer import DeviceBuffer

def test_memory_manager_spills():
    manager = MemoryManager(budget_bytes=200)
    
    batch1 = pa.record_batch([pa.array([1] * 10, type=pa.int64())], names=["a"])
    batch2 = pa.record_batch([pa.array([2] * 10, type=pa.int64())], names=["a"])
    batch3 = pa.record_batch([pa.array([3] * 10, type=pa.int64())], names=["a"])
    
    buf1 = manager.register(batch1)
    assert buf1.tier == Tier.RAM
    
    buf2 = manager.register(batch2)
    assert buf2.tier == Tier.RAM
    
    buf3 = manager.register(batch3)
    
    assert buf1.tier == Tier.DISK
    assert buf2.tier == Tier.RAM
    assert buf3.tier == Tier.RAM
    
    data1 = buf1.get_data()
    assert buf1.tier == Tier.RAM
    assert buf2.tier == Tier.DISK
    assert buf3.tier == Tier.RAM
    
    assert data1.column("a").to_pylist() == [1] * 10
    
    manager.cleanup()

def test_pinning_prevents_spill():
    manager = MemoryManager(budget_bytes=200)
    
    batch1 = pa.record_batch([pa.array([1] * 10, type=pa.int64())], names=["a"])
    batch2 = pa.record_batch([pa.array([2] * 10, type=pa.int64())], names=["a"])
    batch3 = pa.record_batch([pa.array([3] * 10, type=pa.int64())], names=["a"])
    
    buf1 = manager.register(batch1)
    buf1.pin()
    
    buf2 = manager.register(batch2)
    
    buf3 = manager.register(batch3)
    
    assert buf1.tier == Tier.RAM
    assert buf2.tier == Tier.DISK
    assert buf3.tier == Tier.RAM
    
    buf1.unpin()
    
    manager.cleanup()
