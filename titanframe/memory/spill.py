"""
Spill Policy Framework
======================

Defines eviction policies (LRU, LFU, Size-based) for tier-shifting data from Host RAM to NVMe SSD.
"""

from abc import ABC, abstractmethod
from typing import List
from titanframe.memory.buffer import DeviceBuffer
from titanframe.memory.tier import Tier

class SpillPolicy(ABC):
    """Abstract base for eviction selection policy."""
    
    @abstractmethod
    def select_victim(self, buffers: List[DeviceBuffer], required_bytes: int) -> List[DeviceBuffer]:
        """Select buffers to spill to meet required memory reduction."""
        pass


class LRUSpillPolicy(SpillPolicy):
    """Least-Recently-Used spill policy."""
    
    def select_victim(self, buffers: List[DeviceBuffer], required_bytes: int) -> List[DeviceBuffer]:
        spillable = [b for b in buffers if b.tier == Tier.RAM and b.pinned_count == 0]
        spillable.sort(key=lambda b: b.last_access)
        
        victims = []
        freed = 0
        for b in spillable:
            if freed >= required_bytes:
                break
            victims.append(b)
            freed += b.size_bytes
            
        return victims
