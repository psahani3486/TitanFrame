import pathlib
import tempfile
import pyarrow as pa
from typing import List

from titanframe.memory.tier import Tier
from titanframe.memory.buffer import DeviceBuffer

class MemoryManager:
    """Manages memory limits and spills buffers to disk when limits are exceeded."""
    
    def __init__(self, budget_bytes: float):
        self.budget_bytes = float(budget_bytes)
        self.current_usage = 0.0
        self.buffers: List[DeviceBuffer] = []
        
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = pathlib.Path(self._temp_dir.name)
        
    def register(self, batch: pa.RecordBatch) -> DeviceBuffer:
        """Register a new RecordBatch allocation, wrapping it in a DeviceBuffer."""
        size = batch.nbytes
        self.spill_if_needed(size)
        
        buf = DeviceBuffer(batch, self, Tier.RAM)
        self.buffers.append(buf)
        self.current_usage += size
        
        from titanframe.telemetry.tracker import tracker
        tracker.set_budgets(int(self.budget_bytes), 100 * 1024**3) # hardcode spill budget for now
        tracker.record_allocation(size)
        
        return buf
        
    def register_reloaded(self, size: int):
        """Called by a buffer when it reloads itself from disk."""
        self.current_usage += size
        from titanframe.telemetry.tracker import tracker
        tracker.record_allocation(size)
        tracker.record_free(size, is_spill=True)
        
    def free(self, buffer: DeviceBuffer):
        """Explicitly free a buffer."""
        from titanframe.telemetry.tracker import tracker
        
        if buffer.tier == Tier.RAM:
            self.current_usage -= buffer.size_bytes
            tracker.record_free(buffer.size_bytes, is_spill=False)
        elif buffer.tier == Tier.DISK and buffer._file_path:
            buffer._file_path.unlink(missing_ok=True)
            tracker.record_free(buffer.size_bytes, is_spill=True)
            
        if buffer in self.buffers:
            self.buffers.remove(buffer)
            
    def spill_if_needed(self, required_bytes: int):
        """Spills unpinned RAM buffers to DISK (LRU order) until there is enough space."""
        if self.current_usage + required_bytes <= self.budget_bytes:
            return
            
        spillable = [
            b for b in self.buffers
            if b.tier == Tier.RAM and b.pinned_count == 0
        ]
        spillable.sort(key=lambda b: b.last_access)
        
        for buf in spillable:
            if self.current_usage + required_bytes <= self.budget_bytes:
                break
                
            buf.spill_to_disk(self.temp_path)
            self.current_usage -= buf.size_bytes
            
            # Telemetry
            from titanframe.telemetry.tracker import tracker
            tracker.record_free(buf.size_bytes, is_spill=False)
            tracker.record_spill(buf.size_bytes)
            
        if self.current_usage + required_bytes > self.budget_bytes:
            raise MemoryError(
                f"Memory budget exceeded! "
                f"Current: {self.current_usage}, Required: {required_bytes}, "
                f"Budget: {self.budget_bytes}. "
                f"Spillable buffers might all be pinned."
            )
            
    def cleanup(self):
        """Cleans up temporary disk files."""
        for buf in list(self.buffers):
            self.free(buf)
        self._temp_dir.cleanup()
        
    def __del__(self):
        self.cleanup()
