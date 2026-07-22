import time
import uuid
import pathlib
import pyarrow as pa
import pyarrow.ipc as ipc
from typing import Optional

from titanframe.memory.tier import Tier

class DeviceBuffer:
    """A wrapper for a PyArrow RecordBatch that tracks its tier and handles spilling."""
    
    def __init__(self, batch: pa.RecordBatch, manager, tier: Tier = Tier.RAM):
        self.manager = manager
        self.tier = tier
        self.size_bytes = batch.nbytes
        self.pinned_count = 0
        self.last_access = time.time()
        self.id = str(uuid.uuid4())
        
        self._data: Optional[pa.RecordBatch] = batch
        self._file_path: Optional[pathlib.Path] = None

    def pin(self):
        """Pin the buffer so it cannot be spilled."""
        self.pinned_count += 1
        self.last_access = time.time()
        
    def unpin(self):
        """Unpin the buffer, allowing it to be spilled."""
        assert self.pinned_count > 0, "Unpinning an unpinned buffer"
        self.pinned_count -= 1
        
    def get_data(self) -> pa.RecordBatch:
        """Get the data, reloading it from disk if necessary."""
        self.last_access = time.time()
        
        if self.tier == Tier.RAM:
            return self._data
            
        if self.tier == Tier.DISK:
            assert self._file_path is not None, "File path missing for DISK tier"
            
            # Request memory from the manager before reloading
            self.manager.spill_if_needed(self.size_bytes)
            
            with pa.OSFile(str(self._file_path), 'rb') as f:
                with ipc.RecordBatchStreamReader(f) as reader:
                    self._data = reader.read_next_batch()
                    
            self._file_path.unlink(missing_ok=True)
            self._file_path = None
            self.tier = Tier.RAM
            
            self.manager.register_reloaded(self.size_bytes)
            return self._data
            
        raise NotImplementedError(f"Tier {self.tier} reload not implemented")
        
    def spill_to_disk(self, temp_dir: pathlib.Path):
        """Spills the buffer to disk using PyArrow IPC stream format."""
        assert self.pinned_count == 0, "Cannot spill a pinned buffer"
        assert self.tier == Tier.RAM, "Can only spill from RAM"
        assert self._data is not None, "Buffer has no data to spill"
        
        self._file_path = temp_dir / f"{self.id}.arrow"
        
        with pa.OSFile(str(self._file_path), 'wb') as f:
            with ipc.RecordBatchStreamWriter(f, self._data.schema) as writer:
                writer.write_batch(self._data)
                
        # Drop the reference so Python/PyArrow can garbage collect it
        self._data = None
        self.tier = Tier.DISK
