from typing import Optional
from titanframe.memory.manager import MemoryManager
from titanframe.api.config import config

class ExecutionContext:

    def __init__(self, batch_size: Optional[int]=None, memory_manager: Optional[MemoryManager]=None):
        self.batch_size = batch_size or config.batch_size
        if memory_manager is None:
            limit = float(config.cpu_memory_limit) if config.cpu_memory_limit is not None else 8 * 1024 * 1024 * 1024.0
            self.memory_manager = MemoryManager(budget_bytes=limit)
        else:
            self.memory_manager = memory_manager
        self.use_gpu = config.use_gpu
        self.gpu_device_id = config.gpu_device_id

    def create_child(self) -> 'ExecutionContext':
        return ExecutionContext(batch_size=self.batch_size, memory_manager=self.memory_manager)
