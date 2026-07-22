"""
GPU Memory Pool Management
==========================

Provides a memory pool abstraction for GPU allocations, supporting RMM concepts or CuPy pools.
"""

from typing import Optional

try:
    import cupy as cp
except ImportError:
    cp = None

class GPUMemoryPool:
    """
    Manages GPU memory allocations using a pooled strategy to reduce CUDA allocation overhead.
    """
    
    def __init__(self, initial_size_bytes: Optional[int] = None):
        self.initial_size_bytes = initial_size_bytes
        self._pool = None
        if cp is not None:
            try:
                self._pool = cp.get_default_memory_pool()
                if initial_size_bytes:
                    self._pool.set_limit(initial_size_bytes)
            except Exception:
                self._pool = None

    def get_used_bytes(self) -> int:
        """Return currently allocated bytes in the pool."""
        if self._pool is not None:
            try:
                return self._pool.used_bytes()
            except Exception:
                return 0
        return 0

    def get_total_bytes(self) -> int:
        """Return total pool capacity or limit."""
        if self._pool is not None:
            try:
                return self._pool.total_bytes()
            except Exception:
                return 0
        return 0

    def free_all_blocks(self):
        """Release unused memory back to the CUDA driver."""
        if self._pool is not None:
            try:
                self._pool.free_all_blocks()
            except Exception:
                pass
