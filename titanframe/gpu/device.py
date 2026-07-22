"""
Device Manager
==============
Manages GPU devices and memory movement if CuPy is installed.
"""
import pyarrow as pa
from typing import Optional

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

class DeviceManager:
    """Manages GPUs and data placement."""
    
    @staticmethod
    def is_gpu_available() -> bool:
        """Returns True if CuPy is installed and at least one GPU is available."""
        if not HAS_GPU:
            return False
        try:
            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False

    @staticmethod
    def get_device_count() -> int:
        if not HAS_GPU:
            return 0
        try:
            return cp.cuda.runtime.getDeviceCount()
        except Exception:
            return 0
            
    def __init__(self):
        self.available = self.is_gpu_available()
        
    def to_gpu(self, array: pa.Array) -> Optional["cp.ndarray"]:
        """Moves a PyArrow array to GPU as CuPy array."""
        if not self.available:
            raise RuntimeError("GPU is not available")
            
        return cp.asarray(array)
        
    def to_cpu(self, gpu_array: "cp.ndarray") -> pa.Array:
        """Moves a CuPy array back to CPU as PyArrow array."""
        if not self.available:
            raise RuntimeError("GPU is not available")
            
        return pa.array(gpu_array.get())

class GPUDeviceManager(DeviceManager):
    pass
