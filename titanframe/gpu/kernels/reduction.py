"""
Reduction GPU Kernels
=====================
Wraps CuPy reduction operations.
"""
from titanframe.gpu.device import DeviceManager, cp

class ReductionKernels:
    def __init__(self):
        self.device = DeviceManager()
        
    def sum(self, array_pyarrow):
        """Sums a PyArrow array on the GPU."""
        if not self.device.available:
            raise RuntimeError("GPU not available")
            
        arr_gpu = self.device.to_gpu(array_pyarrow)
        result = cp.sum(arr_gpu)
        return result.item()
