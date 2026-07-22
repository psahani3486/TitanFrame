"""
Sort GPU Kernels
================
CuPy sorting wrappers.
"""
from titanframe.gpu.device import DeviceManager, cp

class SortKernels:
    def __init__(self):
        self.device = DeviceManager()
        
    def sort_indices(self, array_pyarrow):
        """Returns sorted indices of a PyArrow array using the GPU."""
        if not self.device.available:
            raise RuntimeError("GPU not available")
            
        arr_gpu = self.device.to_gpu(array_pyarrow)
        indices = cp.argsort(arr_gpu)
        return self.device.to_cpu(indices)
