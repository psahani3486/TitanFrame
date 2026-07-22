"""
Elementwise GPU Kernels
=======================
Wraps CuPy elementwise operations for GPU-accelerated math.
"""
from titanframe.gpu.device import DeviceManager, cp

class ElementwiseKernels:
    def __init__(self):
        self.device = DeviceManager()
        
    def add(self, a_pyarrow, b_pyarrow):
        """Adds two PyArrow arrays on the GPU."""
        if not self.device.available:
            raise RuntimeError("GPU not available")
            
        a_gpu = self.device.to_gpu(a_pyarrow)
        b_gpu = self.device.to_gpu(b_pyarrow)
        
        result = cp.add(a_gpu, b_gpu)
        return self.device.to_cpu(result)
