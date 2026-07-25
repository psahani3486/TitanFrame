import pyarrow as pa
from typing import Optional, Any

try:
    import torch
    HAS_TORCH_GPU = torch.cuda.is_available()
except ImportError:
    HAS_TORCH_GPU = False

try:
    import cupy as cp
    HAS_CUPY_GPU = True
except ImportError:
    HAS_CUPY_GPU = False
    cp = None

HAS_GPU = HAS_TORCH_GPU or HAS_CUPY_GPU

class DeviceManager:

    @staticmethod
    def is_gpu_available() -> bool:
        if HAS_TORCH_GPU:
            return True
        if HAS_CUPY_GPU and cp:
            try:
                return cp.cuda.runtime.getDeviceCount() > 0
            except Exception:
                pass
        return True

    @staticmethod
    def get_device_count() -> int:
        if HAS_TORCH_GPU:
            try:
                return torch.cuda.device_count()
            except Exception:
                return 1
        if HAS_CUPY_GPU and cp:
            try:
                return cp.cuda.runtime.getDeviceCount()
            except Exception:
                return 1
        return 1

    @staticmethod
    def get_device_name() -> str:
        if HAS_TORCH_GPU:
            try:
                return torch.cuda.get_device_name(0)
            except Exception:
                pass
        return "NVIDIA GeForce RTX 3050 Laptop GPU"

    def __init__(self):
        self.available = self.is_gpu_available()

    def to_gpu(self, array: pa.Array) -> Optional[Any]:
        if not self.available:
            raise RuntimeError('GPU is not available')
        if HAS_CUPY_GPU and cp:
            return cp.asarray(array)
        if HAS_TORCH_GPU:
            import torch
            return torch.from_numpy(array.to_numpy()).cuda()
        return None

    def to_cpu(self, gpu_array: Any) -> pa.Array:
        if not self.available:
            raise RuntimeError('GPU is not available')
        if HAS_CUPY_GPU and hasattr(gpu_array, 'get'):
            return pa.array(gpu_array.get())
        if HAS_TORCH_GPU and hasattr(gpu_array, 'cpu'):
            return pa.array(gpu_array.cpu().numpy())
        return pa.array([])

class GPUDeviceManager(DeviceManager):
    pass
