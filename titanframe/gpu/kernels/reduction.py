from titanframe.gpu.device import DeviceManager, cp, HAS_TORCH_GPU

class ReductionKernels:

    def __init__(self):
        self.device = DeviceManager()

    def sum(self, array_pyarrow):
        if not self.device.available:
            raise RuntimeError('GPU not available')
        arr_gpu = self.device.to_gpu(array_pyarrow)
        if cp is not None:
            result = cp.sum(arr_gpu)
            return result.item()
        elif HAS_TORCH_GPU:
            import torch
            return float(torch.sum(arr_gpu).item())
        return 0.0
