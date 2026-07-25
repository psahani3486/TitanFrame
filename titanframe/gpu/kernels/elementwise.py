from titanframe.gpu.device import DeviceManager, cp, HAS_TORCH_GPU

class ElementwiseKernels:

    def __init__(self):
        self.device = DeviceManager()

    def add(self, a_pyarrow, b_pyarrow):
        if not self.device.available:
            raise RuntimeError('GPU not available')
        a_gpu = self.device.to_gpu(a_pyarrow)
        b_gpu = self.device.to_gpu(b_pyarrow)
        if cp is not None:
            result = cp.add(a_gpu, b_gpu)
        elif HAS_TORCH_GPU:
            import torch
            result = torch.add(a_gpu, b_gpu)
        else:
            raise RuntimeError('No GPU tensor library available')
        return self.device.to_cpu(result)
