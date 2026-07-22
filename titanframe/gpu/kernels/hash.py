from titanframe.gpu.device import DeviceManager, cp

class HashKernels:

    def __init__(self):
        self.device = DeviceManager()

    def hash_array(self, array_pyarrow):
        if not self.device.available:
            raise RuntimeError('GPU not available')
        arr_gpu = self.device.to_gpu(array_pyarrow)
        result = cp.zeros_like(arr_gpu)
        return self.device.to_cpu(result)
