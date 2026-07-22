from typing import Any, Optional
try:
    import cupy as cp
except ImportError:
    cp = None

class AsyncTransferManager:

    def __init__(self):
        self.stream = cp.cuda.Stream(non_blocking=True) if cp is not None else None

    def host_to_device(self, host_array: Any) -> Any:
        if cp is None or self.stream is None:
            return host_array
        with self.stream:
            device_arr = cp.asarray(host_array)
        return device_arr

    def device_to_host(self, device_array: Any) -> Any:
        if cp is None or self.stream is None:
            return device_array
        with self.stream:
            host_arr = cp.asnumpy(device_array)
        if self.stream is not None:
            self.stream.synchronize()
        return host_arr

    def synchronize(self):
        if self.stream is not None:
            self.stream.synchronize()
