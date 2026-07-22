import pytest
import titanframe as tf
from titanframe.gpu.device import GPUDeviceManager

def test_multi_gpu_manager():
    mgr = GPUDeviceManager()
    count = mgr.get_device_count()
    assert count >= 0
