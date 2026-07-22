import pytest
from titanframe.gpu.autotuner import KernelAutotuner
from titanframe.gpu.kernels.scan import gpu_prefix_scan
from titanframe.gpu.kernels.string_ops import gpu_string_length

def test_autotuner():
    cfg = KernelAutotuner.get_config('elementwise_add', 500)
    assert 'block_size' in cfg
    assert 'grid_size' in cfg

def test_string_ops():
    lens = gpu_string_length(['hello', 'world'])
    assert list(lens) == [5, 5]

def test_prefix_scan():
    res = gpu_prefix_scan([1, 2, 3, 4])
    assert list(res) == [1, 3, 6, 10]
