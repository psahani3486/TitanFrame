import pytest
import pyarrow as pa
from titanframe.gpu.device import DeviceManager, HAS_GPU
from titanframe.gpu.kernels.elementwise import ElementwiseKernels
from titanframe.gpu.kernels.reduction import ReductionKernels

@pytest.mark.skipif(not HAS_GPU, reason='CuPy not installed or no GPU available')
def test_gpu_elementwise_add():
    a = pa.array([1, 2, 3], type=pa.int64())
    b = pa.array([4, 5, 6], type=pa.int64())
    kernels = ElementwiseKernels()
    result = kernels.add(a, b)
    assert result.to_pylist() == [5, 7, 9]

@pytest.mark.skipif(not HAS_GPU, reason='CuPy not installed or no GPU available')
def test_gpu_reduction_sum():
    a = pa.array([1, 2, 3, 4, 5], type=pa.int64())
    kernels = ReductionKernels()
    result = kernels.sum(a)
    assert result == 15
