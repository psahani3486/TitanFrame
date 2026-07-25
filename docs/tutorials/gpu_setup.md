# TitanFrame Optional GPU Module & CUDA Acceleration Guide

This tutorial covers enabling optional CUDA kernel acceleration for **TitanFrame** using CuPy and Triton on GPU-equipped environments.

---

## 1. Execution Mode Architecture

TitanFrame operates in two execution modes:

1. **Standard Host Mode (CPU Vector Engine)**:
   - **Target**: Cloud Web Services (Render, Railway, Heroku), local laptops, standard CPU VMs.
   - **Backend**: DuckDB Vector Execution Kernel + Apache Arrow IPC streaming.
   - **Hardware Requirement**: Any x86_64 or ARM64 CPU.

2. **GPU Acceleration Mode (CUDA 12.x)**:
   - **Target**: NVIDIA GPU instances (AWS EC2 g4dn/g5, RunPod, Lambda Labs, Google Colab, Kaggle Notebooks).
   - **Backend**: CuPy ndarray memory pools & Triton CUDA kernels for elementwise math, hash aggregations, and sorting.
   - **Hardware Requirement**: NVIDIA GPU (Compute Capability 7.0+) with CUDA 12.x drivers.

---

## 2. Free-Tier Prototyping: Google Colab Setup

To prototype TitanFrame GPU acceleration on a free T4 GPU in Google Colab:

1. Open Google Colab and set **Runtime Type**: `GPU (T4)`.
2. Install TitanFrame with GPU dependencies:
   ```bash
   !pip install titanframe[gpu]
   ```
3. Enable GPU Acceleration in python:
   ```python
   import titanframe as tf

   # Enable CUDA execution module
   tf.config.gpu_enabled = True

   # Verify CUDA device detection
   from titanframe.gpu.device import DeviceManager
   print("GPU Available:", DeviceManager.is_gpu_available())
   print("Device Count:", DeviceManager.get_device_count())

   # Execute out-of-core query with CUDA acceleration
   df = tf.read_parquet("lineitem.parquet")
   res = df.filter(tf.col("l_discount") > 0.05).collect()
   print(res)
   ```

---

## 3. Production GPU Deployment (RunPod / Lambda Labs)

For dedicated high-throughput cluster deployments:

```bash
# Docker base image with CUDA 12.2 support
docker run --gpus all -it -p 8080:8080 python:3.11-slim

# Install TitanFrame
pip install titanframe[gpu]

# Launch Web Telemetry Studio
python -m titanframe --dashboard --port 8080
```
