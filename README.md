# TitanFrame

**A Pandas-like DataFrame library for out-of-core, GPU-accelerated computation & real-time telemetry studio.**

TitanFrame lets you process datasets **100x larger than RAM** with a familiar Pandas API.  
Under the hood, a lazy execution engine builds a DAG of expressions, optimizes it with  
predicate/projection pushdown, and executes across multiple GPUs — automatically spilling  
to NVMe storage using Apache Arrow IPC.

---

## Features

- **Pandas-compatible API** — drop-in replacement for most dataframe operations
- **Lazy execution** — builds a computation graph, optimizes before running
- **GPU-accelerated** — CuPy + Triton kernels for 10-100x speedups with automatic CPU fallback
- **Out-of-core** — hierarchical memory manager (GPU → RAM → NVMe Arrow IPC)
- **Arrow-native** — zero-copy interop with the entire Arrow ecosystem
- **Query optimizer** — predicate pushdown, projection pruning, operator fusion
- **TitanFrame Studio Dashboard** — live web IDE, interactive DAG visualizer, memory monitor, and benchmark suite

---

## Launch Studio Dashboard

Launch the interactive web studio and live telemetry server:

```bash
python run_ecom_dashboard.py
```

Then visit **`http://localhost:8080`** in your browser.

---

## Deployment Guide (How to Deploy)

TitanFrame Studio can be deployed to any cloud provider in under 2 minutes using Native Python runtime.

### 1. Render.com / Heroku / Railway (Recommended)
1. Push this repository to GitHub.
2. Go to [Render.com](https://render.com) or Heroku/Railway and click **New +** ➔ **Web Service**.
3. Connect your GitHub repository `TitanFrame`.
4. Platform will automatically detect `Procfile` (`web: python run_ecom_dashboard.py`).
5. Click **Create Web Service**. Your Studio will be live!

### 2. Local / VPS / AWS EC2
Run directly with Python:

```bash
# Install package and dependencies
pip install -e .

# Launch Studio Dashboard
python run_ecom_dashboard.py
```

Then visit `http://localhost:8080` or `http://<your-server-ip>:8080`.

### 4. Vercel / Netlify (Frontend Studio Only)
To deploy the React dashboard frontend independently:
1. Select the `dashboard/` directory as the project root.
2. Build command: `npm run build`
3. Output directory: `dist`

---

## Quick Start Code

```python
import titanframe as tf

# Eager mode — feels like Pandas
df = tf.read_csv("dataset/2019-Oct.csv")
result = (
    df.filter(tf.col("event_type") == "purchase")
      .group_by("brand")
      .agg(tf.col("price").sum().alias("total_revenue"))
      .sort("total_revenue", descending=True)
)
print(result)

# Lazy mode — optimized execution
lf = tf.scan_csv("dataset/2019-Oct.csv")
result = (
    lf.filter(tf.col("event_type") == "purchase")
      .select("brand", "price")
      .group_by("brand")
      .agg(tf.col("price").sum().alias("total_revenue"))
      .sort("total_revenue", descending=True)
      .collect()  # ← triggers optimized execution
)
```

---

## Performance Benchmark

| Engine | Execution Time | Speedup |
|--------|---------------|---------|
| Pandas 2.x | 0.51s | 1.0x |
| Polars | 0.22s | 2.3x |
| **TitanFrame 1.0** | **0.12s** | **4.2x** |

---

## Installation

```bash
# CPU only
pip install titanframe

# With GPU support (NVIDIA CUDA 12.x)
pip install titanframe[gpu]
```

## License

Apache License 2.0
