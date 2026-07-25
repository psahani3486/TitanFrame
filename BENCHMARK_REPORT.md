# TitanFrame Reproducible Multi-Run Performance Benchmark Report

### Hardware & Environment
- **OS**: Windows 11
- **Python**: v3.13.5
- **CPU Cores**: 16 Worker Threads
- **System Memory**: 15.6 GB RAM

### Benchmark Dataset Specs
- **Target File**: `lineitem.parquet`
- **File Size**: 63.59 MB
- **Total Dataset Rows**: 6,000,000 Rows
- **Benchmark Iterations**: 5 Multi-Run Iterations per Engine

### Statistical Benchmark Results (Mean ± Std Dev)

| Processing Engine | Mean Latency | Standard Deviation (Variance) | Memory Footprint (Delta) | Throughput Rate | Execution Strategy |
|-------------------|--------------|-------------------------------|-------------------------|-----------------|--------------------|
| **Pandas 2.x** | **0.207s** | ±0.017s | 179.6 MB | ~28.99M rows/s | Arrow Streaming / Vectorized |
| **Polars (Lazy Streaming)** | **0.027s** | ±0.002s | 39.3 MB | ~222.22M rows/s | Arrow Streaming / Vectorized |
| **DuckDB (Vector SQL Engine)** | **0.064s** | ±0.002s | 4.6 MB | ~93.75M rows/s | Arrow Streaming / Vectorized |
| **TitanFrame (Arrow Out-of-Core)** | **0.103s** | ±0.025s | 38.2 MB | ~58.25M rows/s | Arrow Streaming / Vectorized |

---
### Reproducible Methodology & Statistical Confidence
This benchmark runs each query engine **5 consecutive times** to calculate mean latency and standard deviation, eliminating single-run cache warm-up variance.

To execute this benchmark on custom datasets:
```bash
python benchmarks/run_reproducible_benchmark.py --rows 6000000 --limit 50 --runs 5
```
