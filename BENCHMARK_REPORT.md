# TitanFrame Reproducible Performance Benchmark Report

### Hardware & Environment
- **OS**: Windows 11
- **Python**: v3.13.5
- **CPU Cores**: 16 Worker Threads
- **System Memory**: 15.6 GB RAM

### Benchmark Dataset Specs
- **Target File**: `lineitem.parquet`
- **File Size**: 63.59 MB
- **Total Dataset Rows**: 6,000,000 Rows

### Comparative Benchmark Results

| Processing Engine | Execution Latency | Memory Footprint (Delta) | Throughput Rate | Execution Strategy |
|-------------------|-------------------|-------------------------|-----------------|--------------------|
| **Pandas 2.x** | **0.291s** | 253.9 MB | ~20.62M rows/s | Out-of-Core Arrow / In-Memory |
| **Polars** | N/A (Polars not installed) | N/A | N/A | N/A |
| **DuckDB (Vector SQL Engine)** | **0.075s** | 6.4 MB | ~80.0M rows/s | Out-of-Core Arrow / In-Memory |
| **TitanFrame (Arrow Out-of-Core)** | **0.156s** | 108.7 MB | ~38.46M rows/s | Out-of-Core Arrow / In-Memory |

---
### Reproducible Methodology
To execute this benchmark on custom datasets:
```bash
python benchmarks/run_reproducible_benchmark.py --rows 6000000 --limit 50
```
