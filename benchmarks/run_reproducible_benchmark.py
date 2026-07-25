"""
TitanFrame Reproducible Multi-Engine Out-of-Core Benchmark Suite
Evaluates Pandas, Polars, DuckDB, and TitanFrame across dataset scales.
"""

import os
import sys
import time
import argparse
import platform
import psutil
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import titanframe as tf

def get_peak_ram_mb():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)

def generate_synthetic_dataset(filename: str, num_rows: int):
    print(f"[*] Generating {num_rows:,} rows of synthetic TPC-H lineitem data...")
    t0 = time.time()
    np.random.seed(42)
    orderkey = np.random.randint(1, num_rows // 4 + 1, size=num_rows, dtype=np.int64)
    quantity = np.random.randint(1, 50, size=num_rows, dtype=np.int32)
    extendedprice = quantity * np.random.uniform(1.0, 100.0, size=num_rows).astype(np.float64)
    discount = np.random.uniform(0.0, 0.1, size=num_rows).astype(np.float64)
    returnflag = np.random.choice(['A', 'N', 'R'], size=num_rows)
    
    df = pd.DataFrame({
        'l_orderkey': orderkey,
        'l_quantity': quantity,
        'l_extendedprice': extendedprice,
        'l_discount': discount,
        'l_returnflag': returnflag
    })
    df.to_parquet(filename, engine='pyarrow', compression='snappy')
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"[+] Dataset saved to {filename} ({size_mb:.1f} MB, took {time.time() - t0:.2f}s)")
    return filename

def benchmark_pandas(file_path: str):
    print("[*] Running Pandas benchmark...")
    mem_before = get_peak_ram_mb()
    t0 = time.time()
    pdf = pd.read_parquet(file_path)
    filtered = pdf[pdf['l_discount'] > 0.05]
    res = filtered.groupby('l_returnflag').agg(
        sum_qty=('l_quantity', 'sum'),
        sum_price=('l_extendedprice', 'sum')
    )
    duration = time.time() - t0
    mem_after = get_peak_ram_mb()
    return {'engine': 'Pandas 2.x', 'duration': round(duration, 3), 'peak_ram_mb': max(0.1, round(mem_after - mem_before, 1)), 'rows': len(pdf)}

def benchmark_polars(file_path: str):
    try:
        import polars as pl
        print("[*] Running Polars benchmark...")
        mem_before = get_peak_ram_mb()
        t0 = time.time()
        res = (
            pl.scan_parquet(file_path)
              .filter(pl.col('l_discount') > 0.05)
              .group_by('l_returnflag')
              .agg(
                  pl.col('l_quantity').sum().alias('sum_qty'),
                  pl.col('l_extendedprice').sum().alias('sum_price')
              )
              .collect()
        )
        duration = time.time() - t0
        mem_after = get_peak_ram_mb()
        return {'engine': 'Polars (Lazy Streaming)', 'duration': round(duration, 3), 'peak_ram_mb': max(0.1, round(mem_after - mem_before, 1)), 'rows': res.height}
    except ImportError:
        return {'engine': 'Polars', 'duration': None, 'peak_ram_mb': None, 'error': 'Polars not installed'}

def benchmark_duckdb(file_path: str):
    try:
        import duckdb
        print("[*] Running DuckDB benchmark...")
        mem_before = get_peak_ram_mb()
        t0 = time.time()
        conn = duckdb.connect()
        res = conn.execute(f"""
            SELECT l_returnflag, SUM(l_quantity) AS sum_qty, SUM(l_extendedprice) AS sum_price
            FROM '{file_path}'
            WHERE l_discount > 0.05
            GROUP BY l_returnflag
        """).fetch_df()
        duration = time.time() - t0
        mem_after = get_peak_ram_mb()
        return {'engine': 'DuckDB (Vector SQL Engine)', 'duration': round(duration, 3), 'peak_ram_mb': max(0.1, round(mem_after - mem_before, 1)), 'rows': len(res)}
    except ImportError:
        return {'engine': 'DuckDB', 'duration': None, 'peak_ram_mb': None, 'error': 'DuckDB not installed'}

def benchmark_titanframe(file_path: str, memory_limit_mb: int = 50):
    print(f"[*] Running TitanFrame Out-of-Core benchmark (RAM limit: {memory_limit_mb} MB)...")
    tf.config.cpu_memory_limit = memory_limit_mb * 1024 * 1024
    tf.config.spill_threshold = 0.85
    mem_before = get_peak_ram_mb()
    t0 = time.time()
    lf = tf.scan_parquet(file_path)
    res = (
        lf.filter(tf.col('l_discount') > 0.05)
          .group_by('l_returnflag')
          .agg(
              tf.col('l_quantity').sum().alias('sum_qty'),
              tf.col('l_extendedprice').sum().alias('sum_price')
          )
          .collect()
    )
    duration = time.time() - t0
    mem_after = get_peak_ram_mb()
    return {'engine': 'TitanFrame (Arrow Out-of-Core)', 'duration': round(duration, 3), 'peak_ram_mb': max(0.1, round(mem_after - mem_before, 1)), 'rows': len(res)}

def generate_markdown_report(results: list, file_path: str, num_rows: int):
    file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
    report_content = f"""# TitanFrame Reproducible Performance Benchmark Report

### Hardware & Environment
- **OS**: {platform.system()} {platform.release()}
- **Python**: v{sys.version.split()[0]}
- **CPU Cores**: {os.cpu_count()} Worker Threads
- **System Memory**: {round(psutil.virtual_memory().total / (1024**3), 1)} GB RAM

### Benchmark Dataset Specs
- **Target File**: `{os.path.basename(file_path)}`
- **File Size**: {file_size_mb} MB
- **Total Dataset Rows**: {num_rows:,} Rows

### Comparative Benchmark Results

| Processing Engine | Execution Latency | Memory Footprint (Delta) | Throughput Rate | Execution Strategy |
|-------------------|-------------------|-------------------------|-----------------|--------------------|
"""
    for r in results:
        if r.get('duration'):
            throughput = round((num_rows / 1000000.0) / r['duration'], 2)
            report_content += f"| **{r['engine']}** | **{r['duration']}s** | {r['peak_ram_mb']} MB | ~{throughput}M rows/s | Out-of-Core Arrow / In-Memory |\n"
        else:
            report_content += f"| **{r['engine']}** | N/A ({r.get('error')}) | N/A | N/A | N/A |\n"

    report_content += """
---
### Reproducible Methodology
To execute this benchmark on custom datasets:
```bash
python benchmarks/run_reproducible_benchmark.py --rows 6000000 --limit 50
```
"""
    report_path = os.path.join(os.path.dirname(file_path), 'BENCHMARK_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"[+] Detailed report generated at: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="TitanFrame Multi-Engine Benchmark Tool")
    parser.add_argument("--rows", type=int, default=6000000, help="Number of synthetic dataset rows")
    parser.add_argument("--limit", type=int, default=50, help="TitanFrame RAM limit in MB for out-of-core spill simulation")
    args = parser.parse_args()

    data_file = "lineitem.parquet"
    if not os.path.exists(data_file):
        generate_synthetic_dataset(data_file, args.rows)

    num_rows = args.rows
    print(f"\n=======================================================")
    print(f" TITANFRAME MULTI-ENGINE OUT-OF-CORE BENCHMARK SUITE")
    print(f" Dataset: {data_file} ({num_rows:,} rows)")
    print(f"=======================================================\n")

    results = []
    results.append(benchmark_pandas(data_file))
    results.append(benchmark_polars(data_file))
    results.append(benchmark_duckdb(data_file))
    results.append(benchmark_titanframe(data_file, memory_limit_mb=args.limit))

    print("\n-------------------------------------------------------")
    print(f"{'ENGINE':<30} | {'TIME (s)':<10} | {'PEAK RAM (MB)':<15}")
    print("-------------------------------------------------------")
    for r in results:
        dur_str = f"{r['duration']}s" if r.get('duration') is not None else "N/A"
        ram_str = f"{r['peak_ram_mb']} MB" if r.get('peak_ram_mb') is not None else "N/A"
        print(f"{r['engine']:<30} | {dur_str:<10} | {ram_str:<15}")
    print("-------------------------------------------------------\n")

    generate_markdown_report(results, data_file, num_rows)

if __name__ == "__main__":
    main()
