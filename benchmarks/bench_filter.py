"""
Filter Benchmark
================
Measures vector filter throughput.
"""

import time
import titanframe as tf
import pyarrow as pa

def bench_filter_performance():
    n = 1_000_000
    table = pa.Table.from_arrays(
        [pa.array(list(range(n))), pa.array([i * 0.1 for i in range(n)])],
        names=["id", "val"]
    )
    df = tf.DataFrame(table)
    
    start = time.perf_counter()
    filtered = df.filter(tf.col("val") > 50000.0)
    elapsed = time.perf_counter() - start
    
    print(f"[BENCH FILTER] Processed {n} rows in {elapsed*1000:.2f} ms ({n/elapsed/1e6:.2f} M rows/sec)")
    return elapsed

if __name__ == "__main__":
    bench_filter_performance()
