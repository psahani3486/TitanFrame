"""
GroupBy Benchmark
=================
Measures high-cardinality aggregation throughput.
"""

import time
import titanframe as tf
import pyarrow as pa

def bench_groupby_performance():
    n = 500_000
    keys = ["A", "B", "C", "D", "E"] * (n // 5)
    table = pa.Table.from_arrays(
        [pa.array(keys), pa.array(list(range(n)))],
        names=["category", "val"]
    )
    df = tf.DataFrame(table)
    
    start = time.perf_counter()
    res = df.group_by("category").agg(tf.col("val").sum().alias("total"))
    elapsed = time.perf_counter() - start
    
    print(f"[BENCH GROUPBY] Processed {n} rows in {elapsed*1000:.2f} ms ({n/elapsed/1e6:.2f} M rows/sec)")
    return elapsed

if __name__ == "__main__":
    bench_groupby_performance()
