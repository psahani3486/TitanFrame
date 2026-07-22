"""
Join Benchmark
==============
Measures Hash Join throughput.
"""

import time
import titanframe as tf
import pyarrow as pa

def bench_join_performance():
    n = 200_000
    t1 = pa.Table.from_arrays([pa.array(range(n)), pa.array(range(n))], names=["id", "val1"])
    t2 = pa.Table.from_arrays([pa.array(range(0, n, 2)), pa.array(range(n // 2))], names=["id", "val2"])
    
    df1 = tf.DataFrame(t1)
    df2 = tf.DataFrame(t2)
    
    start = time.perf_counter()
    res = df1.join(df2, on="id", how="inner")
    elapsed = time.perf_counter() - start
    
    print(f"[BENCH JOIN] Joined {n} rows in {elapsed*1000:.2f} ms")
    return elapsed

if __name__ == "__main__":
    bench_join_performance()
