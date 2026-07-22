"""
Out-of-Core ETL Example
=======================
Demonstrates memory limits, automatic NVMe spilling, and telemetry dashboard integration.
"""

import time
import titanframe as tf

def main():
    print("=== TitanFrame Out-of-Core ETL Pipeline ===")
    
    # 1. Start live telemetry dashboard
    tf.start_dashboard(port=8080)
    print("Dashboard listening on http://localhost:8080")
    
    # 2. Restrict RAM limit artificially to force NVMe spilling
    tf.config.cpu_memory_limit = 10 * 1024 * 1024  # 10 MB RAM budget
    print(f"Memory Limit set to: {tf.config.cpu_memory_limit / 1024 / 1024:.2f} MB")
    
    # 3. Read dataset lazily
    lf = tf.read_parquet("lineitem.parquet")
    
    # 4. Build query
    query = (
        lf.filter(tf.col("l_discount") > 0.05)
        .group_by("l_returnflag")
        .agg(
            tf.col("l_quantity").sum().alias("sum_qty")
        )
        .sort("l_returnflag")
    )
    
    # Explain DAG plan
    print("\nOptimized Query Plan:")
    query.explain()
    
    # 5. Execute query out-of-core
    print("\nExecuting query out-of-core...")
    res = query.collect()
    print("\nQuery Output:")
    print(res)
    
    print("\nSleeping for 5s to allow dashboard inspection...")
    time.sleep(5)
    tf.stop_dashboard()

if __name__ == "__main__":
    main()
