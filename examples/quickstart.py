"""
TitanFrame Quickstart Example Script
Demonstrates Lazy Query Planning, Out-of-Core Execution, and DuckDB SQL Analytics.
"""

import titanframe as tf

def main():
    print("=== TitanFrame Quickstart Example ===")
    
    # 1. Eager DataFrame Construction
    df = tf.DataFrame({
        "brand": ["Apple", "Samsung", "Apple", "Xiaomi", "Samsung", "Apple"],
        "price": [999.0, 799.0, 1199.0, 299.0, 899.0, 1099.0],
        "quantity": [2, 5, 1, 10, 3, 4]
    })
    
    print("\n[+] Eager DataFrame Filter & Aggregation:")
    res_eager = df.filter(tf.col("price") > 500.0).group_by("brand").agg(tf.col("price").sum().alias("total_revenue"))
    print(res_eager.to_pandas())

    # 2. Out-of-Core Lazy Query DAG Execution
    print("\n[+] Out-of-Core Lazy Parquet Scan & Pushdown Query:")
    tf.config.cpu_memory_limit = 50 * 1024 * 1024  # 50 MB RAM Cap
    
    query = (
        tf.scan_parquet("lineitem.parquet")
          .filter(tf.col("l_discount") > 0.05)
          .select("l_returnflag", "l_quantity", "l_extendedprice")
          .group_by("l_returnflag")
          .agg(tf.col("l_quantity").sum().alias("sum_qty"))
          .sort("sum_qty", descending=True)
    )
    
    # Collect results
    res_lazy = query.collect()
    print(res_lazy.to_pandas())

if __name__ == "__main__":
    main()
