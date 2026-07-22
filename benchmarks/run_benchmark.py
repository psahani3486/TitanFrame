import time
import pandas as pd
import titanframe as tf
import os

def run_pandas_benchmark():
    print("Running Pandas Benchmark...")
    start = time.time()
    
    df = pd.read_parquet("lineitem.parquet")
    
    res = (
        df[df["l_discount"] > 0.05]
        .groupby("l_returnflag")
        .agg(
            sum_qty=("l_quantity", "sum"),
            sum_price=("l_extendedprice", "sum")
        )
        .sort_index()
    )
    
    end = time.time()
    print(res)
    return end - start

def run_titanframe_benchmark():
    print("Running TitanFrame Benchmark...")
    start = time.time()
    
    lf = tf.read_parquet("lineitem.parquet")
    
    res = (
        lf.filter(tf.col("l_discount") > 0.05)
          .group_by("l_returnflag")
          .agg(
              tf.col("l_quantity").sum().alias("sum_qty"),
              tf.col("l_extendedprice").sum().alias("sum_price")
          )
          .sort("l_returnflag")
          .collect()
    )
    
    end = time.time()
    print(res)
    return end - start

if __name__ == "__main__":
    if not os.path.exists("lineitem.parquet"):
        print("Please run generate_data.py first!")
        exit(1)
        
    print("=== TPC-H Q1 Inspired Benchmark ===")
    
    pd_time = run_pandas_benchmark()
    print(f"\nPandas Time: {pd_time:.3f}s\n")
    
    tf_time = run_titanframe_benchmark()
    print(f"\nTitanFrame Time: {tf_time:.3f}s\n")
    
    speedup = pd_time / tf_time
    print(f"TitanFrame is {speedup:.2f}x faster!")
