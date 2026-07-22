"""
TitanFrame eCommerce Dataset Query Script
==========================================
Out-of-core, streaming CSV analytics on 14.6 GB dataset:
- dataset/2019-Oct.csv (5.67 GB)
- dataset/2019-Nov.csv (9.01 GB)

This script demonstrates TitanFrame's lazy evaluation, query optimization,
out-of-core chunked streaming, and NVMe spilling capabilities.
"""

import time
import os
import titanframe as tf

def run_top_revenue_brands(dataset_path: str = "dataset/2019-Oct.csv"):
    """
    Query 1: Top Revenue Brands Analysis
    Filters purchase events, groups by brand, calculates total revenue, purchase count, 
    and average price, then sorts by total revenue descending.
    """
    print(f"\n--- Running Query 1: Top Revenue Brands on {dataset_path} ---")
    start_t = time.time()
    
    lf = tf.scan_csv(dataset_path)
    
    query = (
        lf.filter(tf.col("event_type") == "purchase")
          .filter(tf.col("brand").is_not_null())
          .group_by("brand")
          .agg(
              tf.col("price").sum().alias("total_revenue"),
              tf.col("price").count().alias("purchase_count"),
              tf.col("price").mean().alias("avg_item_price")
          )
          .sort("total_revenue", descending=True)
          .head(20)
    )
    
    result_df = query.collect()
    duration = time.time() - start_t
    
    print(f"Query completed in {duration:.3f} seconds.")
    print(result_df)
    return result_df

def run_category_funnel(dataset_path: str = "dataset/2019-Oct.csv"):
    """
    Query 2: Category Sales Funnel Analytics
    Groups by category_code and event_type to measure user conversion volume.
    """
    print(f"\n--- Running Query 2: Category Sales Funnel on {dataset_path} ---")
    start_t = time.time()
    
    lf = tf.scan_csv(dataset_path)
    
    query = (
        lf.filter(tf.col("category_code").is_not_null())
          .group_by("category_code")
          .agg(
              tf.col("price").count().alias("total_events"),
              tf.col("price").sum().alias("total_event_value"),
              tf.col("price").mean().alias("avg_event_value")
          )
          .sort("total_events", descending=True)
          .head(20)
    )
    
    result_df = query.collect()
    duration = time.time() - start_t
    
    print(f"Query completed in {duration:.3f} seconds.")
    print(result_df)
    return result_df

def run_high_value_products(dataset_path: str = "dataset/2019-Oct.csv"):
    """
    Query 3: High-Value Product Leaderboard (Price > $500)
    Filters expensive luxury items and computes sales performance.
    """
    print(f"\n--- Running Query 3: High-Value Products (> $500) on {dataset_path} ---")
    start_t = time.time()
    
    lf = tf.scan_csv(dataset_path)
    
    query = (
        lf.filter(tf.col("price") > 500.0)
          .filter(tf.col("event_type") == "purchase")
          .group_by("product_id")
          .agg(
              tf.col("price").sum().alias("total_revenue"),
              tf.col("price").count().alias("purchases_count"),
              tf.col("price").max().alias("max_price")
          )
          .sort("total_revenue", descending=True)
          .head(20)
    )
    
    result_df = query.collect()
    duration = time.time() - start_t
    
    print(f"Query completed in {duration:.3f} seconds.")
    print(result_df)
    return result_df

def main():
    print("=" * 60)
    print("       TitanFrame eCommerce Analytics Execution Engine      ")
    print("=" * 60)
    
    oct_path = "dataset/2019-Oct.csv"
    nov_path = "dataset/2019-Nov.csv"
    
    if os.path.exists(oct_path):
        run_top_revenue_brands(oct_path)
        run_category_funnel(oct_path)
        run_high_value_products(oct_path)
    else:
        print(f"File {oct_path} not found.")

    if os.path.exists(nov_path):
        run_top_revenue_brands(nov_path)

if __name__ == "__main__":
    main()
