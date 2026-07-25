"""
TitanFrame 20 GB Dataset Out-of-Core Execution Script
Executes out-of-core streaming queries over the 20.17 GB dataset (180.9 Million rows) under a strict 50 MB RAM ceiling.
"""

import os
import sys
import time
import titanframe as tf

def main():
    dataset_path = os.path.join("dataset", "2019-Dec-20GB.csv")
    
    if not os.path.exists(dataset_path):
        print(f"[-] File not found: {dataset_path}")
        print("    Run: python tools/generate_20gb_dataset.py --size-gb 20 --output dataset/2019-Dec-20GB.csv")
        return
        
    file_size_gb = round(os.path.getsize(dataset_path) / (1024**3), 2)
    print(f"=======================================================")
    print(f" TITANFRAME 20 GB OUT-OF-CORE STREAMING ENGINE")
    print(f" Dataset File: {dataset_path} ({file_size_gb} GB)")
    print(f" Host RAM Limit: 50 MB Cap (Strict Out-of-Core Spilling)")
    print(f"=======================================================\n")
    
    # Configure strict memory ceiling
    tf.config.cpu_memory_limit = 50 * 1024 * 1024  # 50 MB RAM limit
    tf.config.spill_threshold = 0.85                # Trigger spill at 85% RAM usage
    
    t0 = time.time()
    print("[*] Building Lazy Execution Plan (Predicate Pushdown + Projection Pruning)...")
    
    query = (
        tf.scan_csv(dataset_path)
          .filter(tf.col("event_type") == "purchase")
          .select("brand", "price")
          .group_by("brand")
          .agg(
              tf.col("price").sum().alias("total_revenue"),
              tf.col("price").count().alias("purchase_count"),
              tf.col("price").mean().alias("avg_price")
          )
          .sort("total_revenue", descending=True)
          .head(10)
    )
    
    print("[*] Executing Out-of-Core Arrow IPC Streaming Pipeline...")
    res = query.collect()
    duration = time.time() - t0
    
    print(f"\n[+] Query Execution Completed in {duration:.2f} seconds!")
    print("\n--- Top 10 Revenue Brands (20.17 GB Out-of-Core Dataset) ---")
    print(res.to_pandas().to_string(index=False))
    print("-------------------------------------------------------------")

if __name__ == "__main__":
    main()
