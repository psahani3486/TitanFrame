# Quickstart Tutorial

This guide covers getting started with **TitanFrame** for fast, memory-efficient data processing.

## 1. Installation

```bash
pip install titanframe
```

## 2. Basic Eager Workflow

```python
import titanframe as tf

# Construct from dictionary or Arrow table
df = tf.DataFrame({"a": [1, 2, 3, 4], "b": [10.0, 20.0, 30.0, 40.0]})

# Filter & Project
result = df.filter(tf.col("a") > 2).select(tf.col("b") * 2)
print(result)
```

## 3. Out-of-Core Lazy Queries

For datasets larger than RAM, use `LazyFrame`:

```python
import titanframe as tf

# Launch background dashboard for live telemetry
tf.start_dashboard(port=8080)

# Build lazy query DAG
query = (
    tf.read_parquet("large_dataset.parquet")
    .filter(tf.col("discount") > 0.05)
    .group_by("shipmode")
    .agg(tf.col("quantity").sum().alias("total_qty"))
    .sort("total_qty", descending=True)
)

# Inspect optimized query execution plan
query.explain()

# Collect results (Spills automatically to NVMe if RAM limit is hit)
df_result = query.collect()
print(df_result)
```
