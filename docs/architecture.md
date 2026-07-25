# TitanFrame Technical Architecture & System Design

**TitanFrame** is an out-of-core, vector data analytics engine designed to process multi-gigabyte datasets with minimal memory overhead using Apache Arrow IPC streaming and DuckDB execution kernels.

---

## High-Level System Architecture

```
                                  +---------------------------------------+
                                  |     User-Facing API & SQL Studio      |
                                  |   (tf.DataFrame / SQL Monaco Editor)  |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |     Logical Plan Builder (DAG)        |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |  Rule-Based Query Optimizer           |
                                  |  - Predicate Pushdown                 |
                                  |  - Projection Pruning                 |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  | Physical Operator Execution Tree      |
                                  | (ScanExec -> FilterExec -> HashAgg)   |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |  Hierarchical Out-of-Core Spill Pool  |
                                  |  (Host RAM 65.5K Chunks -> NVMe Spill)|
                                  +---------------------------------------+
```

---

## Core Components

1. **Logical Plan DAG**:
   - Represents the high-level data transformation pipeline (`Scan`, `Filter`, `Select`, `Aggregate`, `Sort`).
2. **Rule-Based Optimizer**:
   - Pushes filter predicates down into Parquet and CSV readers (`predicate_pushdown.py`).
   - Eliminates unreferenced column schema attributes (`projection_pushdown.py`).
3. **Physical Streaming Operators**:
   - Executes queries in streaming 65,536-row `RecordBatch` chunks using Apache Arrow zero-copy memory arrays.
4. **NVMe Out-of-Core Memory Manager**:
   - Monitors RAM allocation against configured memory ceilings (e.g. 50 MB RAM cap).
   - Serializes intermediate batch buffers to disk using LZ4 Arrow IPC format when memory thresholds are reached.
5. **Unified Telemetry Source of Truth**:
   - Real-time backend `/api/engine/status` endpoint powering the React Web Studio with operator latency profiling and physical DAG visualization.
