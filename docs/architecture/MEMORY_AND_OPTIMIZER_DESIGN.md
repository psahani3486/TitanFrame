# TitanFrame Memory Manager & Query Optimizer Architecture Design Document

## 1. System Overview & Architectural Motivation

TitanFrame is designed as an out-of-core, streaming vector data engine that enables analytical processing over datasets exceeding host memory limits.

Rather than loading entire Parquet or CSV files into contiguous in-memory arrays (the standard eager model), TitanFrame decomposes datasets into streaming **Apache Arrow `RecordBatch` chunks** (default 65,536 rows per batch). This columnar format aligns with CPU cache lines and SIMD vector registers, enabling pipelined query execution with bounded memory footprints.

```
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│  Parquet / CSV File    │ ───► │ ScanExec (Arrow IPC)   │ ───► │ Predicate Filter       │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
                                                                            │
                                                                            ▼
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│  Sink / Results Output │ ◄─── │ HashAggExec Node       │ ◄─── │ NVMe Spill Manager     │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

---

## 2. Memory Manager & Out-of-Core Spill Mechanics

### 2.1 Arrow RecordBatch Chunk Streaming
Memory allocation is governed by `TelemetryTracker` and `Config`. Execution operates on streaming batches rather than full tables:
- **Default Chunk Size**: 65,536 rows (~4 MB to 16 MB per chunk depending on schema width).
- **Zero-Copy Serialization**: Memory buffers leverage PyArrow IPC zero-copy slices to avoid unnecessary array duplication during filtering and projection.

### 2.2 Memory Budgeting & Spill Trigger Threshold
- **Host RAM Budget**: Configured via `tf.config.cpu_memory_limit` (e.g. 50 MB to 512 MB).
- **Spill Trigger**: Monitored via high-watermark allocation ratios:
  $$\text{Memory Usage Ratio} = \frac{\text{Allocated Host RAM}}{\text{Configured Budget}}$$
  When usage crosses `tf.config.spill_threshold` (e.g. $85\%$), the engine automatically serializes intermediate hash aggregation buffers or chunk streams to NVMe disk storage (`~/.titanframe/spill/`).
- **Disk Format**: Spilled chunks are formatted as LZ4-compressed Apache Arrow IPC streams, allowing fast asynchronous re-reading during hash merge steps.

---

## 3. Narrow & Deep Logical Query Optimizer

TitanFrame's query optimizer is a rule-based logical plan rewriter inspired by **Apache Arrow DataFusion** and **Apache Calcite**.

```
Logical Unoptimized Plan ──► [Optimizer Driver] ──► Logical Optimized Plan ──► Physical Execution Tree
```

### Rule 1: Predicate Pushdown (`predicate_pushdown.py`)
Pushes filter expressions down the logical plan tree to the lowest scan node:
- **Benefit**: Prunes non-matching Parquet row groups and CSV chunk batches *before* loading columns into system RAM.
- **Implementation**: Inspects `LogicalFilter` nodes, extracts binary boolean expressions, and merges them into `LogicalScan(predicate=...)`.

### Rule 2: Projection Pruning (`projection_pushdown.py`)
Traverses expressions from top to bottom, computing the exact subset of required schema fields:
- **Benefit**: Eliminates unreferenced columns early in the pipeline, reducing Arrow IPC buffer size by up to $75\%$.
- **Implementation**: Collects required column symbols from aggregations and projections, pruning unreferenced fields from `LogicalScan(columns=[...])`.

### Rule 3: Constant Folding & Expression Simplification (`constant_folding.py`)
Evaluates static literal expressions at plan compile time (e.g. `col("price") * (1.0 - 0.05)` $\rightarrow$ `col("price") * 0.95`).

---

## 4. Physical Execution Operators

1. **`ScanExec`**: Streams columnar batches from Parquet or CSV files with column projection & predicate pushdown filters.
2. **`FilterExec`**: Evaluates boolean predicate masks using SIMD-aligned vector kernels.
3. **`HashAggExec`**: Accumulates group-by keys and aggregations using a parallel hash table accumulator with spill-aware fallback.
4. **`SinkExec`**: Collects top-K sorted results into an Apache Arrow `Table`.
