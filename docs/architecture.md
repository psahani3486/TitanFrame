# TitanFrame Architecture

TitanFrame processes data using a 5-phase execution model, inspired by modern query engines like Apache Spark and Polars.

## 1. Lazy Frontend (API Layer)
When users call operations like `lf.filter()`, `lf.select()`, or `lf.group_by()` on a `LazyFrame`, no data is actually processed. Instead, the API builds a **Logical Plan**—a tree of operations describing *what* should be done (e.g. `Projection`, `Filter`, `Aggregation`).

## 2. Query Optimizer
Before execution, the `QueryOptimizer` traverses the Logical Plan and applies a series of heuristic rules:
- **Predicate Pushdown**: Moves `Filter` operations down the tree as close to the data source (`Scan`) as possible, minimizing the amount of data read.
- **Projection Pushdown**: Drops unused columns early in the pipeline.
- **Constant Folding**: Evaluates constant expressions (e.g. `1 + 2`) at compile-time instead of row-by-row during execution.

## 3. Physical Planner
The `PhysicalPlanner` translates the optimized Logical Plan into a **Physical Plan**. 
While Logical Nodes describe *what* to do, Physical Nodes describe *how* to do it. For example, a logical `Join` node might be translated into a `HashJoinExec` or `SortMergeExec` physical node depending on the data. 

All Physical Nodes implement a streaming `execute()` method that yields `Chunk`s of Arrow data.

## 4. Execution Engine (DAG Scheduler)
The `DAGScheduler` coordinates the execution of the Physical Plan. It manages the flow of `Chunk`s through the pipeline, tracks progress via `tqdm`, and handles thread-pool allocation (when multi-threading is enabled). Data is processed in small batches (e.g. 64K rows) to keep memory usage low.

## 5. Memory Manager
TitanFrame is designed for out-of-core execution. The `MemoryManager` monitors RAM usage during execution. If memory pressure becomes too high (e.g., during a massive `HashJoin`), it will transparently spill data to NVMe storage and stream it back when needed, preventing Out-Of-Memory (OOM) crashes.
