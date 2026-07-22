# TitanFrame API Reference

Welcome to the **TitanFrame API Reference**. TitanFrame provides both Eager (`DataFrame`) and Lazy (`LazyFrame`) computation models for high-throughput, GPU-accelerated, out-of-core data analytics.

---

## 1. Top-Level Entry Points (`titanframe`)

### `titanframe.read_parquet(source, columns=None, predicate=None) -> LazyFrame`
Reads a Parquet file or directory deferred for execution. Supports predicate pushdown and projection pushdown.

### `titanframe.read_csv(source, has_header=True) -> LazyFrame`
Chunked lazy CSV reader.

### `titanframe.read_ipc(source) -> LazyFrame`
Reads Arrow IPC streaming format files.

### `titanframe.read_json(source) -> LazyFrame`
Reads newline-delimited JSON (NDJSON) or standard JSON datasets.

### `titanframe.DataFrame(data) -> DataFrame`
Constructs an eager DataFrame backed by an Apache Arrow `Table` or `RecordBatch`.

### `titanframe.col(name) -> ColumnExpr`
Constructs a column expression representing a field in the DataFrame schema.

### `titanframe.lit(value) -> LiteralExpr`
Wraps a Python literal value into an expression.

### `titanframe.start_dashboard(port=8000)`
Launches the real-time background telemetry dashboard.

---

## 2. DataFrame (Eager API)

- `.select(*exprs)`: Project expressions onto new columns.
- `.filter(expr)`: Filter rows based on a boolean expression.
- `.with_columns(*exprs)`: Append or overwrite column values.
- `.group_by(*keys)`: Proxy for aggregation groupings.
- `.join(other, on=None, how='inner')`: Perform hash-joins across tables.
- `.sort(by, descending=False)`: Sort rows using radix/merge execution.
- `.head(n=5)`: Slice the first `n` rows.
- `.to_pandas()`: Zero-copy conversion to a Pandas DataFrame.

---

## 3. LazyFrame (Deferred Execution)

- `.select(*exprs)` -> `LazyFrame`
- `.filter(expr)` -> `LazyFrame`
- `.with_columns(*exprs)` -> `LazyFrame`
- `.group_by(*keys)` -> `LazyGroupBy`
- `.join(other, on, how='inner')` -> `LazyFrame`
- `.sort(by, descending=False)` -> `LazyFrame`
- `.collect()` -> `DataFrame` (Triggers optimizer, physical planner, and execution scheduler)
- `.explain(optimized=True)` -> Displays the execution DAG plan tree.
