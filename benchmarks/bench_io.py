import time
import tempfile
import pathlib
import pyarrow as pa
import titanframe as tf

def bench_io_performance():
    n = 100000
    table = pa.Table.from_arrays([pa.array(range(n)), pa.array([f'item_{i}' for i in range(n)])], names=['id', 'name'])
    df = tf.DataFrame(table)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / 'test.parquet'
        start = time.perf_counter()
        df.write_parquet(path)
        write_time = time.perf_counter() - start
        start = time.perf_counter()
        read_df = tf.read_parquet(path).collect()
        read_time = time.perf_counter() - start
        print(f'[BENCH IO] Parquet Write: {write_time * 1000:.2f} ms | Parquet Read: {read_time * 1000:.2f} ms')
if __name__ == '__main__':
    bench_io_performance()
