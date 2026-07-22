import time
import titanframe as tf

def main():
    print('=== TitanFrame Out-of-Core ETL Pipeline ===')
    tf.start_dashboard(port=8080)
    print('Dashboard listening on http://localhost:8080')
    tf.config.cpu_memory_limit = 10 * 1024 * 1024
    print(f'Memory Limit set to: {tf.config.cpu_memory_limit / 1024 / 1024:.2f} MB')
    lf = tf.read_parquet('lineitem.parquet')
    query = lf.filter(tf.col('l_discount') > 0.05).group_by('l_returnflag').agg(tf.col('l_quantity').sum().alias('sum_qty')).sort('l_returnflag')
    print('\nOptimized Query Plan:')
    query.explain()
    print('\nExecuting query out-of-core...')
    res = query.collect()
    print('\nQuery Output:')
    print(res)
    print('\nSleeping for 5s to allow dashboard inspection...')
    time.sleep(5)
    tf.stop_dashboard()
if __name__ == '__main__':
    main()
