import time
import titanframe as tf

def test_dashboard():
    tf.start_dashboard(port=8080)
    lf = tf.read_parquet('lineitem.parquet')
    tf.config.cpu_memory_limit = 10 * 1024 * 1024
    print('Running query...')
    try:
        res = lf.filter(tf.col('l_discount') > 0.05).group_by('l_returnflag').agg(tf.col('l_quantity').sum().alias('sum_qty')).sort('l_returnflag').collect()
        print('Query finished!', flush=True)
        print(res, flush=True)
    except Exception as e:
        print(f'Error during query: {e}', flush=True)
    print('Dashboard is still running. Visit http://localhost:8080', flush=True)
    print('Keeping the server alive for 1 hour so you can check the UI...', flush=True)
    for _ in range(3600):
        time.sleep(1)
    tf.stop_dashboard()
if __name__ == '__main__':
    test_dashboard()
