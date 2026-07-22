"""
GPU Accelerated Analytics Example
=================================
Demonstrates GPU memory configuration and GPU execution.
"""

import titanframe as tf

def main():
    print("=== TitanFrame GPU Analytics ===")
    
    # Configure global GPU device
    tf.config.use_gpu = True
    tf.config.gpu_device_id = 0
    
    print(f"GPU Enabled: {tf.config.use_gpu} | Device ID: {tf.config.gpu_device_id}")
    
    # Run lazy execution on GPU
    df = tf.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    
    res = df.select(
        tf.col("x") * 2 + tf.col("y"),
        tf.col("x").sum().alias("sum_x")
    )
    
    print("\nResult:")
    print(res)

if __name__ == "__main__":
    main()
