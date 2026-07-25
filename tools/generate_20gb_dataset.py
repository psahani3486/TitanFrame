"""
TitanFrame 20 GB Synthetic Dataset Generator CLI Tool
Generates streaming e-commerce / TPC-H lineitem CSV datasets of custom sizes (default: 20 GB).
"""

import os
import sys
import time
import argparse
import random

def generate_20gb_csv(output_path: str, target_size_gb: float = 20.0):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    target_bytes = int(target_size_gb * 1024 * 1024 * 1024)
    
    print(f"=======================================================")
    print(f" TITANFRAME LARGE DATASET GENERATOR")
    print(f" Output File: {output_path}")
    print(f" Target Size: {target_size_gb} GB ({target_bytes:,} bytes)")
    print(f"=======================================================\n")
    
    brands = ['apple', 'samsung', 'xiaomi', 'huawei', 'lg', 'sony', 'lenovo', 'asus', 'oppo', 'vivo']
    categories = [
        'electronics.smartphone',
        'computers.notebook',
        'appliances.kitchen.refrigerators',
        'apparel.shoes',
        'electronics.clocks',
        'appliances.environment.air_conditioner'
    ]
    event_types = ['view', 'view', 'view', 'cart', 'purchase']
    
    t0 = time.time()
    header = "event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session\n"
    
    bytes_written = 0
    row_count = 0
    buffer_chunk_size = 50000  # Write in 50,000 row blocks to keep memory low
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        bytes_written += len(header.encode('utf-8'))
        
        while bytes_written < target_bytes:
            lines = []
            for _ in range(buffer_chunk_size):
                row_count += 1
                day = (row_count % 28) + 1
                hour = (row_count % 24)
                minute = (row_count % 60)
                sec = (row_count % 60)
                etime = f"2019-12-{day:02d} {hour:02d}:{minute:02d}:{sec:02d} UTC"
                etype = random.choice(event_types)
                pid = 1000000 + (row_count % 1000)
                cid = 2053013555631882655 + (row_count % 50)
                ccode = random.choice(categories)
                brand = random.choice(brands)
                price = round(random.uniform(10.0, 1500.0), 2)
                uid = 512000000 + (row_count % 5000)
                session = f"session_{row_count % 20000}"
                
                line = f"{etime},{etype},{pid},{cid},{ccode},{brand},{price},{uid},{session}\n"
                lines.append(line)
            
            chunk_str = "".join(lines)
            chunk_bytes = len(chunk_str.encode('utf-8'))
            f.write(chunk_str)
            bytes_written += chunk_bytes
            
            curr_gb = round(bytes_written / (1024**3), 2)
            elapsed = round(time.time() - t0, 1)
            mb_per_sec = round((bytes_written / (1024**2)) / max(elapsed, 0.1), 1)
            print(f"[*] Progress: {curr_gb:.2f} GB / {target_size_gb} GB ({row_count:,} rows) | Speed: {mb_per_sec} MB/s", end='\r')
            
            if bytes_written >= target_bytes:
                break
                
    total_time = round(time.time() - t0, 2)
    final_gb = round(os.path.getsize(output_path) / (1024**3), 2)
    print(f"\n\n[+] Successfully generated 20 GB dataset!")
    print(f"    - File Path: {output_path}")
    print(f"    - File Size: {final_gb} GB ({os.path.getsize(output_path):,} bytes)")
    print(f"    - Total Rows: {row_count:,} Rows")
    print(f"    - Generation Time: {total_time} seconds\n")

def main():
    parser = argparse.ArgumentParser(description="TitanFrame 20 GB Synthetic Dataset Generator")
    parser.add_argument("--size-gb", type=float, default=20.0, help="Target dataset size in GB (default: 20.0)")
    parser.add_argument("--output", type=str, default="dataset/2019-Dec-20GB.csv", help="Output file path")
    args = parser.parse_args()
    
    generate_20gb_csv(args.output, args.size_gb)

if __name__ == "__main__":
    main()
