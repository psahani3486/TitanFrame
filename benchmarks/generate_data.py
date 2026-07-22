import numpy as np
import pandas as pd
import os

NUM_ROWS = 2_000_000

def generate():
    print(f"Generating {NUM_ROWS} rows of synthetic TPC-H lineitem data...")
    np.random.seed(42)
    
    orderkey = np.random.randint(1, 500000, size=NUM_ROWS)
    partkey = np.random.randint(1, 200000, size=NUM_ROWS)
    quantity = np.random.randint(1, 50, size=NUM_ROWS)
    extendedprice = quantity * np.random.uniform(1.0, 100.0, size=NUM_ROWS)
    discount = np.random.uniform(0.0, 0.1, size=NUM_ROWS)
    tax = np.random.uniform(0.0, 0.08, size=NUM_ROWS)
    
    returnflag = np.random.choice(['A', 'N', 'R'], size=NUM_ROWS)
    
    df = pd.DataFrame({
        "l_orderkey": orderkey,
        "l_partkey": partkey,
        "l_quantity": quantity,
        "l_extendedprice": extendedprice,
        "l_discount": discount,
        "l_tax": tax,
        "l_returnflag": returnflag
    })
    
    print("Saving to lineitem.parquet...")
    df.to_parquet("lineitem.parquet", engine="pyarrow")
    
    print("Done!")

if __name__ == "__main__":
    generate()
