"""
TitanFrame Quickstart Example
=============================
Demonstrates basic DataFrame creation, filtering, and aggregation.
"""

import titanframe as tf

def main():
    print("=== TitanFrame Quickstart ===")
    
    # Create eager DataFrame
    df = tf.DataFrame({
        "employee_id": [101, 102, 103, 104, 105],
        "department": ["Eng", "Sales", "Eng", "HR", "Sales"],
        "salary": [120000, 95000, 140000, 80000, 110000]
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Filter and Aggregate
    res = (
        df.filter(tf.col("salary") >= 90000)
        .group_by("department")
        .agg(
            tf.col("salary").mean().alias("avg_salary"),
            tf.col("employee_id").count().alias("headcount")
        )
    )
    
    print("\nAggregated Results:")
    print(res)

if __name__ == "__main__":
    main()
