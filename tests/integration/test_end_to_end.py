"""Integration tests for end-to-end LazyFrame queries."""

import pytest
import titanframe as tf

def test_e2e_lazy_query(tmp_path):
    df_init = tf.DataFrame({
        "cat": ["A", "B", "A", "C", "B"],
        "val": [10, 20, 30, 40, 50]
    })
    parquet_file = tmp_path / "data.parquet"
    df_init.to_parquet(parquet_file)
    
    result = (
        tf.read_parquet(parquet_file)
        .filter(tf.col("val") >= 20)
        .group_by("cat")
        .agg(tf.col("val").sum().alias("total"))
        .sort("cat")
        .collect()
    )
    
    assert len(result) > 0
    cats = result.to_pandas()["cat"].tolist()
    totals = result.to_pandas()["total"].tolist()
    
    assert "B" in cats
    assert "C" in cats
