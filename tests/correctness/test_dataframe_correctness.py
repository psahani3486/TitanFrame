"""
TitanFrame Correctness & Parity Test Suite
Verifies row-by-row, column-by-column equality against Pandas and Polars.
"""

import pytest
import pandas as pd
import pyarrow as pa
import numpy as np

import titanframe as tf

@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'],
        'price': [10.5, 20.0, 15.2, 5.0, 99.9, 45.0, 12.0, 8.5, 33.1, 50.0],
        'quantity': [2, 5, 1, 10, 3, 4, 2, 8, 6, 1]
    }
    pdf = pd.DataFrame(data)
    tf_df = tf.DataFrame(data)
    return pdf, tf_df, data

class TestDataFrameCorrectness:

    def test_filter_correctness_against_pandas(self, sample_dataframe):
        pdf, tf_df, _ = sample_dataframe
        
        # Filter: price > 15.0
        pd_filtered = pdf[pdf['price'] > 15.0].reset_index(drop=True)
        tf_filtered = tf_df.filter(tf.col('price') > 15.0).to_pandas().reset_index(drop=True)
        
        pd.testing.assert_frame_equal(pd_filtered, tf_filtered)

    def test_select_projection_correctness(self, sample_dataframe):
        pdf, tf_df, _ = sample_dataframe
        
        pd_selected = pdf[['category', 'price']].reset_index(drop=True)
        tf_selected = tf_df.select('category', 'price').to_pandas().reset_index(drop=True)
        
        pd.testing.assert_frame_equal(pd_selected, tf_selected)

    def test_groupby_aggregation_correctness(self, sample_dataframe):
        pdf, tf_df, _ = sample_dataframe
        
        pd_agg = pdf.groupby('category', as_index=False)['price'].sum().sort_values('category').reset_index(drop=True)
        tf_agg = tf_df.group_by('category').agg(tf.col('price').sum().alias('price')).sort('category').to_pandas().reset_index(drop=True)
        
        pd.testing.assert_series_equal(pd_agg['category'], tf_agg['category'])
        np.testing.assert_allclose(pd_agg['price'].values, tf_agg['price'].values, rtol=1e-5)

    def test_polars_parity(self, sample_dataframe):
        try:
            import polars as pl
            pdf, tf_df, data = sample_dataframe
            pl_df = pl.DataFrame(data)
            
            pl_res = pl_df.filter(pl.col('quantity') >= 5).sort('id')
            tf_res = tf_df.filter(tf.col('quantity') >= 5).sort('id').to_pandas()
            
            np.testing.assert_array_equal(pl_res['id'].to_numpy(), tf_res['id'].to_numpy())
            np.testing.assert_array_equal(pl_res['quantity'].to_numpy(), tf_res['quantity'].to_numpy())
        except ImportError:
            pytest.skip("Polars not installed")
