import pytest
import pandas as pd
import numpy as np
import pyarrow as pa
from titanframe.api.dataframe import DataFrame
from titanframe.expr.column_expr import col
from titanframe.expr.literal_expr import lit

@pytest.fixture
def pandas_df():
    return pd.DataFrame({'A': [1, 2, 3, 4, None], 'B': [10.5, 20.0, None, 40.2, 50.1], 'C': ['foo', 'bar', 'baz', None, 'qux']})

@pytest.fixture
def tf_df(pandas_df):
    return DataFrame(pandas_df)

def test_compat_select_filter(pandas_df, tf_df):
    pd_res = pandas_df[pandas_df['A'] > 2][['B', 'C']].reset_index(drop=True)
    tf_res = tf_df.filter(col('A') > 2).select('B', 'C').to_pandas()
    pd.testing.assert_frame_equal(pd_res, tf_res)

def test_compat_fillna(pandas_df, tf_df):
    pd_res = pandas_df.fillna(0).reset_index(drop=True)
    pd_res_num = pandas_df[['A', 'B']].fillna(0).reset_index(drop=True)
    tf_res_num = tf_df.select('A', 'B').fillna(0).to_pandas()
    pd.testing.assert_frame_equal(pd_res_num, tf_res_num, check_dtype=False)

def test_compat_dropna(pandas_df, tf_df):
    pd_res = pandas_df.dropna().reset_index(drop=True)
    tf_res = tf_df.dropna().to_pandas()
    pd.testing.assert_frame_equal(pd_res, tf_res)

def test_compat_concat(pandas_df, tf_df):
    pd_res = pd.concat([pandas_df, pandas_df]).reset_index(drop=True)
    tf_res = tf_df.concat(tf_df).to_pandas()
    pd.testing.assert_frame_equal(pd_res, tf_res)

def test_compat_merge(pandas_df, tf_df):
    df2_pd = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'D': ['x', 'y', 'z']})
    df2_tf = DataFrame(df2_pd)
    pd_res = pd.merge(pandas_df, df2_pd, on='A', how='inner').reset_index(drop=True)
    tf_res = tf_df.merge(df2_tf, on=['A'], how='inner').to_pandas()
    tf_res = tf_res[pd_res.columns]
    pd.testing.assert_frame_equal(pd_res, tf_res)
