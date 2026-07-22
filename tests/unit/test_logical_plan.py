import pytest
from titanframe.core.dtypes import Int64, Utf8, Bool
from titanframe.core.schema import Schema, SchemaError
from titanframe.expr.column_expr import col
from titanframe.expr.literal_expr import lit
from titanframe.plan.logical.scan import Scan, ScanFormat
from titanframe.plan.logical.projection import Projection
from titanframe.plan.logical.filter import Filter
from titanframe.plan.logical.aggregation import Aggregation
from titanframe.plan.logical.join import Join, JoinType
from titanframe.plan.logical.sort import Sort
from titanframe.plan.logical.limit import Limit
from titanframe.plan.logical.distinct import Distinct
from titanframe.plan.logical.union import Union
from titanframe.plan.logical.sink import Sink, SinkFormat

@pytest.fixture
def scan_node():
    schema = Schema({'id': Int64, 'name': Utf8, 'age': Int64, 'active': Bool})
    return Scan('users.parquet', ScanFormat.PARQUET, schema)

def test_scan_node(scan_node):
    assert scan_node.node_name() == 'Scan'
    assert 'users.parquet' in scan_node.node_description()
    assert scan_node.output_schema().num_columns == 4
    assert len(scan_node.children()) == 0

def test_projection_node(scan_node):
    proj = Projection(scan_node, [col('id'), (col('age') + lit(1)).alias('age_next')])
    assert proj.node_name() == 'Projection'
    schema = proj.output_schema()
    assert schema.num_columns == 2
    assert schema.names == ['id', 'age_next']
    children = proj.children()
    assert len(children) == 1
    assert children[0] is scan_node

def test_filter_node(scan_node):
    filt = Filter(scan_node, col('age') > lit(18))
    assert filt.node_name() == 'Filter'
    schema = filt.output_schema()
    assert schema == scan_node.output_schema()
    assert filt.children()[0] is scan_node

def test_aggregation_node(scan_node):
    agg = Aggregation(scan_node, [col('active')], [col('age').sum().alias('total_age')])
    assert agg.node_name() == 'Aggregation'
    schema = agg.output_schema()
    assert schema.names == ['active', 'total_age']
    assert schema['active'] == Bool

def test_join_node(scan_node):
    schema2 = Schema({'id': Int64, 'dept': Utf8})
    scan2 = Scan('departments.parquet', ScanFormat.PARQUET, schema2)
    join = Join(scan_node, scan2, ['id'], 'inner')
    assert join.node_name() == 'Join'
    schema = join.output_schema()
    assert schema.names == ['id', 'name', 'age', 'active', 'dept']
    children = join.children()
    assert len(children) == 2

def test_sort_node(scan_node):
    sort = Sort(scan_node, [col('age').desc()])
    assert sort.node_name() == 'Sort'
    schema = sort.output_schema()
    assert schema == scan_node.output_schema()

def test_limit_node(scan_node):
    limit = Limit(scan_node, 10, offset=5)
    assert limit.node_name() == 'Limit'
    assert 'limit=10' in limit.node_description()
    assert 'offset=5' in limit.node_description()
    schema = limit.output_schema()
    assert schema == scan_node.output_schema()

def test_distinct_node(scan_node):
    dist = Distinct(scan_node, subset=['name'])
    assert dist.node_name() == 'Distinct'
    assert "subset=['name']" in dist.node_description()
    schema = dist.output_schema()
    assert schema == scan_node.output_schema()

def test_union_node(scan_node):
    scan2 = Scan('users2.parquet', ScanFormat.PARQUET, scan_node.output_schema())
    union = Union([scan_node, scan2])
    assert union.node_name() == 'Union'
    assert 'inputs=2' in union.node_description()
    schema = union.output_schema()
    assert schema == scan_node.output_schema()
    assert len(union.children()) == 2

def test_union_schema_mismatch_raises(scan_node):
    schema2 = Schema({'id': Int64, 'name': Utf8})
    scan2 = Scan('users2.parquet', ScanFormat.PARQUET, schema2)
    with pytest.raises(SchemaError, match='Schema mismatch'):
        Union([scan_node, scan2])

def test_sink_node(scan_node):
    sink = Sink(scan_node, 'output.csv', SinkFormat.CSV)
    assert sink.node_name() == 'Sink'
    assert 'output.csv' in sink.node_description()
    schema = sink.output_schema()
    assert schema == scan_node.output_schema()
