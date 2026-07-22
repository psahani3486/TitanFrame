import pytest
import pyarrow as pa
from titanframe.core.dtypes import Int32, Int64, Float64, Utf8, Bool, Null
from titanframe.core.schema import Schema, SchemaError

class TestSchemaConstruction:

    def test_from_dict(self):
        s = Schema({'a': Int32, 'b': Utf8})
        assert s.names == ['a', 'b']
        assert s.dtypes == [Int32, Utf8]

    def test_from_tuples(self):
        s = Schema([('x', Float64), ('y', Int64)])
        assert s.names == ['x', 'y']

    def test_empty(self):
        s = Schema()
        assert s.num_columns == 0
        assert s.names == []

    def test_preserves_order(self):
        s = Schema({'z': Int32, 'a': Utf8, 'm': Float64})
        assert s.names == ['z', 'a', 'm']

class TestSchemaProperties:

    @pytest.fixture
    def schema(self):
        return Schema({'id': Int64, 'name': Utf8, 'score': Float64})

    def test_num_columns(self, schema: Schema):
        assert schema.num_columns == 3

    def test_getitem(self, schema: Schema):
        assert schema['id'] == Int64
        assert schema['name'] == Utf8

    def test_getitem_missing_raises(self, schema: Schema):
        with pytest.raises(SchemaError, match='not found'):
            _ = schema['missing']

    def test_contains(self, schema: Schema):
        assert 'id' in schema
        assert 'missing' not in schema

    def test_len(self, schema: Schema):
        assert len(schema) == 3

    def test_iter(self, schema: Schema):
        assert list(schema) == ['id', 'name', 'score']

    def test_index(self, schema: Schema):
        assert schema.index('id') == 0
        assert schema.index('name') == 1
        assert schema.index('score') == 2

    def test_index_missing_raises(self, schema: Schema):
        with pytest.raises(SchemaError, match='not found'):
            schema.index('missing')

    def test_field(self, schema: Schema):
        assert schema.field(0) == ('id', Int64)
        assert schema.field(2) == ('score', Float64)

    def test_field_out_of_range(self, schema: Schema):
        with pytest.raises(IndexError):
            schema.field(10)

class TestSchemaAlgebra:

    @pytest.fixture
    def left(self):
        return Schema({'id': Int64, 'name': Utf8, 'score': Float64})

    @pytest.fixture
    def right(self):
        return Schema({'id': Int64, 'grade': Utf8, 'score': Float64})

    def test_select(self, left: Schema):
        result = left.select(['id', 'score'])
        assert result.names == ['id', 'score']
        assert result['id'] == Int64

    def test_select_missing_raises(self, left: Schema):
        with pytest.raises(SchemaError, match='Cannot select'):
            left.select(['id', 'nonexistent'])

    def test_drop(self, left: Schema):
        result = left.drop(['name'])
        assert result.names == ['id', 'score']

    def test_drop_nonexistent_is_ok(self, left: Schema):
        result = left.drop(['nonexistent'])
        assert result == left

    def test_rename(self, left: Schema):
        result = left.rename({'name': 'full_name', 'score': 'points'})
        assert result.names == ['id', 'full_name', 'points']

    def test_rename_collision_raises(self, left: Schema):
        with pytest.raises(SchemaError, match='collision'):
            left.rename({'name': 'id'})

    def test_merge_no_collision(self):
        a = Schema({'x': Int32})
        b = Schema({'y': Float64})
        result = a.merge(b)
        assert result.names == ['x', 'y']

    def test_merge_with_collision_adds_suffix(self, left: Schema, right: Schema):
        result = left.merge(right, suffix='_right')
        assert 'id_right' in result
        assert 'score_right' in result
        assert result['id'] == Int64
        assert result['id_right'] == Int64

    def test_intersect(self, left: Schema, right: Schema):
        result = left.intersect(right)
        assert 'id' in result
        assert 'score' in result
        assert 'name' not in result

    def test_diff(self, left: Schema, right: Schema):
        result = left.diff(right)
        assert result.names == ['name']

    def test_append(self, left: Schema):
        result = left.append('active', Bool)
        assert result.num_columns == 4
        assert result['active'] == Bool

    def test_append_duplicate_raises(self, left: Schema):
        with pytest.raises(SchemaError, match='already exists'):
            left.append('id', Int32)

    def test_with_column_replaces(self, left: Schema):
        result = left.with_column('score', Int32)
        assert result['score'] == Int32

    def test_with_column_adds_new(self, left: Schema):
        result = left.with_column('new_col', Bool)
        assert result['new_col'] == Bool

class TestSchemaValidation:

    def test_is_compatible_same(self):
        s = Schema({'a': Int32, 'b': Utf8})
        assert s.is_compatible(s)

    def test_is_compatible_different(self):
        a = Schema({'a': Int32})
        b = Schema({'a': Float64})
        assert not a.is_compatible(b)

    def test_is_subset(self):
        full = Schema({'a': Int32, 'b': Utf8, 'c': Float64})
        subset = Schema({'a': Int32, 'c': Float64})
        assert subset.is_subset_of(full)
        assert not full.is_subset_of(subset)

    def test_assert_compatible_passes(self):
        s = Schema({'a': Int32})
        s.assert_compatible(s)

    def test_assert_compatible_different_count(self):
        a = Schema({'a': Int32})
        b = Schema({'a': Int32, 'b': Utf8})
        with pytest.raises(SchemaError, match='left has 1 columns, right has 2'):
            a.assert_compatible(b)

    def test_assert_compatible_different_names(self):
        a = Schema({'a': Int32})
        b = Schema({'b': Int32})
        with pytest.raises(SchemaError, match='column name'):
            a.assert_compatible(b, context='test')

    def test_assert_compatible_different_types(self):
        a = Schema({'a': Int32})
        b = Schema({'a': Float64})
        with pytest.raises(SchemaError, match='type'):
            a.assert_compatible(b)

class TestSchemaArrowInterop:

    def test_round_trip(self):
        original = Schema({'id': Int64, 'name': Utf8, 'value': Float64})
        arrow_schema = original.to_arrow()
        restored = Schema.from_arrow(arrow_schema)
        assert original == restored

    def test_to_arrow_types(self):
        s = Schema({'x': Int32, 'y': Bool})
        arrow_s = s.to_arrow()
        assert arrow_s.field('x').type == pa.int32()
        assert arrow_s.field('y').type == pa.bool_()

class TestSchemaInference:

    def test_from_dict(self):
        data = {'id': [1, 2, 3], 'name': ['a', 'b', 'c']}
        s = Schema.from_dict(data)
        assert s['id'] == Int64
        assert s['name'] == Utf8

    def test_from_dict_with_nulls(self):
        data = {'x': [None, None, 1]}
        s = Schema.from_dict(data)
        assert s['x'] == Int64

    def test_from_dict_all_nulls(self):
        data = {'x': [None, None]}
        s = Schema.from_dict(data)
        assert s['x'] == Null

class TestSchemaEquality:

    def test_equal_schemas(self):
        a = Schema({'x': Int32, 'y': Utf8})
        b = Schema({'x': Int32, 'y': Utf8})
        assert a == b
        assert hash(a) == hash(b)

    def test_unequal_schemas(self):
        a = Schema({'x': Int32})
        b = Schema({'x': Float64})
        assert a != b

    def test_different_order_is_unequal(self):
        a = Schema({'x': Int32, 'y': Utf8})
        b = Schema({'y': Utf8, 'x': Int32})
        assert a != b

    def test_repr(self):
        s = Schema({'a': Int32})
        assert 'Int32' in repr(s)
        assert "'a'" in repr(s)
