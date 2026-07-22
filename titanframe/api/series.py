from __future__ import annotations
from typing import Any, Optional, Sequence
import pyarrow as pa
import pyarrow.compute as pc
from titanframe.core.column import ChunkedColumn
from titanframe.core.dtypes import DType, from_arrow

class Series:
    __slots__ = ('_column',)

    def __init__(self, name: str, data: ChunkedColumn | pa.ChunkedArray | pa.Array | list | None=None, dtype: Optional[DType]=None):
        if isinstance(data, ChunkedColumn):
            self._column = data
        elif isinstance(data, pa.ChunkedArray):
            self._column = ChunkedColumn.from_pyarrow(name, data)
        elif isinstance(data, pa.Array):
            dt = dtype or from_arrow(data.type)
            self._column = ChunkedColumn(name, dt, [data])
        elif isinstance(data, list):
            if dtype is not None:
                arr = pa.array(data, type=dtype.arrow_type)
                self._column = ChunkedColumn(name, dtype, [arr])
            else:
                arr = pa.array(data)
                dt = from_arrow(arr.type)
                self._column = ChunkedColumn(name, dt, [arr])
        elif data is None:
            from titanframe.core.dtypes import Null
            dt = dtype or Null
            self._column = ChunkedColumn(name, dt)
        else:
            raise TypeError(f'Cannot construct Series from {type(data).__name__}')

    @property
    def name(self) -> str:
        return self._column.name

    @property
    def dtype(self) -> DType:
        return self._column.dtype

    @property
    def num_rows(self) -> int:
        return self._column.num_rows

    @property
    def shape(self) -> tuple[int]:
        return (self.num_rows,)

    @property
    def nbytes(self) -> int:
        return self._column.nbytes

    @property
    def null_count(self) -> int:
        return self._column.null_count

    @property
    def has_nulls(self) -> bool:
        return self._column.has_nulls

    def __len__(self) -> int:
        return self.num_rows

    @property
    def _chunked_column(self) -> ChunkedColumn:
        return self._column

    def _to_chunked_array(self) -> pa.ChunkedArray:
        return self._column.to_pyarrow()

    def _to_combined_array(self) -> pa.Array:
        return self._to_chunked_array().combine_chunks()

    def sum(self) -> Any:
        return pc.sum(self._to_chunked_array()).as_py()

    def mean(self) -> Any:
        return pc.mean(self._to_chunked_array()).as_py()

    def min(self) -> Any:
        return pc.min(self._to_chunked_array()).as_py()

    def max(self) -> Any:
        return pc.max(self._to_chunked_array()).as_py()

    def count(self) -> int:
        return pc.count(self._to_chunked_array()).as_py()

    def std(self, ddof: int=1) -> Any:
        return pc.stddev(self._to_chunked_array(), ddof=ddof).as_py()

    def var(self, ddof: int=1) -> Any:
        return pc.variance(self._to_chunked_array(), ddof=ddof).as_py()

    def median(self) -> Any:
        result = pc.approximate_median(self._to_chunked_array())
        return result.as_py()

    def any(self) -> bool:
        return pc.any(self._to_chunked_array()).as_py()

    def all(self) -> bool:
        return pc.all(self._to_chunked_array()).as_py()

    def nunique(self) -> int:
        return pc.count_distinct(self._to_chunked_array()).as_py()

    def value_counts(self) -> dict[Any, int]:
        result = pc.value_counts(self._to_combined_array())
        return {item['values'].as_py(): item['counts'].as_py() for item in result.to_pylist()}

    def __add__(self, other: Any) -> Series:
        return self._binary_op(pc.add, other)

    def __radd__(self, other: Any) -> Series:
        return self._binary_op(pc.add, other, reverse=True)

    def __sub__(self, other: Any) -> Series:
        return self._binary_op(pc.subtract, other)

    def __rsub__(self, other: Any) -> Series:
        return self._binary_op(pc.subtract, other, reverse=True)

    def __mul__(self, other: Any) -> Series:
        return self._binary_op(pc.multiply, other)

    def __rmul__(self, other: Any) -> Series:
        return self._binary_op(pc.multiply, other, reverse=True)

    def __truediv__(self, other: Any) -> Series:
        return self._binary_op(pc.divide, other)

    def __mod__(self, other: Any) -> Series:
        arr = self._to_combined_array()
        other_arr = self._scalar_or_array(other)
        div_result = pc.divide(arr, other_arr)
        mul_result = pc.multiply(div_result, other_arr)
        result = pc.subtract(arr, mul_result)
        return Series(self.name, result)

    def __neg__(self) -> Series:
        return Series(self.name, pc.negate(self._to_combined_array()))

    def __abs__(self) -> Series:
        return Series(self.name, pc.abs(self._to_combined_array()))

    def __eq__(self, other: Any) -> Series:
        return self._binary_op(pc.equal, other)

    def __ne__(self, other: Any) -> Series:
        return self._binary_op(pc.not_equal, other)

    def __lt__(self, other: Any) -> Series:
        return self._binary_op(pc.less, other)

    def __le__(self, other: Any) -> Series:
        return self._binary_op(pc.less_equal, other)

    def __gt__(self, other: Any) -> Series:
        return self._binary_op(pc.greater, other)

    def __ge__(self, other: Any) -> Series:
        return self._binary_op(pc.greater_equal, other)

    def __and__(self, other: Any) -> Series:
        return self._binary_op(pc.and_, other)

    def __or__(self, other: Any) -> Series:
        return self._binary_op(pc.or_, other)

    def __invert__(self) -> Series:
        return Series(self.name, pc.invert(self._to_combined_array()))

    def is_null(self) -> Series:
        return Series(self.name, pc.is_null(self._to_combined_array()))

    def is_not_null(self) -> Series:
        return Series(self.name, pc.is_valid(self._to_combined_array()))

    def fill_null(self, value: Any) -> Series:
        arr = self._to_combined_array()
        if isinstance(value, Series):
            fill_arr = value._to_combined_array()
            result = pc.if_else(pc.is_null(arr), fill_arr, arr)
        else:
            result = pc.fill_null(arr, pa.scalar(value, type=arr.type))
        return Series(self.name, result)

    def drop_null(self) -> Series:
        return Series(self.name, pc.drop_null(self._to_combined_array()))

    def cast(self, dtype: DType) -> Series:
        result = pc.cast(self._to_combined_array(), dtype.arrow_type)
        return Series(self.name, result)

    def rename(self, name: str) -> Series:
        return Series(name, self._column.rename(name))

    def sort(self, descending: bool=False) -> Series:
        order = 'descending' if descending else 'ascending'
        indices = pc.sort_indices(self._to_combined_array(), sort_keys=[(self.name, order)])
        result = pc.take(self._to_combined_array(), indices)
        return Series(self.name, result)

    def argsort(self, descending: bool=False) -> Series:
        from titanframe.core.dtypes import Int64
        order = 'descending' if descending else 'ascending'
        indices = pc.sort_indices(self._to_combined_array(), sort_keys=[(self.name, order)])
        return Series(f'{self.name}_argsort', indices)

    def unique(self) -> Series:
        result = pc.unique(self._to_combined_array())
        return Series(self.name, result)

    def is_duplicated(self) -> Series:
        vc = pc.value_counts(self._to_combined_array())
        dup_values = set()
        for item in vc.to_pylist():
            if item['counts'].as_py() > 1:
                dup_values.add(item['values'].as_py())
        result = pa.array([v in dup_values for v in self.to_list()])
        return Series(self.name, result)

    def to_list(self) -> list[Any]:
        return self._column.to_pylist()

    def to_pyarrow(self) -> pa.ChunkedArray:
        return self._column.to_pyarrow()

    def to_pandas(self) -> Any:
        return self._to_chunked_array().to_pandas()

    def __getitem__(self, key: Any) -> Any:
        arr = self._to_combined_array()
        if isinstance(key, int):
            return arr[key].as_py()
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                indices = pa.array(range(start, stop, step))
                result = pc.take(arr, indices)
            else:
                result = arr.slice(start, stop - start)
            return Series(self.name, result)
        if isinstance(key, Series):
            mask = key._to_combined_array()
            result = pc.filter(arr, mask)
            return Series(self.name, result)
        raise TypeError(f'Invalid index type: {type(key).__name__}')

    def head(self, n: int=5) -> Series:
        return self[:n]

    def tail(self, n: int=5) -> Series:
        return self[-n:]

    @property
    def str(self) -> SeriesStringAccessor:
        return SeriesStringAccessor(self)

    def _scalar_or_array(self, other: Any) -> Any:
        if isinstance(other, Series):
            return other._to_combined_array()
        return pa.scalar(other)

    def _binary_op(self, fn: Any, other: Any, reverse: bool=False) -> Series:
        arr = self._to_combined_array()
        other_val = self._scalar_or_array(other)
        if reverse:
            result = fn(other_val, arr)
        else:
            result = fn(arr, other_val)
        return Series(self.name, result)

    def __repr__(self) -> str:
        values = self.to_list()
        max_show = 10
        lines = [f'Series({self.name!r}, dtype={self.dtype})']
        for i, v in enumerate(values[:max_show]):
            lines.append(f'  {i}: {v}')
        if len(values) > max_show:
            lines.append(f'  ... ({len(values) - max_show} more)')
        lines.append(f'Length: {len(values)}')
        return '\n'.join(lines)

class SeriesStringAccessor:
    __slots__ = ('_series',)

    def __init__(self, series: Series):
        self._series = series

    def _apply(self, fn: Any, *args: Any) -> Series:
        arr = self._series._to_combined_array()
        result = fn(arr, *args)
        return Series(self._series.name, result)

    def lower(self) -> Series:
        return self._apply(pc.utf8_lower)

    def upper(self) -> Series:
        return self._apply(pc.utf8_upper)

    def length(self) -> Series:
        return self._apply(pc.utf8_length)

    def strip(self) -> Series:
        return self._apply(pc.utf8_trim_whitespace)

    def lstrip(self) -> Series:
        return self._apply(pc.utf8_ltrim_whitespace)

    def rstrip(self) -> Series:
        return self._apply(pc.utf8_rtrim_whitespace)

    def contains(self, pattern: str) -> Series:
        arr = self._series._to_combined_array()
        result = pc.match_substring(arr, pattern)
        return Series(self._series.name, result)

    def starts_with(self, prefix: str) -> Series:
        return self._apply(pc.starts_with, prefix)

    def ends_with(self, suffix: str) -> Series:
        return self._apply(pc.ends_with, suffix)

    def replace(self, pattern: str, replacement: str) -> Series:
        return self._apply(pc.replace_substring, pattern, replacement)

    def slice(self, start: int, length: Optional[int]=None) -> Series:
        if length is not None:
            return self._apply(pc.utf8_slice_codeunits, start, start + length)
        return self._apply(pc.utf8_slice_codeunits, start)

    def is_alpha(self) -> Series:
        return self._apply(pc.utf8_is_alpha)

    def is_numeric(self) -> Series:
        return self._apply(pc.utf8_is_numeric)

    def is_alphanumeric(self) -> Series:
        return self._apply(pc.utf8_is_alnum)

    def reverse(self) -> Series:
        return self._apply(pc.utf8_reverse)
