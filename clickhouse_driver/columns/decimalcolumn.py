from decimal import Decimal, localcontext

import numpy as np

from .base import FormatColumn
from .exceptions import ColumnTypeMismatchException
from .intcolumn import Int128Column, Int256Column


class DecimalColumn(FormatColumn):
    py_types = (Decimal, float, int)
    max_precision = None
    int_size = None

    def __init__(self, precision, scale, types_check=False, **kwargs):
        self.precision = precision
        self.scale = scale
        super(DecimalColumn, self).__init__(**kwargs)

        if types_check:
            max_signed_int = (1 << (8 * self.int_size - 1)) - 1

            def check_item(value):
                if value < -max_signed_int or value > max_signed_int:
                    raise ColumnTypeMismatchException(value)

            self.check_item = check_item

    def after_read_items(self, items, nulls_map=None):
        if self.scale >= 1:
            scale = 10 ** self.scale

            if nulls_map is None:
                return tuple(Decimal(item) / scale for item in items)
            else:
                return tuple(
                    (None if is_null else Decimal(items[i]) / scale)
                    for i, is_null in enumerate(nulls_map)
                )
        else:
            if nulls_map is None:
                return tuple(Decimal(item) for item in items)
            else:
                return tuple(
                    (None if is_null else Decimal(items[i]))
                    for i, is_null in enumerate(nulls_map)
                )

    def before_write_items(self, items, nulls_map=None):
        null_value = self.null_value

        if self.scale >= 1:
            scale = 10 ** self.scale

            for i, item in enumerate(items):
                if nulls_map and nulls_map[i]:
                    items[i] = null_value
                else:
                    items[i] = int(Decimal(str(item)) * scale)

        else:
            for i, item in enumerate(items):
                if nulls_map and nulls_map[i]:
                    items[i] = null_value
                else:
                    items[i] = int(Decimal(str(item)))

    def prepare_items(self, items):
        nullable = self.nullable
        null_value = self.null_value

        check_item = self.check_item
        if self.types_check_enabled:
            check_item_type = self.check_item_type
        else:
            check_item_type = False

        if (not self.nullable and not check_item_type and
                not check_item and not self.before_write_items):
            return items

        nulls_map = [False] * len(items) if self.nullable else None
        for i, x in enumerate(items):
            if x is None and nullable:
                nulls_map[i] = True
                x = null_value

            else:
                if check_item_type:
                    check_item_type(x)

                if check_item:
                    check_item(x)

            items[i] = x

        if self.before_write_items:
            self.before_write_items(items, nulls_map=nulls_map)
            if type(items) == np.ndarray:
                # 在 largeint.pyx 中需要进行位运算，因此必须保证 item 类型为 int
                return items.astype(np.int64, copy=False).tolist()

        return items

    # Override default precision to the maximum supported by underlying type.
    def _write_data(self, items, buf):
        with localcontext() as ctx:
            ctx.prec = self.max_precision
            super(DecimalColumn, self)._write_data(items, buf)

    def _read_data(self, n_items, buf, nulls_map=None):
        with localcontext() as ctx:
            ctx.prec = self.max_precision
            return super(DecimalColumn, self)._read_data(
                n_items, buf, nulls_map=nulls_map
            )


class Decimal32Column(DecimalColumn):
    format = 'i'
    max_precision = 9
    int_size = 4


class Decimal64Column(DecimalColumn):
    format = 'q'
    max_precision = 18
    int_size = 8


class Decimal128Column(DecimalColumn, Int128Column):
    max_precision = 38


class Decimal256Column(DecimalColumn, Int256Column):
    max_precision = 76


def create_decimal_column(spec, column_options):
    precision, scale = spec[8:-1].split(',')
    precision, scale = int(precision), int(scale)

    # Maximum precisions for underlying types are:
    # Int32    10**9
    # Int64   10**18
    # Int128  10**38
    # Int256  10**76
    if precision <= 9:
        cls = Decimal32Column
    elif precision <= 18:
        cls = Decimal64Column
    elif precision <= 38:
        cls = Decimal128Column
    else:
        cls = Decimal256Column

    return cls(precision, scale, **column_options)
