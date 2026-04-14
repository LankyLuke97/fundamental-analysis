from datetime import datetime

import embar.column.common as common_types
import embar.column.pg as pg_types
from embar.config import EmbarConfig
from embar.table import Table

class Transaction(Table):
    id: pg_types.Serial = pg_types.serial(primary=True)
    ticker: pg_types.Varchar = pg_types.varchar(length=15,not_null=True)
    price: pg_types.Numeric = pg_types.numeric(precision=19,scale=2,not_null=True)
    fees: pg_types.Numeric = pg_types.numeric(precision=19,scale=2,not_null=True)
    currency: pg_types.Varchar = pg_types.varchar(length=3,default='GBP')
    quantity: common_types.Float = common_types.float_col(not_null=True)
    datetime: pg_types.Timestamp = pg_types.timestamp()
    notes: common_types.Text = common_types.text(default='')
