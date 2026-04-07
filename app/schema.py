import embar.column.common as common_types
import embar.column.pg as pg_types
from embar.config import EmbarConfig
from embar.table import Table

class Transaction(Table):
    embar_config: EmbarConfig = EmbarConfig(table_name='transactions')

    id: pg_types.Serial = pg_types.serial(primary=True)
    ticker: pg_types.Varchar = pg_types.varchar(length=15)
    price: pg_types.Numeric = pg_types.numeric(precision=19,scale=2)
    fees: pg_types.Numeric = pg_types.numeric(precision=19,scale=2)
    currency: pg_types.Varchar = pg_types.varchar(length=3)
    quantity: common_types.Float = common_types.float_col()
    datetime: pg_types.Timestamp = pg_types.timestamp()
    notes: common_types.Text = common_types.text()
