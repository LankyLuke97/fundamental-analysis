import embar.coloumn.common as pg_types
from embar.config import EmbarConfig
from embar.table import Table

class Transaction(Table):
    embar_config: EmbarConfig = EmbarConfig(table_name='transactions')

    id: pg_types.Serial = pg_types.serial(primary=True)
    ticker: pg_types.Varchar = pg_types.varchar(length=15)
    price: pg_types.Numeric = pg_types.numeric(precision=19,scale=2)
    fees: pg_types.Numeric = pg_types.numeric(precision=19,scale=2)
    currency: pg_types.Varchar = pg_types.varchar(length=3)
    quantity: pg_types.Float = pg_types.float_col()
    datetime: pg_types.Timestamp = pg_types.timestamp()
    notes: pg_types.Text = pg_types.text()
