from app.database_manager import open_connection_pool
from app.schema import Transaction

from datetime import datetime

with open('data/orders.csv', mode='r', encoding='utf-8') as orders:
    order_list = orders.readlines()
    headers = order_list[0].strip().split(',')
    transactions = []
    for order in order_list[1:]:
        const = {k:v for k, v in zip(headers, order.strip().split(','))}
        if const['type'] == 'sell': const['quantity'] *= -1
        del const['type']
        const['datetime'] = datetime.strptime(const['datetime'], '%Y-%m-%d')
        const['id'] = 0
        transactions.append(Transaction(**const))
    with open_connection_pool(connection_string='postgres://myuser:mypassword@localhost:5433/mydatabase') as db:
        db.insert(Transaction).values(*transactions).run()
