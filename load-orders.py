from sqlmodel import insert

from app.database_manager import connect
from app.schema import Transaction

from datetime import datetime
from dotenv import dotenv_values
from decimal import Decimal

env = dotenv_values('app/.env')

with open('data/orders.csv', mode='r', encoding='utf-8') as orders:
    order_list = orders.readlines()
    headers = order_list[0].strip().split(',')
    transactions = []
    for order in order_list[1:]:
        const = {k:v for k, v in zip(headers, order.strip().split(','))}
        const['price'] = Decimal(const['price'])
        const['quantity'] = float(const['quantity'])
        if const['type'] == 'buy': const['quantity'] *= -1
        del const['type']
        const['fees'] = Decimal(const['fees'])
        const['timestamp'] = datetime.strptime(const['datetime'], '%Y-%m-%d')
        del const['datetime']
        transactions.append(const)
    connection_string = f"{env['DB_PROTOCOL']}://{env['DB_USER']}:{env['DB_PASSWORD']}@{env['DB_HOST']}:{env['DB_PORT']}/{env['DB_NAME']}"
    with connect(connection_string=connection_string) as session:
        session.exec(insert(Transaction), params=transactions)
        session.commit()
