from contextlib import asynccontextmanager, contextmanager
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from typing import Annotated

from embar.db.pg import AsyncPgDb, PgDb

@contextmanager
def open_connection_pool(protocol='postgres', username=None, password=None, hostname='localhost', port=5432, database='postgres', open=True, **kwargs):
    if not username: raise ValueError("Username must be provided for connection string")
    if not password: raise ValueError("Password must be provided for connection string")

    connection_string = f"{protocol}://{username}:{password}@{hostname}:{port}/{database}"
    try:
        pool = ConnectionPool(conninfo=connection_string, open=open, **kwargs)
        yield PgDb(pool)
    finally:
        pool.close()

@asynccontextmanager
async def open_async_connection_pool(protocol='postgres', username=None, password=None, hostname='localhost', port=5432, database='postgres', open=True, **kwargs):
    if not username: raise ValueError("Username must be provided for connection string")
    if not password: raise ValueError("Password must be provided for connection string")

    connection_string = f"{protocol}://{username}:{password}@{hostname}:{port}/{database}"
    try:
        pool = await ConnectionPool(conninfo=connection_string, open=open, **kwargs)
        yield AsyncPgDb(pool)
    finally:
        pool.close()

