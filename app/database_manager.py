from contextlib import contextmanager

from sqlmodel import create_engine, Session

engine = None

@contextmanager
def connect(protocol='postgres', username=None, password=None, hostname='localhost', port=5432, database='postgres', echo=False, **kwargs):
    if 'connection_string' in kwargs: 
        connection_string = kwargs['connection_string']
        del kwargs['connection_string']
    else: 
        if not username: raise ValueError("Username must be provided for connection string")
        if not password: raise ValueError("Password must be provided for connection string")
        connection_string = f"{protocol}://{username}:{password}@{hostname}:{port}/{database}"
    try:
        engine = engine or create_engine(connection_string, echo=echo)
        pool = ConnectionPool(conninfo=connection_string, open=open, **kwargs)
        yield PgDb(pool)
    finally:
        if pool: pool.close()

