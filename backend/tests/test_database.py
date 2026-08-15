from sqlalchemy import event
from sqlmodel import create_engine, Session, SQLModel

# When creating the database tables with SQLModel.metadata.create_all,
# the models must first be imported to register them. This is not needed
# when using Alembic (or equivalent).
from app.db.schema import all  # noqa: F401


# This is temporary 'up-and-running' code to test some of the models


def test_get_session():
    sqlite_url = "sqlite://"
    engine = create_engine(sqlite_url, echo=False)

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
