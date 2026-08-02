from sqlmodel import create_engine, Session, SQLModel

# When creating the database tables with SQLModel.metadata.create_all,
# the models must first be imported to register them. This is not needed
# when using Alembic (or equivalent).
from app.db.schema import all  # noqa: F401


# This is temporary 'up-and-running' code to test some of the models
sqlite_url = "sqlite://"
engine = create_engine(sqlite_url, echo=False)

SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        print("Getting real database session")
        yield session
