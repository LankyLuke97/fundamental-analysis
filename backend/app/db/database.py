from sqlmodel import create_engine, Session, SQLModel

# When creating the database tables with SQLModel.metadata.create_all,
# the models must first be imported to register them. This is not needed
# when using Alembic (or equivalent).
from app.db.schemas import cash_flow, cash_flow_tag, cash_flow_category  # noqa : F401


# This is temporary 'up-and-running' code to test some of the models
sqlite_url = "sqlite://"
engine = create_engine(sqlite_url, echo=True)

SQLModel.metadata.create_all(engine)
