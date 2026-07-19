from sqlmodel import create_engine, Session, SQLModel

from app.db.schemas import cash_flow, cash_flow_tag, cash_flow_category


# This is temporary 'up-and-running' code to test some of the models
sqlite_url = "sqlite://"
engine = create_engine(sqlite_url, echo=True)

SQLModel.metadata.create_all(engine)
