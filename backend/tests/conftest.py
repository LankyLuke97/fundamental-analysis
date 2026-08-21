from fastapi.testclient import TestClient
from pytest import fixture
from sqlalchemy import event
from sqlmodel import create_engine, Session, SQLModel, StaticPool

# When creating the database tables with SQLModel.metadata.create_all,
# the models must first be imported to register them. This is not needed
# when using Alembic (or equivalent).
from app.db.database import get_session
from app.db.schema import all  # noqa: F401
from app.main import app


@fixture(name="clear_overrides", autouse=True)
def clear_overrides_fixture():
    yield
    app.dependency_overrides.clear()


@fixture(name="session")
def session_fixture():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@fixture(name="client")
def client_fixture(session: Session):
    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override

    client = TestClient(app)
    yield client
