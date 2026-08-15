from pytest import fixture

from app.main import app


@fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()
