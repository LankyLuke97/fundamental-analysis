from contextlib import contextmanager
from typing import Iterable
from decimal import Decimal

from pytest import fixture, raises
from sqlmodel import SQLModel, Session

from app.db.schema.cash_flow import CashFlow
from app.db.schema.cash_flow_category import CashFlowCategory
from app.service.cash_flow import CashFlowNotFound, CashFlowService


@contextmanager
def temp_database_data(data: Iterable[SQLModel], session: Session):
    session.add_all(data)
    session.commit()
    for datum in data:
        session.refresh(datum)
    yield data


@fixture
def service(test_session):
    yield CashFlowService(session=test_session)


@fixture
def add_categories(test_session):
    categories = [
        CashFlowCategory(
            name="Test category 1",
        ),
        CashFlowCategory(
            name="Test category 2",
            description="Optional category description",
        ),
        CashFlowCategory(
            name="Test category 3",
        ),
    ]

    with temp_database_data(categories, test_session) as data:
        yield data


@fixture
def add_cash_flows(test_session, add_categories):
    categories = add_categories
    cash_flows = [
        CashFlow(
            value=Decimal("100.00"),
            category_id=categories[0].id,
        ),
        CashFlow(
            value=Decimal("75.00"),
            category_id=categories[0].id,
        ),
        CashFlow(
            value=Decimal("10.00"),
            category_id=categories[1].id,
        ),
    ]
    with temp_database_data(cash_flows, test_session) as data:
        yield data


def test_get_cash_flow(service, add_cash_flows):
    expected = add_cash_flows[0]
    retrieved = service.get_cash_flow(expected.id)
    assert retrieved == expected


def test_get_missing_cash_flow(service):
    with raises(CashFlowNotFound):
        service.get_cash_flow(-1)


def test_add_cash_flow(service, add_categories):
    category = add_categories[0]

    cash_flow = service.add_cash_flow(
        CashFlow(
            value=Decimal("200.00"),
            category_id=category.id,
        )
    )

    assert cash_flow.id is not None
    assert Decimal("200.00") == cash_flow.value
    assert category.id == cash_flow.category_id
