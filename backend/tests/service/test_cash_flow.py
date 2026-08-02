from decimal import Decimal

from fastapi import Depends
from pytest import fixture

from app.db.database import get_session
from app.db.schema.cash_flow import CashFlow
from app.db.schema.cash_flow_category import CashFlowCategory
from app.main import app
from app.service.cash_flow import CashFlowService

from tests.test_database import get_test_session


app.dependency_overrides[get_session] = get_test_session


@fixture
def add_cash_flow():
    pass


def test_get_cash_flow():
    session = get_test_session()
    category = CashFlowCategory(name="Test Category")
    session.add(category)
    session.commit()
    session.refresh(category)
    cash_flow = CashFlow(value=Decimal("100.00"), category_id=(category.id))
    session.add(cash_flow)
    session.commit()
    session.refresh(cash_flow)
    service = CashFlowService(session=session)
    returned_cash_flow = service.get_cash_flow(cash_flow.id)
    assert cash_flow == returned_cash_flow
