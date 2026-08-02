from sqlmodel import Session, select

from app.db.schema.cash_flow import (
    CashFlow,
    CashFlowCreate,
    CashFlowRead,
    CashFlowUpdate,
    CashFlowDelete,
)


class CashFlowNotFound(Exception):
    pass


class CashFlowService:
    def __init__(self, session: Session):
        self._db = session

    def get_cash_flow(self, cash_flow_id: int) -> CashFlow:
        query = select(CashFlow).where(CashFlow.id == cash_flow_id)
        cash_flow = self._db.exec(query).first()
        if not cash_flow:
            raise CashFlowNotFound
        return cash_flow
