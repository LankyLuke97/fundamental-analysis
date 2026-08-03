from sqlmodel import Session, select

from app.db.schema.cash_flow import (
    CashFlow,
)


class CashFlowNotFound(Exception):
    pass


class CashFlowService:
    def __init__(self, session: Session):
        self._db = session

    def add_cash_flow(self, cash_flow: CashFlow) -> CashFlow:
        self._db.add(cash_flow)
        self._db.commit()
        self._db.refresh(cash_flow)
        if not cash_flow or not cash_flow.id:
            raise Exception  # to-do
        return cash_flow

    def get_cash_flow(self, cash_flow_id: int) -> CashFlow:
        query = select(CashFlow).where(CashFlow.id == cash_flow_id)
        cash_flow = self._db.exec(query).first()
        if not cash_flow:
            raise CashFlowNotFound
        return cash_flow
