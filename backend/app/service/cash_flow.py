from sqlmodel import Session, select

from app.db.schema.cash_flow import CashFlow, CashFlowUpdate


class CashFlowNotFound(Exception):
    pass


class CashFlowService:
    def __init__(self, session: Session):
        self._db = session

    def _fetch_cash_flow_from_db(self, cash_flow_id: int) -> CashFlow:
        query = select(CashFlow).where(CashFlow.id == cash_flow_id)
        stored_cash_flow = self._db.exec(query).first()
        if not stored_cash_flow:
            raise CashFlowNotFound
        return stored_cash_flow

    def add_cash_flow(self, cash_flow: CashFlow) -> CashFlow:
        self._db.add(cash_flow)
        self._db.commit()
        self._db.refresh(cash_flow)
        return cash_flow

    def get_cash_flow(self, cash_flow_id: int) -> CashFlow:
        return self._fetch_cash_flow_from_db(cash_flow_id=cash_flow_id)

    def list_cash_flows(self) -> list[CashFlow]:
        query = select(CashFlow)
        cash_flows = list(self._db.exec(query).all())
        return cash_flows

    def update_cash_flow(
        self, cash_flow_id: int, cash_flow: CashFlowUpdate
    ) -> CashFlow:
        stored_cash_flow = self._fetch_cash_flow_from_db(cash_flow_id=cash_flow_id)
        stored_cash_flow.sqlmodel_update(cash_flow.model_dump(exclude_unset=True))
        self._db.add(stored_cash_flow)
        self._db.commit()
        self._db.refresh(stored_cash_flow)
        return stored_cash_flow

    def delete_cash_flow(self, cash_flow_id: int) -> None:
        stored_cash_flow = self._fetch_cash_flow_from_db(cash_flow_id=cash_flow_id)
        self._db.delete(stored_cash_flow)
        self._db.commit()
