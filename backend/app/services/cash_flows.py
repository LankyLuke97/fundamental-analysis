from sqlmodel import Session, select

from app.db.schemas.cash_flow import (
    CashFlowCreate,
    CashFlowRead,
    CashFlowUpdate,
    CashFlowDelete,
)


class CashFlowService:
    def __init__(self, session: Session):
        self._db = session
