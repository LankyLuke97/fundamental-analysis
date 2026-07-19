from typing import Optional, TYPE_CHECKING
from sqlmodel import Field, SQLModel, Relationship


if TYPE_CHECKING:
    from app.db.schemas.cash_flow import CashFlow


class CashFlowCategory(SQLModel, table=True):
    __tablename__ = "cash_flow_categories"

    id: int = Field(default=None, primary_key=True)
    name: str
    description: Optional[str] = Field(default=None)

    cash_flows: list["CashFlow"] = Relationship(back_populates="category")
