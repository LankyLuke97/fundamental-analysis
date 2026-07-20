from typing import Optional, TYPE_CHECKING
from sqlmodel import Field, SQLModel, Relationship


if TYPE_CHECKING:
    from app.db.schemas.cash_flow import CashFlow


class CashFlowCategoryBase(SQLModel):
    name: str = Field(max_length=128)
    description: Optional[str] = Field(max_length=1024)


class CashFlowCategory(CashFlowCategoryBase, table=True):
    __tablename__ = "cash_flow_categories"

    id: Optional[int] = Field(default=None, primary_key=True)

    cash_flows: list["CashFlow"] = Relationship(back_populates="category")


class CashFlowCategoryCreate(CashFlowCategoryBase):
    pass


class CashFlowCategoryRead(CashFlowCategoryBase):
    id: int


class CashFlowCategoryUpdate(CashFlowCategoryBase):
    id: int
    name: Optional[str]


class CashFlowCategoryDelete(CashFlowCategoryBase):
    id: int
