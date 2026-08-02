from typing import Optional, TYPE_CHECKING

from sqlmodel import Field, SQLModel, Relationship


if TYPE_CHECKING:
    from app.db.schemas.cash_flow import CashFlow


class CashFlowTagLink(SQLModel, table=True):
    cash_flow_id: int = Field(primary_key=True, foreign_key="cash_flows.id")
    cash_flow_tag_id: int = Field(primary_key=True, foreign_key="cash_flow_tags.id")


class CashFlowTagBase(SQLModel):
    name: str = Field(max_length=128)


class CashFlowTag(CashFlowTagBase, table=True):
    __tablename__ = "cash_flow_tags"

    id: Optional[int] = Field(default=None, primary_key=True)

    cash_flows: list["CashFlow"] = Relationship(
        back_populates="tags", link_model=CashFlowTagLink
    )


class CashFlowTagCreate(CashFlowTagBase):
    pass


class CashFlowTagRead(CashFlowTagBase):
    id: int


class CashFlowTagUpdate(CashFlowTagBase):
    id: int
    name: Optional[str]


class CashFlowTagDelete(CashFlowTagBase):
    id: int
