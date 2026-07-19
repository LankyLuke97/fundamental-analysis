from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import VARCHAR, TEXT
from sqlmodel import Column, Field, SQLModel, TIMESTAMP, text, Relationship

from app.db.schemas.cash_flow_category import CashFlowCategory
from app.db.schemas.cash_flow_tag import CashFlowTag, CashFlowTagLink


class CashFlow(SQLModel, table=True):
    __tablename__ = "cash_flows"

    id: Optional[int] = Field(default=None, primary_key=True)
    value: Decimal = Field(default=0, max_digits=19, decimal_places=2)
    category_id: int = Field(foreign_key="cash_flow_categories.id")
    currency: Optional[str] = Field(default="GBP", sa_type=VARCHAR(3))
    conversion_rate: Optional[Decimal] = Field(
        default=1.0, max_digits=19, decimal_places=2
    )
    expense: bool = True
    timestamp: Optional[datetime] = Field(
        default=None,
        sa_column=Column(
            TIMESTAMP(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
    )
    notes: Optional[str] = Field(default=None, sa_type=TEXT)

    category: CashFlowCategory = Relationship(back_populates="cash_flows")
    tags: list[CashFlowTag] = Relationship(
        back_populates="cash_flows", link_model=CashFlowTagLink
    )
