from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import VARCHAR, TEXT
from sqlmodel import Column, Field, SQLModel, TIMESTAMP, text


class Transaction(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ticker: str = Field(sa_type=VARCHAR(6))
    price: Decimal = Field(default=0, max_digits=19, decimal_places=2)
    fees: Optional[Decimal] = Field(default=None, max_digits=19, decimal_places=2)
    currency: str = Field(default='GBP', sa_type=VARCHAR(3))
    quantity: float
    timestamp: Optional[datetime] = Field(default=None, sa_column=Column(TIMESTAMP(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")))
    notes: Optional[str] = Field(default=None, sa_type=TEXT)
