from typing import Sequence
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.db.schemas.cash_flow import (
    CashFlowCreate,
    CashFlowRead,
    CashFlowUpdate,
    CashFlowDelete,
)


router = APIRouter()
