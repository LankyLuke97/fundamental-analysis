from starlette.status import HTTP_204_NO_CONTENT
from typing import Sequence, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.db.schemas.cash_flow import (
    CashFlowCreate,
    CashFlowRead,
    CashFlowUpdate,
    CashFlowDelete,
)


router = APIRouter(prefix="cash_flows")


@router.get("/{cash_flow_id}", response_model=CashFlowRead)
async def get_cash_flow(cash_flow_id: int, repository):
    return await repository.get_cash_flow(cash_flow_id)


@router.get("", response_model=Sequence[CashFlowRead])
async def get_cash_flows(repository: Any):
    return await repository.list_cash_flows()


@router.post("", response_model=CashFlowRead)
async def create_cash_flow(cash_flow: CashFlowCreate, repository: Any):
    return await repository.add_cash_flow(cash_flow)


@router.patch("/{cash_flow_id}", response_model=CashFlowRead)
async def update_cash_flow(
    cash_flow_id: int, cash_flow: CashFlowUpdate, repository: Any
):
    return await repository.update_cash_flow(cash_flow_id, cash_flow)


@router.delete("/{cash_fllow_id}", status_code=HTTP_204_NO_CONTENT)
async def delete_cash_flow(cash_flow_id: int, repository: Any):
    return await repository.delete_cash_flow(cash_flow_id)
