from starlette.status import HTTP_204_NO_CONTENT
from typing import Sequence, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.db.schema.cash_flow import (
    CashFlowCreate,
    CashFlowRead,
    CashFlowUpdate,
)
from app.db.database import get_session
from app.services.cash_flow_service import CashFlowService


router = APIRouter(prefix="cash_flows")


def get_cash_flow_service() -> CashFlowService:
    return CashFlowService(session=Depends(get_session))


@router.get("/{cash_flow_id}", response_model=CashFlowRead)
async def get_cash_flow(
    cash_flow_id: int, service: CashFlowService = Depends(get_cash_flow_service)
):
    return await service.get_cash_flow(cash_flow_id)


@router.get("", response_model=Sequence[CashFlowRead])
async def get_cash_flows(service: CashFlowService = Depends(get_cash_flow_service)):
    return await service.list_cash_flows()


@router.post("", response_model=CashFlowRead)
async def create_cash_flow(
    cash_flow: CashFlowCreate, service: CashFlowService = Depends(get_cash_flow_service)
):
    return await service.add_cash_flow(cash_flow)


@router.patch("/{cash_flow_id}", response_model=CashFlowRead)
async def update_cash_flow(
    cash_flow_id: int,
    cash_flow: CashFlowUpdate,
    service: CashFlowService = Depends(get_cash_flow_service),
):
    return await service.update_cash_flow(cash_flow_id, cash_flow)


@router.delete("/{cash_fllow_id}", status_code=HTTP_204_NO_CONTENT)
async def delete_cash_flow(
    cash_flow_id: int, service: CashFlowService = Depends(get_cash_flow_service)
):
    return await service.delete_cash_flow(cash_flow_id)
    l
