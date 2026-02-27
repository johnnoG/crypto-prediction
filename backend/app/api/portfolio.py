from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError

try:
    from auth import get_current_user
    from db import get_db
    from schemas import PortfolioHoldingCreate, PortfolioHoldingUpdate, PortfolioHoldingResponse, MessageResponse
    from models.user import User
    from models.portfolio import PortfolioHolding
except ImportError:
    from auth import get_current_user
    from db import get_db
    from schemas import PortfolioHoldingCreate, PortfolioHoldingUpdate, PortfolioHoldingResponse, MessageResponse
    from models.user import User
    from models.portfolio import PortfolioHolding


router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


@router.get("/holdings", response_model=List[PortfolioHoldingResponse])
async def get_holdings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[PortfolioHoldingResponse]:
    """Return all portfolio holdings for the current user."""
    holdings = (
        db.query(PortfolioHolding)
        .filter(PortfolioHolding.user_id == current_user.id)
        .order_by(PortfolioHolding.created_at.desc())
        .all()
    )
    return [PortfolioHoldingResponse.from_orm(h) for h in holdings]


@router.post("/holdings", response_model=PortfolioHoldingResponse, status_code=status.HTTP_201_CREATED)
async def add_holding(
    data: PortfolioHoldingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PortfolioHoldingResponse:
    """Add a new holding. If the coin already exists, raises 409."""
    existing = (
        db.query(PortfolioHolding)
        .filter(
            and_(
                PortfolioHolding.user_id == current_user.id,
                PortfolioHolding.crypto_id == data.crypto_id,
            )
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{data.crypto_symbol} is already in your portfolio. Use PUT to update it.",
        )

    try:
        holding = PortfolioHolding(
            user_id=current_user.id,
            crypto_id=data.crypto_id,
            crypto_symbol=data.crypto_symbol.upper(),
            crypto_name=data.crypto_name,
            amount=data.amount,
            avg_buy_price=data.avg_buy_price,
        )
        db.add(holding)
        db.commit()
        db.refresh(holding)
        return PortfolioHoldingResponse.from_orm(holding)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{data.crypto_symbol} is already in your portfolio.",
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.put("/holdings/{holding_id}", response_model=PortfolioHoldingResponse)
async def update_holding(
    holding_id: int,
    data: PortfolioHoldingUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PortfolioHoldingResponse:
    """Update amount and/or average buy price of a holding."""
    holding = (
        db.query(PortfolioHolding)
        .filter(
            and_(
                PortfolioHolding.id == holding_id,
                PortfolioHolding.user_id == current_user.id,
            )
        )
        .first()
    )
    if not holding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Holding not found")

    try:
        if data.amount is not None:
            holding.amount = data.amount
        if data.avg_buy_price is not None:
            holding.avg_buy_price = data.avg_buy_price
        db.commit()
        db.refresh(holding)
        return PortfolioHoldingResponse.from_orm(holding)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/holdings/{holding_id}", response_model=MessageResponse)
async def delete_holding(
    holding_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Remove a holding from the portfolio."""
    holding = (
        db.query(PortfolioHolding)
        .filter(
            and_(
                PortfolioHolding.id == holding_id,
                PortfolioHolding.user_id == current_user.id,
            )
        )
        .first()
    )
    if not holding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Holding not found")

    try:
        symbol = holding.crypto_symbol
        db.delete(holding)
        db.commit()
        return MessageResponse(
            message="Holding removed",
            detail=f"{symbol} has been removed from your portfolio",
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
