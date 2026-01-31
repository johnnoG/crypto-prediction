from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError

try:
    from auth import get_current_user
    from db import get_db
    from schemas import WatchlistCreate, WatchlistUpdate, WatchlistResponse, MessageResponse
    from models.user import User
    from models.watchlist import UserWatchlist
except ImportError:
    from auth import get_current_user
    from db import get_db
    from schemas import WatchlistCreate, WatchlistUpdate, WatchlistResponse, MessageResponse
    from models.user import User
    from models.watchlist import UserWatchlist


router = APIRouter(prefix="/watchlist", tags=["Watchlist"])


@router.post("/", response_model=WatchlistResponse, status_code=status.HTTP_201_CREATED)
async def add_to_watchlist(
    watchlist_data: WatchlistCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> WatchlistResponse:
    """Add a cryptocurrency to the user's watchlist."""
    print(f"[DEBUG] Adding to watchlist for user_id: {current_user.id}")
    print(f"[DEBUG] Watchlist data: {watchlist_data}")
    try:
        # Check if already in watchlist
        print(f"[DEBUG] Checking existing for user_id: {current_user.id}, crypto_symbol: {watchlist_data.crypto_symbol.upper()}")
        existing = db.query(UserWatchlist).filter(
            and_(
                UserWatchlist.user_id == current_user.id,
                UserWatchlist.crypto_symbol == watchlist_data.crypto_symbol.upper()
            )
        ).first()

        print(f"[DEBUG] Found existing item: {existing}")
        if existing:
            print(f"[DEBUG] Raising 409 conflict")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"{watchlist_data.crypto_symbol} is already in your watchlist"
            )

        # Create new watchlist entry
        new_watchlist_item = UserWatchlist(
            user_id=current_user.id,
            crypto_symbol=watchlist_data.crypto_symbol.upper(),
            crypto_name=watchlist_data.crypto_name,
            crypto_id=watchlist_data.crypto_id,
            notes=watchlist_data.notes,
            is_favorite=watchlist_data.is_favorite,
            notification_enabled=watchlist_data.notification_enabled
        )

        db.add(new_watchlist_item)
        db.commit()
        db.refresh(new_watchlist_item)

        return WatchlistResponse.from_orm(new_watchlist_item)

    except HTTPException:
        raise
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{watchlist_data.crypto_symbol} is already in your watchlist"
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add to watchlist: {str(e)}"
        )


@router.get("/", response_model=List[WatchlistResponse])
async def get_watchlist(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[WatchlistResponse]:
    """Get the user's watchlist."""
    print(f"[DEBUG] Getting watchlist for user_id: {current_user.id}")
    watchlist_items = db.query(UserWatchlist).filter(
        UserWatchlist.user_id == current_user.id
    ).order_by(UserWatchlist.created_at.desc()).all()
    print(f"[DEBUG] Found {len(watchlist_items)} watchlist items")

    # Also check all items in database for debugging
    all_items = db.query(UserWatchlist).all()
    print(f"[DEBUG] Total items in UserWatchlist table: {len(all_items)}")
    for item in all_items:
        print(f"[DEBUG] Item: user_id={item.user_id}, crypto_symbol={item.crypto_symbol}, crypto_id={item.crypto_id}")

    return [WatchlistResponse.from_orm(item) for item in watchlist_items]


@router.get("/{watchlist_id}", response_model=WatchlistResponse)
async def get_watchlist_item(
    watchlist_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> WatchlistResponse:
    """Get a specific watchlist item."""
    item = db.query(UserWatchlist).filter(
        and_(
            UserWatchlist.id == watchlist_id,
            UserWatchlist.user_id == current_user.id
        )
    ).first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist item not found"
        )

    return WatchlistResponse.from_orm(item)


@router.put("/{watchlist_id}", response_model=WatchlistResponse)
async def update_watchlist_item(
    watchlist_id: int,
    update_data: WatchlistUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> WatchlistResponse:
    """Update a watchlist item."""
    item = db.query(UserWatchlist).filter(
        and_(
            UserWatchlist.id == watchlist_id,
            UserWatchlist.user_id == current_user.id
        )
    ).first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist item not found"
        )

    try:
        # Update fields
        if update_data.notes is not None:
            item.notes = update_data.notes
        if update_data.is_favorite is not None:
            item.is_favorite = update_data.is_favorite
        if update_data.notification_enabled is not None:
            item.notification_enabled = update_data.notification_enabled

        db.commit()
        db.refresh(item)

        return WatchlistResponse.from_orm(item)

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update watchlist item: {str(e)}"
        )


@router.delete("/{watchlist_id}", response_model=MessageResponse)
async def remove_from_watchlist(
    watchlist_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> MessageResponse:
    """Remove an item from the watchlist."""
    item = db.query(UserWatchlist).filter(
        and_(
            UserWatchlist.id == watchlist_id,
            UserWatchlist.user_id == current_user.id
        )
    ).first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist item not found"
        )

    try:
        crypto_name = item.crypto_symbol
        db.delete(item)
        db.commit()

        return MessageResponse(
            message="Removed from watchlist",
            detail=f"{crypto_name} has been removed from your watchlist"
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove from watchlist: {str(e)}"
        )


@router.delete("/crypto/{crypto_symbol}", response_model=MessageResponse)
async def remove_crypto_from_watchlist(
    crypto_symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> MessageResponse:
    """Remove a cryptocurrency from watchlist by symbol."""
    item = db.query(UserWatchlist).filter(
        and_(
            UserWatchlist.user_id == current_user.id,
            UserWatchlist.crypto_symbol == crypto_symbol.upper()
        )
    ).first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{crypto_symbol} not found in your watchlist"
        )

    try:
        db.delete(item)
        db.commit()

        return MessageResponse(
            message="Removed from watchlist",
            detail=f"{crypto_symbol} has been removed from your watchlist"
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove from watchlist: {str(e)}"
        )