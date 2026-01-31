from __future__ import annotations

from typing import List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import and_

try:
    from auth import get_current_user
    from db import get_db
    from schemas import AlertCreate, AlertResponse, MessageResponse
    from models.user import User
    from models.alert import UserAlert, AlertType, AlertStatus
except ImportError:
    from auth import get_current_user
    from db import get_db
    from schemas import AlertCreate, AlertResponse, MessageResponse
    from models.user import User
    from models.alert import UserAlert, AlertType, AlertStatus


router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.post("/", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    alert_data: AlertCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> AlertResponse:
    """Create a new price alert for the current user."""
    try:
        # Convert string alert_type to enum
        try:
            alert_type_enum = AlertType(alert_data.alert_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid alert type: {alert_data.alert_type}"
            )

        # Create new alert
        new_alert = UserAlert(
            user_id=current_user.id,
            crypto_symbol=alert_data.crypto_symbol.upper(),
            crypto_name=alert_data.crypto_name,
            alert_type=alert_type_enum,
            target_price=alert_data.target_price,
            condition=alert_data.condition,
            message=alert_data.message,
            expires_at=alert_data.expires_at,
            status=AlertStatus.ACTIVE
        )

        db.add(new_alert)
        db.commit()
        db.refresh(new_alert)

        return AlertResponse.from_orm(new_alert)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create alert: {str(e)}"
        )


@router.get("/", response_model=List[AlertResponse])
async def get_user_alerts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[AlertResponse]:
    """Get all alerts for the current user."""
    alerts = db.query(UserAlert).filter(
        and_(
            UserAlert.user_id == current_user.id,
            UserAlert.is_active == True
        )
    ).order_by(UserAlert.created_at.desc()).all()

    return [AlertResponse.from_orm(alert) for alert in alerts]


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> AlertResponse:
    """Get a specific alert by ID."""
    alert = db.query(UserAlert).filter(
        and_(
            UserAlert.id == alert_id,
            UserAlert.user_id == current_user.id
        )
    ).first()

    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )

    return AlertResponse.from_orm(alert)


@router.delete("/{alert_id}", response_model=MessageResponse)
async def delete_alert(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> MessageResponse:
    """Delete an alert."""
    alert = db.query(UserAlert).filter(
        and_(
            UserAlert.id == alert_id,
            UserAlert.user_id == current_user.id
        )
    ).first()

    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )

    try:
        # Soft delete by setting is_active to False
        alert.is_active = False
        alert.status = AlertStatus.CANCELLED
        alert.updated_at = datetime.utcnow()
        db.commit()

        return MessageResponse(
            message="Alert deleted successfully",
            detail=f"Alert for {alert.crypto_symbol} has been removed"
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete alert: {str(e)}"
        )


@router.get("/crypto/{crypto_symbol}", response_model=List[AlertResponse])
async def get_alerts_for_crypto(
    crypto_symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[AlertResponse]:
    """Get all alerts for a specific cryptocurrency."""
    alerts = db.query(UserAlert).filter(
        and_(
            UserAlert.user_id == current_user.id,
            UserAlert.crypto_symbol == crypto_symbol.upper(),
            UserAlert.is_active == True
        )
    ).order_by(UserAlert.created_at.desc()).all()

    return [AlertResponse.from_orm(alert) for alert in alerts]