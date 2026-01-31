from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, validator


# User Schemas
class UserBase(BaseModel):
    """Base user schema with common fields."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)


class UserCreate(UserBase):
    """Schema for user creation."""
    password: str = Field(..., min_length=8, max_length=256)
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        return v


class UserUpdate(BaseModel):
    """Schema for user updates."""
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    preferences: Optional[str] = None


class PasswordChange(BaseModel):
    """Schema for password change."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=256)

    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserResponse(UserBase):
    """Schema for user response (public data)."""
    id: int
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    full_name: str
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


# Token Schemas
class Token(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenRefresh(BaseModel):
    """Schema for token refresh."""
    refresh_token: str


class TokenData(BaseModel):
    """Token payload data."""
    user_id: Optional[int] = None
    email: Optional[str] = None


# OAuth Schemas
class OAuthLoginRequest(BaseModel):
    """OAuth login request."""
    provider: str = Field(..., description="OAuth provider (google, github)")


class OAuthCallbackRequest(BaseModel):
    """OAuth callback request."""
    code: str = Field(..., description="Authorization code from OAuth provider")
    state: Optional[str] = Field(None, description="State parameter for CSRF protection")


class OAuthUrlResponse(BaseModel):
    """OAuth authorization URL response."""
    authorization_url: str
    state: str


# Authentication Response Schemas
class AuthResponse(BaseModel):
    """Complete authentication response."""
    user: UserResponse
    token: Token
    message: str = "Authentication successful"


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    detail: Optional[str] = None


# Alert Schemas
class AlertCreate(BaseModel):
    """Schema for creating a new alert."""
    crypto_symbol: str = Field(..., min_length=1, max_length=20)
    crypto_name: str = Field(..., min_length=1, max_length=100)
    alert_type: str = Field(..., description="Alert type: price_target, forecast_change, volatility")
    target_price: Optional[float] = Field(None, gt=0, description="Target price for price alerts")
    condition: Optional[str] = Field(None, description="Condition: above, below, reaches")
    message: Optional[str] = Field(None, max_length=500)
    expires_at: Optional[datetime] = None

    @validator('alert_type')
    def validate_alert_type(cls, v):
        """Validate alert type."""
        valid_types = ['price_target', 'forecast_change', 'volatility']
        if v not in valid_types:
            raise ValueError(f'Alert type must be one of: {", ".join(valid_types)}')
        return v

    @validator('condition')
    def validate_condition(cls, v):
        """Validate condition."""
        if v is not None:
            valid_conditions = ['above', 'below', 'reaches']
            if v not in valid_conditions:
                raise ValueError(f'Condition must be one of: {", ".join(valid_conditions)}')
        return v


class AlertResponse(BaseModel):
    """Schema for alert response."""
    id: int
    crypto_symbol: str
    crypto_name: str
    alert_type: str
    target_price: Optional[float]
    condition: Optional[str]
    status: str
    message: Optional[str]
    is_active: bool
    created_at: datetime
    triggered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Watchlist Schemas
class WatchlistCreate(BaseModel):
    """Schema for adding to watchlist."""
    crypto_symbol: str = Field(..., min_length=1, max_length=20)
    crypto_name: str = Field(..., min_length=1, max_length=100)
    crypto_id: str = Field(..., min_length=1, max_length=50, description="CoinGecko ID")
    notes: Optional[str] = Field(None, max_length=1000)
    is_favorite: bool = False
    notification_enabled: bool = True


class WatchlistUpdate(BaseModel):
    """Schema for updating watchlist item."""
    notes: Optional[str] = Field(None, max_length=1000)
    is_favorite: Optional[bool] = None
    notification_enabled: Optional[bool] = None


class WatchlistResponse(BaseModel):
    """Schema for watchlist response."""
    id: int
    crypto_symbol: str
    crypto_name: str
    crypto_id: str
    notes: Optional[str]
    is_favorite: bool
    notification_enabled: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
