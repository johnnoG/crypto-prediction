from __future__ import annotations

from datetime import timedelta
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

try:
    from auth import AuthService, get_current_user, security
    from db import get_db
    from schemas import (
        UserCreate, UserLogin, UserResponse, UserUpdate,
        AuthResponse, Token, TokenRefresh, MessageResponse,
        OAuthLoginRequest, OAuthCallbackRequest, OAuthUrlResponse
    )
    from models.user import User
    from services.oauth_service import get_oauth_service
except ImportError:
    from auth import AuthService, get_current_user, security
    from db import get_db
    from schemas import (
        UserCreate, UserLogin, UserResponse, UserUpdate,
        AuthResponse, Token, TokenRefresh, MessageResponse,
        OAuthLoginRequest, OAuthCallbackRequest, OAuthUrlResponse
    )
    from models.user import User
    from services.oauth_service import get_oauth_service


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    user_data: UserCreate,
    db: Session = Depends(get_db)
) -> AuthResponse:
    """Register a new user account.
    
    Creates a new user with secure password hashing and returns
    authentication tokens for immediate login.
    """
    try:
        # Create the user
        user = AuthService.create_user(
            db=db,
            email=user_data.email,
            username=user_data.username,
            password=user_data.password,
            first_name=user_data.first_name,
            last_name=user_data.last_name
        )
        
        # Generate tokens
        access_token = AuthService.create_access_token(
            data={"sub": str(user.id), "email": user.email}
        )
        refresh_token = AuthService.create_refresh_token(
            data={"sub": str(user.id), "email": user.email}
        )
        
        # Create response
        token = Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=30 * 60  # 30 minutes in seconds
        )
        
        return AuthResponse(
            user=UserResponse.from_orm(user),
            token=token,
            message="Account created successfully! Welcome to CryptoForecast."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create account: {str(e)}"
        )


@router.post("/signin", response_model=AuthResponse)
async def signin(
    credentials: UserLogin,
    db: Session = Depends(get_db)
) -> AuthResponse:
    """Authenticate user and return access tokens.
    
    Validates user credentials and returns JWT tokens for API access.
    """
    user = AuthService.authenticate_user(
        db=db,
        email=credentials.email,
        password=credentials.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if user.is_locked:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Account is temporarily locked due to failed login attempts. Please try again later."
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account is deactivated. Please contact support."
        )
    
    # Generate tokens
    access_token = AuthService.create_access_token(
        data={"sub": str(user.id), "email": user.email}
    )
    refresh_token = AuthService.create_refresh_token(
        data={"sub": str(user.id), "email": user.email}
    )
    
    token = Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=30 * 60  # 30 minutes in seconds
    )
    
    return AuthResponse(
        user=UserResponse.from_orm(user),
        token=token,
        message="Sign in successful! Welcome back."
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    token_data: TokenRefresh,
    db: Session = Depends(get_db)
) -> Token:
    """Refresh access token using refresh token.
    
    Validates the refresh token and issues a new access token.
    """
    # Verify refresh token
    payload = AuthService.verify_token(token_data.refresh_token, "refresh")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user and validate
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user or not user.is_active or user.is_locked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate new tokens
    access_token = AuthService.create_access_token(
        data={"sub": str(user.id), "email": user.email}
    )
    new_refresh_token = AuthService.create_refresh_token(
        data={"sub": str(user.id), "email": user.email}
    )
    
    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=30 * 60
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(
    current_user: User = Depends(get_current_user)
) -> MessageResponse:
    """Logout user (client should discard tokens).
    
    Note: In a stateless JWT system, logout is handled client-side
    by discarding tokens. Server-side blacklisting could be added later.
    """
    return MessageResponse(
        message="Logged out successfully",
        detail="Please discard your authentication tokens"
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
) -> UserResponse:
    """Get current user profile information."""
    return UserResponse.from_orm(current_user)


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> UserResponse:
    """Update current user profile information."""
    # Update user fields
    if user_update.first_name is not None:
        current_user.first_name = user_update.first_name
    if user_update.last_name is not None:
        current_user.last_name = user_update.last_name
    if user_update.preferences is not None:
        current_user.preferences = user_update.preferences
    
    db.commit()
    db.refresh(current_user)
    
    return UserResponse.from_orm(current_user)


@router.get("/verify-token")
async def verify_token_endpoint(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Verify if the current token is valid.
    
    Useful for frontend to check token validity without full user data.
    """
    return {
        "valid": True,
        "user_id": current_user.id,
        "email": current_user.email,
        "username": current_user.username
    }


# OAuth Routes
@router.get("/oauth/{provider}", response_model=OAuthUrlResponse)
async def oauth_login(provider: str) -> OAuthUrlResponse:
    """Initiate OAuth login flow.
    
    Returns the authorization URL for the specified OAuth provider.
    """
    try:
        oauth_service = get_oauth_service(provider)
        
        if not oauth_service.is_configured():
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=(
                    f"{provider.title()} OAuth is not configured. "
                    "Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI."
                ),
            )
        
        # Generate state for CSRF protection
        import secrets
        state = secrets.token_urlsafe(32)
        
        authorization_url = oauth_service.get_authorization_url(state)
        
        return OAuthUrlResponse(
            authorization_url=authorization_url,
            state=state
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth initialization failed: {str(e)}"
        )


@router.get("/oauth/{provider}/callback", response_model=AuthResponse)
async def oauth_callback(
    provider: str,
    code: str,
    state: Optional[str] = None,
    db: Session = Depends(get_db)
) -> AuthResponse:
    """Handle OAuth callback and authenticate user.
    
    Exchanges the authorization code for user information and creates/updates
    the user account, then returns authentication tokens.
    """
    try:
        oauth_service = get_oauth_service(provider)
        
        if not oauth_service.is_configured():
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=(
                    f"{provider.title()} OAuth is not configured. "
                    "Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI."
                ),
            )
        
        # Exchange code for access token
        token_data = await oauth_service.exchange_code_for_token(code)
        access_token = token_data.get("access_token")
        
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to obtain access token"
            )
        
        # Get user information from OAuth provider
        user_info = await oauth_service.get_user_info(access_token)
        
        # Find or create user in our database
        user = oauth_service.find_or_create_user(db, user_info)
        
        # Update last login
        from datetime import datetime
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Generate our own JWT tokens
        jwt_access_token = AuthService.create_access_token(
            data={"sub": str(user.id), "email": user.email}
        )
        jwt_refresh_token = AuthService.create_refresh_token(
            data={"sub": str(user.id), "email": user.email}
        )
        
        token = Token(
            access_token=jwt_access_token,
            refresh_token=jwt_refresh_token,
            expires_in=30 * 60  # 30 minutes
        )
        
        return AuthResponse(
            user=UserResponse.from_orm(user),
            token=token,
            message=f"Successfully signed in with {provider.title()}!"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth authentication failed: {str(e)}"
        )


@router.post("/oauth/{provider}/mobile", response_model=AuthResponse)
async def oauth_mobile_callback(
    provider: str,
    callback_data: OAuthCallbackRequest,
    db: Session = Depends(get_db)
) -> AuthResponse:
    """Handle OAuth callback for mobile applications.
    
    Mobile apps can use this endpoint to directly send the authorization code
    instead of using the web callback flow.
    """
    return await oauth_callback(
        provider=provider,
        code=callback_data.code,
        state=callback_data.state,
        db=db
    )
