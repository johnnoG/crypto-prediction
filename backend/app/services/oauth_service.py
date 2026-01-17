from __future__ import annotations

import secrets
from typing import Dict, Any, Optional
from urllib.parse import urlencode

import httpx
from authlib.integrations.base_client import OAuthError

try:
    from config import get_settings
    from models.user import User
    from auth import AuthService
    from db import get_db
except ImportError:
    from config import get_settings
    from models.user import User
    from auth import AuthService
    from db import get_db

from sqlalchemy.orm import Session


class GoogleOAuthService:
    """Google OAuth 2.0 service for social authentication."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client_id = self.settings.google_client_id
        self.client_secret = self.settings.google_client_secret
        self.redirect_uri = self.settings.google_redirect_uri
        
        # Google OAuth endpoints
        self.auth_url = "https://accounts.google.com/o/oauth2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"
        self.user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        
    def is_configured(self) -> bool:
        """Check if Google OAuth is properly configured."""
        return bool(self.client_id and self.client_secret)
    
    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """Generate Google OAuth authorization URL."""
        if not self.is_configured():
            raise ValueError("Google OAuth is not configured")
        
        if not state:
            state = secrets.token_urlsafe(32)
            
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "openid email profile",
            "state": state,
            "access_type": "offline",
            "prompt": "select_account"
        }
        
        return f"{self.auth_url}?{urlencode(params)}"
    
    async def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        if not self.is_configured():
            raise ValueError("Google OAuth is not configured")
            
        data = {
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(self.token_url, data=data)
            
            if response.status_code != 200:
                raise OAuthError(f"Token exchange failed: {response.text}")
                
            return response.json()
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from Google using access token."""
        headers = {"Authorization": f"Bearer {access_token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(self.user_info_url, headers=headers)
            
            if response.status_code != 200:
                raise OAuthError(f"Failed to get user info: {response.text}")
                
            return response.json()
    
    def find_or_create_user(self, db: Session, google_user_info: Dict[str, Any]) -> User:
        """Find existing user or create new user from Google profile."""
        google_id = str(google_user_info.get("id"))
        email = google_user_info.get("email")
        
        if not google_id or not email:
            raise ValueError("Invalid Google user info")
        
        # First, try to find user by Google ID
        user = db.query(User).filter(User.google_id == google_id).first()
        
        if user:
            # Update user info if needed
            if user.email != email:
                user.email = email
            if not user.first_name and google_user_info.get("given_name"):
                user.first_name = google_user_info.get("given_name")
            if not user.last_name and google_user_info.get("family_name"):
                user.last_name = google_user_info.get("family_name")
            db.commit()
            return user
        
        # Try to find user by email (existing user linking Google account)
        user = db.query(User).filter(User.email == email).first()
        
        if user:
            # Link Google account to existing user
            user.google_id = google_id
            db.commit()
            return user
        
        # Create new user
        username = self._generate_username_from_email(db, email)
        
        user = User(
            email=email,
            username=username,
            google_id=google_id,
            first_name=google_user_info.get("given_name"),
            last_name=google_user_info.get("family_name"),
            hashed_password="",  # No password for OAuth users
            is_verified=google_user_info.get("verified_email", False)
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return user
    
    def _generate_username_from_email(self, db: Session, email: str) -> str:
        """Generate unique username from email."""
        base_username = email.split("@")[0].lower()
        # Remove non-alphanumeric characters except underscores and hyphens
        base_username = "".join(c for c in base_username if c.isalnum() or c in "_-")
        
        username = base_username
        counter = 1
        
        # Ensure username is unique
        while db.query(User).filter(User.username == username).first():
            username = f"{base_username}{counter}"
            counter += 1
        
        return username


class GitHubOAuthService:
    """GitHub OAuth service for social authentication (future implementation)."""
    
    def __init__(self):
        self.settings = get_settings()
        # Add GitHub OAuth configuration here
        pass
    
    def is_configured(self) -> bool:
        """Check if GitHub OAuth is properly configured."""
        return False  # Not implemented yet


# Factory function to get OAuth service by provider
def get_oauth_service(provider: str):
    """Get OAuth service instance by provider name."""
    services = {
        "google": GoogleOAuthService,
        "github": GitHubOAuthService,
    }
    
    service_class = services.get(provider.lower())
    if not service_class:
        raise ValueError(f"Unsupported OAuth provider: {provider}")
    
    return service_class()
