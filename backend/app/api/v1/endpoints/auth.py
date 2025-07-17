# app/api/v1/endpoints/auth.py
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.auth import (
    authenticate_user, create_access_token, create_refresh_token, 
    get_current_active_user, verify_refresh_token,
    ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS, require_admin
)
from app.schemas import UserLogin, Token, UserCreate, User, RefreshTokenRequest, AccessTokenResponse
from app.services.user_service import UserService
from app.models import User as UserModel

router = APIRouter(tags=["authentication"])

@router.post("/register", response_model=User)
async def register(user_create: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    try:
        user_service = UserService(db)
        user = user_service.create_user(user_create)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin, request: Request, db: Session = Depends(get_db)):
    """Login user and return access token."""
    print(f"Login attempt: username='{user_credentials.username}', password length={len(user_credentials.password)}")
    
    user = authenticate_user(db, user_credentials.username, user_credentials.password)
    if not user:
        print(f"Authentication failed for user: {user_credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    print(f"Authentication successful for user: {user.username}")
    
    # Update last login
    user_service = UserService(db)
    user_service.update_last_login(user.id)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(
        data={"sub": user.username}, expires_delta=refresh_token_expires
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": user
    }

@router.get("/me", response_model=User)
async def get_current_user_info(current_user: UserModel = Depends(get_current_active_user)):
    """Get current user information."""
    return current_user

@router.post("/logout")
async def logout(current_user: UserModel = Depends(get_current_active_user)):
    """Logout user (client should delete token)."""
    return {"message": "Successfully logged out"}

@router.post("/verify-token")
async def verify_token(current_user: UserModel = Depends(get_current_active_user)):
    """Verify if token is valid."""
    return {"valid": True, "user": current_user}

@router.post("/change-password")
async def change_password(
    old_password: str,
    new_password: str,
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Change user password."""
    # Verify old password
    from app.auth import verify_password
    if not verify_password(old_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Update password
    user_service = UserService(db)
    from app.schemas import UserUpdate
    user_update = UserUpdate(password=new_password)
    updated_user = user_service.update_user(current_user.id, user_update)
    
    return {"message": "Password updated successfully"}

@router.post("/refresh", response_model=AccessTokenResponse)
async def refresh_access_token(request: RefreshTokenRequest, db: Session = Depends(get_db)):
    """Refresh access token using refresh token."""
    try:
        # Verify refresh token and get username
        username = verify_refresh_token(request.refresh_token)
        
        # Get user from database
        user = db.query(UserModel).filter(UserModel.username == username).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
