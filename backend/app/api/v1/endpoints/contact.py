# app/api/v1/endpoints/contact.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.database import get_db
from app.schemas import ContactMessage, ContactMessageCreate, ContactMessageUpdate
from app.services.contact_service import ContactService
from app.services.email_service import email_service
from app.auth import require_admin, get_current_active_user
from app.models import User as UserModel

router = APIRouter()

@router.post("/", response_model=ContactMessage)
async def create_contact_message(
    contact_data: ContactMessageCreate,
    db: Session = Depends(get_db)
):
    """Create a new contact message and send email notification"""
    try:
        # Create contact message in database
        contact_service = ContactService(db)
        contact_message = contact_service.create_contact_message(contact_data)
        
        # Send email notification to admin
        email_sent = await email_service.send_contact_form_email(
            name=contact_data.name,
            email=contact_data.email,
            subject=contact_data.subject,
            message=contact_data.message
        )
        
        if not email_sent:
            # Log the error but don't fail the request
            # The contact message is still saved in database
            pass
        
        return contact_message
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create contact message: {str(e)}"
        )

# Admin endpoints
@router.get("/admin/messages", response_model=List[ContactMessage])
async def get_contact_messages(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get all contact messages (admin only)"""
    contact_service = ContactService(db)
    return contact_service.get_contact_messages(skip=skip, limit=limit, status=status)

@router.get("/admin/messages/{message_id}", response_model=ContactMessage)
async def get_contact_message(
    message_id: int,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get a specific contact message (admin only)"""
    contact_service = ContactService(db)
    contact_message = contact_service.get_contact_message(message_id)
    
    if not contact_message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact message not found"
        )
    
    # Mark as read when admin views it
    if contact_message.status == "unread":
        contact_service.mark_as_read(message_id)
    
    return contact_message

@router.put("/admin/messages/{message_id}", response_model=ContactMessage)
async def update_contact_message(
    message_id: int,
    update_data: ContactMessageUpdate,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Update a contact message (admin only)"""
    contact_service = ContactService(db)
    contact_message = contact_service.update_contact_message(message_id, update_data)
    
    if not contact_message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact message not found"
        )
    
    return contact_message

@router.post("/admin/messages/{message_id}/mark-read")
async def mark_message_as_read(
    message_id: int,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Mark a contact message as read (admin only)"""
    contact_service = ContactService(db)
    contact_message = contact_service.mark_as_read(message_id)
    
    if not contact_message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact message not found"
        )
    
    return {"message": "Contact message marked as read"}

@router.post("/admin/messages/{message_id}/mark-replied")
async def mark_message_as_replied(
    message_id: int,
    admin_notes: str = "",
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Mark a contact message as replied (admin only)"""
    contact_service = ContactService(db)
    contact_message = contact_service.mark_as_replied(message_id, admin_notes)
    
    if not contact_message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact message not found"
        )
    
    return {"message": "Contact message marked as replied"}

@router.get("/admin/stats")
async def get_contact_stats(
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get contact message statistics (admin only)"""
    contact_service = ContactService(db)
    return contact_service.get_contact_stats()

@router.delete("/admin/messages/{message_id}")
async def delete_contact_message(
    message_id: int,
    admin_user: UserModel = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete a contact message (admin only)"""
    contact_service = ContactService(db)
    success = contact_service.delete_contact_message(message_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact message not found"
        )
    
    return {"message": "Contact message deleted successfully"}
