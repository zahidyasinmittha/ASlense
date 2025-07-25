# app/services/contact_service.py
from sqlalchemy.orm import Session
from typing import List, Optional
from app.models import ContactMessage
from app.schemas import ContactMessageCreate, ContactMessageUpdate

class ContactService:
    """Service for handling contact message operations"""
    
    def __init__(self, db: Session):
        self.db = db

    def create_contact_message(self, contact_data: ContactMessageCreate) -> ContactMessage:
        """Create a new contact message"""
        db_contact = ContactMessage(
            name=contact_data.name,
            email=contact_data.email,
            subject=contact_data.subject,
            message=contact_data.message,
            status="unread"
        )
        self.db.add(db_contact)
        self.db.commit()
        self.db.refresh(db_contact)
        return db_contact

    def get_contact_messages(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[ContactMessage]:
        """Get contact messages with optional filtering"""
        query = self.db.query(ContactMessage)
        
        if status:
            query = query.filter(ContactMessage.status == status)
            
        return query.order_by(ContactMessage.created_at.desc()).offset(skip).limit(limit).all()

    def get_contact_message(self, message_id: int) -> Optional[ContactMessage]:
        """Get a specific contact message by ID"""
        return self.db.query(ContactMessage).filter(ContactMessage.id == message_id).first()

    def update_contact_message(
        self, 
        message_id: int, 
        update_data: ContactMessageUpdate
    ) -> Optional[ContactMessage]:
        """Update a contact message"""
        db_contact = self.get_contact_message(message_id)
        if not db_contact:
            return None

        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(db_contact, field, value)

        self.db.commit()
        self.db.refresh(db_contact)
        return db_contact

    def mark_as_read(self, message_id: int) -> Optional[ContactMessage]:
        """Mark a contact message as read"""
        return self.update_contact_message(
            message_id, 
            ContactMessageUpdate(status="read")
        )

    def mark_as_replied(self, message_id: int, admin_notes: str = "") -> Optional[ContactMessage]:
        """Mark a contact message as replied"""
        return self.update_contact_message(
            message_id, 
            ContactMessageUpdate(status="replied", admin_notes=admin_notes)
        )

    def get_contact_stats(self) -> dict:
        """Get contact message statistics"""
        total = self.db.query(ContactMessage).count()
        unread = self.db.query(ContactMessage).filter(ContactMessage.status == "unread").count()
        read = self.db.query(ContactMessage).filter(ContactMessage.status == "read").count()
        replied = self.db.query(ContactMessage).filter(ContactMessage.status == "replied").count()
        
        return {
            "total": total,
            "unread": unread,
            "read": read,
            "replied": replied
        }

    def delete_contact_message(self, message_id: int) -> bool:
        """Delete a contact message"""
        db_contact = self.get_contact_message(message_id)
        if not db_contact:
            return False
            
        self.db.delete(db_contact)
        self.db.commit()
        return True
