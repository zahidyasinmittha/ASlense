# app/api/v1/endpoints/psl_alphabet.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Optional

from app.core.database import get_db
from app.auth import require_admin, get_current_user
from app.schemas import (
    PSLAlphabet, PSLAlphabetCreate, PSLAlphabetUpdate
)
from app.models import PSLAlphabet as PSLAlphabetModel, User

router = APIRouter()

# Public endpoints (for learning)
@router.get("/", response_model=List[PSLAlphabet])
async def get_psl_alphabet(
    skip: int = 0,
    limit: int = 50,
    difficulty: Optional[str] = Query(None, description="Filter by difficulty (easy, medium, hard)"),
    letter: Optional[str] = Query(None, description="Search by specific letter"),
    is_active: bool = Query(True, description="Filter by active status"),
    db: Session = Depends(get_db)
):
    """Get PSL alphabet data with optional filters."""
    query = db.query(PSLAlphabetModel).filter(PSLAlphabetModel.is_active == is_active)
    
    if difficulty:
        query = query.filter(PSLAlphabetModel.difficulty.ilike(f"%{difficulty}%"))
    
    if letter:
        query = query.filter(PSLAlphabetModel.letter.ilike(f"%{letter}%"))
    
    alphabet_data = query.order_by(PSLAlphabetModel.letter).offset(skip).limit(limit).all()
    return alphabet_data

@router.get("/count")
async def get_psl_alphabet_count(
    difficulty: Optional[str] = Query(None),
    is_active: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get total count of PSL alphabet entries."""
    query = db.query(PSLAlphabetModel).filter(PSLAlphabetModel.is_active == is_active)
    
    if difficulty:
        query = query.filter(PSLAlphabetModel.difficulty.ilike(f"%{difficulty}%"))
    
    total_count = query.count()
    return {"count": total_count}

@router.get("/letters")
async def get_available_letters(
    is_active: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get all available letters."""
    letters = db.query(PSLAlphabetModel.letter).filter(
        PSLAlphabetModel.is_active == is_active
    ).order_by(PSLAlphabetModel.letter).all()
    return [letter[0] for letter in letters]

@router.get("/difficulties")
async def get_available_difficulties(db: Session = Depends(get_db)):
    """Get all available difficulty levels."""
    difficulties = db.query(PSLAlphabetModel.difficulty).distinct().filter(
        PSLAlphabetModel.difficulty.isnot(None)
    ).all()
    return [diff[0] for diff in difficulties if diff[0]]

@router.get("/{letter_id}", response_model=PSLAlphabet)
async def get_psl_letter(
    letter_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific PSL alphabet entry by ID."""
    alphabet_entry = db.query(PSLAlphabetModel).filter(PSLAlphabetModel.id == letter_id).first()
    if not alphabet_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PSL alphabet entry not found"
        )
    return alphabet_entry

@router.get("/letter/{letter}", response_model=PSLAlphabet)
async def get_psl_letter_by_character(
    letter: str,
    db: Session = Depends(get_db)
):
    """Get PSL alphabet entry by letter character."""
    alphabet_entry = db.query(PSLAlphabetModel).filter(
        PSLAlphabetModel.letter.ilike(letter.upper()),
        PSLAlphabetModel.is_active == True
    ).first()
    if not alphabet_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PSL alphabet entry for letter '{letter}' not found"
        )
    return alphabet_entry

# Admin endpoints (CRUD operations)
@router.post("/", response_model=PSLAlphabet)
async def create_psl_alphabet_entry(
    psl_data: PSLAlphabetCreate,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Create a new PSL alphabet entry (Admin only)."""
    # Check if letter already exists
    existing_entry = db.query(PSLAlphabetModel).filter(
        PSLAlphabetModel.letter.ilike(psl_data.letter.upper())
    ).first()
    
    if existing_entry:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PSL alphabet entry for letter '{psl_data.letter}' already exists"
        )
    
    # Create new entry
    db_psl = PSLAlphabetModel(
        letter=psl_data.letter.upper(),
        file_path=psl_data.file_path,
        label=psl_data.label,
        difficulty=psl_data.difficulty.lower(),
        description=psl_data.description,
        is_active=psl_data.is_active
    )
    
    db.add(db_psl)
    db.commit()
    db.refresh(db_psl)
    
    return db_psl

@router.put("/{letter_id}", response_model=PSLAlphabet)
async def update_psl_alphabet_entry(
    letter_id: int,
    psl_data: PSLAlphabetUpdate,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Update a PSL alphabet entry (Admin only)."""
    alphabet_entry = db.query(PSLAlphabetModel).filter(PSLAlphabetModel.id == letter_id).first()
    
    if not alphabet_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PSL alphabet entry not found"
        )
    
    # Update fields if provided
    update_data = psl_data.model_dump(exclude_unset=True)
    
    if 'letter' in update_data:
        # Check if new letter conflicts with existing entries
        existing_entry = db.query(PSLAlphabetModel).filter(
            PSLAlphabetModel.letter.ilike(update_data['letter'].upper()),
            PSLAlphabetModel.id != letter_id
        ).first()
        
        if existing_entry:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"PSL alphabet entry for letter '{update_data['letter']}' already exists"
            )
        
        update_data['letter'] = update_data['letter'].upper()
    
    if 'difficulty' in update_data:
        update_data['difficulty'] = update_data['difficulty'].lower()
    
    for field, value in update_data.items():
        setattr(alphabet_entry, field, value)
    
    db.commit()
    db.refresh(alphabet_entry)
    
    return alphabet_entry

@router.delete("/{letter_id}")
async def delete_psl_alphabet_entry(
    letter_id: int,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete a PSL alphabet entry (Admin only)."""
    alphabet_entry = db.query(PSLAlphabetModel).filter(PSLAlphabetModel.id == letter_id).first()
    
    if not alphabet_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PSL alphabet entry not found"
        )
    
    db.delete(alphabet_entry)
    db.commit()
    
    return {"message": f"PSL alphabet entry for letter '{alphabet_entry.letter}' deleted successfully"}

@router.get("/admin/all", response_model=List[PSLAlphabet])
async def get_all_psl_alphabet_admin(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    difficulty: Optional[str] = None,
    is_active: Optional[bool] = None,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get all PSL alphabet entries including inactive ones with search and filters (Admin only)."""
    query = db.query(PSLAlphabetModel)
    
    # Apply search filter
    if search:
        query = query.filter(
            or_(
                PSLAlphabetModel.letter.ilike(f"%{search}%"),
                PSLAlphabetModel.label.ilike(f"%{search}%"),
                PSLAlphabetModel.file_path.ilike(f"%{search}%")
            )
        )
    
    # Apply difficulty filter
    if difficulty:
        query = query.filter(PSLAlphabetModel.difficulty == difficulty)
    
    # Apply status filter
    if is_active is not None:
        query = query.filter(PSLAlphabetModel.is_active == is_active)
    
    alphabet_data = query.order_by(PSLAlphabetModel.letter).offset(skip).limit(limit).all()
    return alphabet_data

@router.patch("/{letter_id}/toggle-status")
async def toggle_psl_alphabet_status(
    letter_id: int,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Toggle active status of a PSL alphabet entry (Admin only)."""
    alphabet_entry = db.query(PSLAlphabetModel).filter(PSLAlphabetModel.id == letter_id).first()
    
    if not alphabet_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PSL alphabet entry not found"
        )
    
    alphabet_entry.is_active = not alphabet_entry.is_active
    db.commit()
    db.refresh(alphabet_entry)
    
    status_text = "activated" if alphabet_entry.is_active else "deactivated"
    return {
        "message": f"PSL alphabet entry for letter '{alphabet_entry.letter}' {status_text}",
        "is_active": alphabet_entry.is_active
    }
