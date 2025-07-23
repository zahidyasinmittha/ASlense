#!/usr/bin/env python3
"""
Simple script to populate PSL alphabet data directly into the database
"""

import csv
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import SessionLocal
from app.models import PSLAlphabet

def populate_psl_data():
    """Populate PSL alphabet data from CSV file"""
    db = SessionLocal()
    
    try:
        # Read CSV file
        csv_file = "sample_psl_data.csv"
        entries_created = 0
        errors = []
        
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            for row_num, row in enumerate(csv_reader, start=2):
                try:
                    # Use the full label + file_path as unique identifier for PSL images
                    label = row['label'].strip()
                    file_path = row['file_path'].strip()
                    unique_id = f"{label}_{file_path}"  # Create unique identifier
                    
                    # Check if entry already exists
                    existing_entry = db.query(PSLAlphabet).filter(
                        PSLAlphabet.letter == unique_id
                    ).first()
                    
                    if existing_entry:
                        print(f"Entry '{unique_id}' already exists, skipping...")
                        continue
                    
                    # Create new entry
                    psl_entry = PSLAlphabet(
                        letter=unique_id,  # Use unique identifier
                        file_path=file_path,
                        label=label,
                        difficulty=row['difficulty'].strip().lower(),
                        description=None,  # No description in CSV
                        is_active=True
                    )
                    
                    db.add(psl_entry)
                    entries_created += 1
                    print(f"Created PSL entry: {label} - {file_path}")
                    
                except Exception as e:
                    errors.append(f"Row {row_num}: Error processing row - {str(e)}")
        
        # Commit all changes
        db.commit()
        
        print(f"\nCompleted PSL alphabet data population:")
        print(f"Entries created: {entries_created}")
        print(f"Errors: {len(errors)}")
        
        if errors:
            print("\nErrors encountered:")
            for error in errors:
                print(f"  - {error}")
        
        # Get final count
        total_count = db.query(PSLAlphabet).count()
        print(f"Total PSL alphabet entries in database: {total_count}")
        
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    populate_psl_data()
