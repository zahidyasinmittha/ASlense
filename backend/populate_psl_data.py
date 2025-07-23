#!/usr/bin/env python3
"""
Script to populate PSL alphabet data from CSV file
"""

import csv
import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
CSV_FILE = "sample_psl_data.csv"

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"  # You may need to update this

def get_admin_token():
    """Get admin authentication token"""
    login_data = {
        "username": ADMIN_USERNAME,
        "password": ADMIN_PASSWORD
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/auth/login", data=login_data)
    
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        print(f"Login failed: {response.text}")
        return None

def create_psl_entry(token, entry_data):
    """Create a PSL alphabet entry"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/psl-alphabet/",
        headers=headers,
        json=entry_data
    )
    
    return response

def main():
    print("Starting PSL alphabet data population...")
    
    # Get admin token
    token = get_admin_token()
    if not token:
        print("Failed to get admin token")
        return
    
    print("Got admin token successfully")
    
    # Read CSV file
    entries_created = 0
    errors = 0
    
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            for row in csv_reader:
                # Extract letter from label
                label = row['label'].strip()
                letter = None
                
                if '-' in label:
                    letter = label.split('-')[-1].strip()
                else:
                    for char in label:
                        if char.isalpha():
                            letter = char.upper()
                            break
                
                if not letter or len(letter) != 1:
                    print(f"Could not extract letter from label: {label}")
                    errors += 1
                    continue
                
                entry_data = {
                    "letter": letter.upper(),
                    "file_path": row['file_path'].strip(),
                    "label": label,
                    "difficulty": row['difficulty'].strip().lower(),
                    "description": row.get('description', '').strip() or None,
                    "is_active": True
                }
                
                response = create_psl_entry(token, entry_data)
                
                if response.status_code == 200:
                    print(f"Created PSL entry for letter: {letter}")
                    entries_created += 1
                else:
                    print(f"Failed to create entry for letter {letter}: {response.text}")
                    errors += 1
    
    except FileNotFoundError:
        print(f"CSV file not found: {CSV_FILE}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    print(f"\nCompleted PSL alphabet data population:")
    print(f"Entries created: {entries_created}")
    print(f"Errors: {errors}")
    
    # Get final stats
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/api/v1/psl-alphabet/count", headers=headers)
    if response.status_code == 200:
        total_count = response.json()["count"]
        print(f"Total PSL alphabet entries in database: {total_count}")

if __name__ == "__main__":
    main()
