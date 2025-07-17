"""Generate a sample CSV file for testing the CSV import functionality"""
import csv
from pathlib import Path

def create_sample_csv():
    """Create a sample CSV file with video metadata"""
    
    # Sample data
    sample_data = [
        {
            "videos": "001173649988146952-DEAF SCHOOL.mp4",
            "word": "DEAF SCHOOL",
            "title": "Deaf School Sign",
            "description": "ASL sign for deaf school",
            "Difficulty": "Beginner",
            "Category": "Education"
        },
        {
            "videos": "004622817625515863-INTERESTED.mp4", 
            "word": "INTERESTED",
            "title": "Interested Sign",
            "description": "ASL sign for interested",
            "Difficulty": "Beginner",
            "Category": "Emotions"
        },
        {
            "videos": "005944813363069956-EVERY NIGHT.mp4",
            "word": "EVERY NIGHT",
            "title": "Every Night Sign", 
            "description": "ASL sign for every night",
            "Difficulty": "Intermediate",
            "Category": "Time"
        },
        {
            "videos": "006111636181615365-GO THROUGH.mp4",
            "word": "GO THROUGH",
            "title": "Go Through Sign",
            "description": "ASL sign for go through",
            "Difficulty": "Intermediate", 
            "Category": "Actions"
        },
        {
            "videos": "008340922277732199-CAR BATTERY.mp4",
            "word": "CAR BATTERY",
            "title": "Car Battery Sign",
            "description": "ASL sign for car battery",
            "Difficulty": "Advanced",
            "Category": "Objects"
        }
    ]
    
    # Create CSV file
    csv_path = Path("sample_videos.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["videos", "word", "title", "description", "Difficulty", "Category"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in sample_data:
            writer.writerow(row)
    
    print(f"Sample CSV created: {csv_path.absolute()}")
    print(f"Contains {len(sample_data)} sample video entries")
    print("\nCSV Content:")
    print("-" * 50)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        print(f.read())

if __name__ == "__main__":
    create_sample_csv()
