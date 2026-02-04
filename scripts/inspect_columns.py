
import pandas as pd
from pathlib import Path

# Define paths
PROJECT_ROOT = Path(r"c:\Users\gulma\Desktop\Timetabling-at-the-University-of-Edinburgh\Timetabling-at-the-University-of-Edinburgh")
DATA_RAW = PROJECT_ROOT / "data" / "raw"

def list_data_columns():
    files = {
        "Events": "2024-5_Event_Module_Room.xlsx",
        "Rooms": "Rooms_and_Room_Types.xlsx"
    }
    
    for name, filename in files.items():
        print(f"\n--- {name} ({filename}) ---")
        try:
            df = pd.read_excel(DATA_RAW / filename, nrows=5) # Read brief header
            print(f"Columns found ({len(df.columns)}):")
            for col in df.columns:
                print(f"  - {col}")
                
            # Check for potentially missed constraints
            if name == "Rooms":
                if 'Zone' in df.columns or 'Campus' in df.columns:
                    print(f"  [NOTE] Location columns found in Rooms.")
            
            if name == "Events":
                 if 'Suitabilities' in df.columns or 'Zone' in df.columns:
                    print(f"  [NOTE] Requirement columns found in Events.")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    list_data_columns()
