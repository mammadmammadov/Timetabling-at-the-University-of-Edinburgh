
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\gulma\Desktop\Timetabling-at-the-University-of-Edinburgh\Timetabling-at-the-University-of-Edinburgh")
DATA_RAW = PROJECT_ROOT / "data" / "raw"

def inspect_rooms_sheets():
    filepath = DATA_RAW / "Rooms_and_Room_Types.xlsx"
    print(f"Inspecting: {filepath}")
    
    # Get all sheet names
    xls = pd.ExcelFile(filepath)
    print(f"\nSheets found: {xls.sheet_names}")
    
    for sheet_name in xls.sheet_names:
        print(f"\n--- Sheet: '{sheet_name}' ---")
        df = pd.read_excel(xls, sheet_name=sheet_name, nrows=5)
        print(f"Columns ({len(df.columns)}):")
        for col in df.columns:
            print(f"  - {col}")
        print(f"Sample data:")
        print(df.head(3).to_string())

if __name__ == "__main__":
    inspect_rooms_sheets()
