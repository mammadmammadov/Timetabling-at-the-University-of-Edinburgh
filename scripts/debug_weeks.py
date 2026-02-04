
import pandas as pd
from pathlib import Path
import re

# Define paths
PROJECT_ROOT = Path(r"c:\Users\gulma\Desktop\Timetabling-at-the-University-of-Edinburgh\Timetabling-at-the-University-of-Edinburgh")
DATA_RAW = PROJECT_ROOT / "data" / "raw"

def analyze_weeks_formats():
    print("Loading Event Data...")
    try:
        df = pd.read_excel(DATA_RAW / "2024-5_Event_Module_Room.xlsx")
        
        if 'Weeks' not in df.columns:
            print("Error: 'Weeks' column not found.")
            return
            
        print(f"Total rows: {len(df)}")
        
        # Get unique non-null week strings
        unique_weeks = df['Weeks'].dropna().unique()
        print(f"Unique 'Weeks' formats found: {len(unique_weeks)}")
        
        print("\n--- Sample Formats ---")
        for w in list(unique_weeks)[:20]:
            print(f"'{w}'")
            
        print("\n--- Analyzing unrecognized formats ---")
        # Reuse logic from current fix to see what fails
        # Logic: split by ',', then check for '-' or int
        
        unparsed = []
        for w_str in unique_weeks:
            is_valid = True
            parts = str(w_str).split(',')
            for part in parts:
                part = part.strip()
                if not part: continue
                
                if '-' in part:
                    try:
                        s, e = part.split('-')
                        int(s), int(e)
                    except:
                        is_valid = False
                else:
                    try:
                        int(part)
                    except:
                        is_valid = False
            
            if not is_valid:
                unparsed.append(w_str)
        
        if unparsed:
            print(f"\nPotential parsing failures ({len(unparsed)}):")
            for w in unparsed[:20]:
                print(f"FAIL: '{w}'")
        else:
            print("\nAll formats appear compatible with current logic.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_weeks_formats()
