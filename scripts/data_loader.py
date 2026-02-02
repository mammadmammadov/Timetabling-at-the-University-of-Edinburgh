"""
Data Loading and Preprocessing Module for University Timetabling Analysis.

Loads Excel files, parses timeslots, and prepares data for CSP analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from functools import lru_cache
import re

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


class TimetableDataLoader:
    """Loads and processes all timetabling data files."""
    
    def __init__(self):
        self._events_df: Optional[pd.DataFrame] = None
        self._rooms_df: Optional[pd.DataFrame] = None
        self._student_events_df: Optional[pd.DataFrame] = None
        self._dpt_df: Optional[pd.DataFrame] = None
        self._programme_course_df: Optional[pd.DataFrame] = None
    
    @property
    def events(self) -> pd.DataFrame:
        """Load and cache events data with parsed timeslots."""
        if self._events_df is None:
            self._events_df = self._load_events()
        return self._events_df
    
    @property
    def rooms(self) -> pd.DataFrame:
        """Load and cache rooms data."""
        if self._rooms_df is None:
            self._rooms_df = pd.read_excel(DATA_RAW / "Rooms_and_Room_Types.xlsx")
        return self._rooms_df
    
    @property
    def student_events(self) -> pd.DataFrame:
        """Load and cache student-event mappings."""
        if self._student_events_df is None:
            self._student_events_df = pd.read_excel(
                DATA_RAW / "2024-5_Student_Programme_Module_Event.xlsx"
            )
        return self._student_events_df
    
    @property
    def dpt_data(self) -> pd.DataFrame:
        """Load and cache DPT programme data."""
        if self._dpt_df is None:
            self._dpt_df = pd.read_excel(DATA_RAW / "2024-5_DPT_Data.xlsx")
        return self._dpt_df
    
    @property
    def programme_courses(self) -> pd.DataFrame:
        """Load and cache programme-course mappings."""
        if self._programme_course_df is None:
            self._programme_course_df = pd.read_excel(DATA_RAW / "Programme-Course.xlsx")
        return self._programme_course_df
    
    def _load_events(self) -> pd.DataFrame:
        """Load events and parse timeslot information."""
        df = pd.read_excel(DATA_RAW / "2024-5_Event_Module_Room.xlsx")
        
        # Parse timeslots into day and hour components
        df = self._parse_timeslots(df)
        
        # Calculate end time based on duration
        df['End Hour'] = df.apply(
            lambda r: r['Start Hour'] + (r['Duration (minutes)'] / 60) if pd.notna(r['Start Hour']) else None,
            axis=1
        )
        
        return df
    
    def _parse_timeslots(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse 'Timeslot' column into 'Day' and 'Start Hour' columns."""
        
        def parse_slot(slot: str) -> Tuple[Optional[str], Optional[int]]:
            if pd.isna(slot):
                return None, None
            
            # Pattern: "Day HH:MM" e.g., "Tuesday 11:00"
            match = re.match(r'(\w+)\s+(\d{1,2}):(\d{2})', str(slot))
            if match:
                day = match.group(1)
                hour = int(match.group(2))
                minute = int(match.group(3))
                return day, hour + minute / 60
            return None, None
        
        parsed = df['Timeslot'].apply(parse_slot)
        df['Day'] = parsed.apply(lambda x: x[0])
        df['Start Hour'] = parsed.apply(lambda x: x[1])
        
        # Map days to numeric for sorting/analysis
        day_order = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        df['Day Number'] = df['Day'].map(day_order)
        
        return df
    
    def get_events_in_timeslot(self, day: str, hour: int) -> pd.DataFrame:
        """Get all events scheduled at a specific day and hour."""
        events = self.events
        mask = (events['Day'] == day) & (events['Start Hour'] <= hour) & (events['End Hour'] > hour)
        return events[mask]
    
    def get_room_capacity(self, room_name: str) -> Optional[int]:
        """Get capacity for a specific room."""
        rooms = self.rooms
        match = rooms[rooms['Description'] == room_name]
        if len(match) > 0:
            return match.iloc[0]['Capacity']
        return None
    
    def get_student_events_by_student(self, student_id: str) -> pd.DataFrame:
        """Get all events for a specific student."""
        return self.student_events[self.student_events['AnonID'] == student_id]
    
    def get_unique_students(self) -> np.ndarray:
        """Get array of unique student IDs."""
        return self.student_events['AnonID'].unique()
    
    def get_events_by_scenario(self, scenario: str) -> Dict[str, pd.DataFrame]:
        """
        Split events into 'within_bounds' and 'displaced' based on scenario.
        
        Scenarios:
        - 'baseline': Mon-Fri 9am-6pm
        - 'scenario_a': Mon-Fri 9am-5pm
        - 'scenario_b': Mon-Thu 9am-6pm, Fri 9am-12pm
        """
        events = self.events.copy()
        
        if scenario == 'baseline':
            # Current policy: Mon-Fri 9-6
            displaced_mask = (
                (events['Day'].isin(['Saturday', 'Sunday'])) |
                (events['Start Hour'] < 9) |
                (events['End Hour'] > 18)
            )
        elif scenario == 'scenario_a':
            # 9-5 Mon-Fri
            displaced_mask = (
                (events['Day'].isin(['Saturday', 'Sunday'])) |
                (events['Start Hour'] < 9) |
                (events['End Hour'] > 17)  # Cut at 5pm
            )
        elif scenario == 'scenario_b':
            # Mon-Thu 9-6, Fri 9-12
            friday_cut = (events['Day'] == 'Friday') & (events['Start Hour'] >= 12)
            weekend = events['Day'].isin(['Saturday', 'Sunday'])
            outside_hours = (events['Start Hour'] < 9) | (events['End Hour'] > 18)
            displaced_mask = friday_cut | weekend | outside_hours
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        return {
            'within_bounds': events[~displaced_mask],
            'displaced': events[displaced_mask]
        }


def get_data_summary() -> Dict:
    """Get summary statistics for all data files."""
    loader = TimetableDataLoader()
    
    summary = {
        'events': {
            'total': len(loader.events),
            'with_timeslot': loader.events['Timeslot'].notna().sum(),
            'unique_rooms': loader.events['Room'].nunique(),
            'unique_modules': loader.events['Module Code'].nunique(),
        },
        'rooms': {
            'total': len(loader.rooms),
            'total_capacity': loader.rooms['Capacity'].sum(),
        },
        'students': {
            'unique_students': loader.student_events['AnonID'].nunique(),
            'total_enrollments': len(loader.student_events),
        }
    }
    
    return summary


if __name__ == "__main__":
    # Test loading
    print("Loading data...")
    loader = TimetableDataLoader()
    
    print(f"\nEvents: {len(loader.events):,} records")
    print(f"Rooms: {len(loader.rooms):,} records")
    print(f"Student-Events: {len(loader.student_events):,} records")
    
    print("\nSample parsed timeslots:")
    print(loader.events[['Timeslot', 'Day', 'Start Hour', 'End Hour', 'Duration (minutes)']].head(10))
    
    print("\nEvents by day:")
    print(loader.events['Day'].value_counts())
