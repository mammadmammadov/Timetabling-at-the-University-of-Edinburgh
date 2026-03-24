import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import re

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


class TimetableDataLoader:
    """loading and processing timetabling data files."""
    
    def __init__(self):
        self._events_df: Optional[pd.DataFrame] = None
        self._rooms_df: Optional[pd.DataFrame] = None
        self._student_events_df: Optional[pd.DataFrame] = None
        self._dpt_df: Optional[pd.DataFrame] = None
        self._programme_course_df: Optional[pd.DataFrame] = None
        self._travel_times: Optional[Dict[Tuple[str, str], int]] = None
    
    @property
    def events(self) -> pd.DataFrame:
        """loading and caching events data with parsed timeslots."""
        if self._events_df is None:
            self._events_df = self._load_events()
        return self._events_df
    
    @property
    def rooms(self) -> pd.DataFrame:
        """loading and caching rooms data."""
        if self._rooms_df is None:
            df = pd.read_excel(DATA_RAW / "Rooms_and_Room_Types.xlsx")
            if 'Campus' in df.columns:
                df['Campus'] = df['Campus'].str.title()
                # special cases 
                df['Campus'] = df['Campus'].replace('Bioquarter', 'BioQuarter')
            self._rooms_df = df
        return self._rooms_df
    
    @property
    def student_events(self) -> pd.DataFrame:
        """loading and caching student-event mappings."""
        if self._student_events_df is None:
            self._student_events_df = pd.read_excel(
                DATA_RAW / "2024-5_Student_Programme_Module_Event.xlsx"
            )
        return self._student_events_df
    
    @property
    def dpt_data(self) -> pd.DataFrame:
        """loading and caching DPT programme data."""
        if self._dpt_df is None:
            self._dpt_df = pd.read_excel(DATA_RAW / "2024-5_DPT_Data.xlsx")
        return self._dpt_df
    
    @property
    def programme_courses(self) -> pd.DataFrame:
        """loading and caching programme-course mappings."""
        if self._programme_course_df is None:
            self._programme_course_df = pd.read_excel(DATA_RAW / "Programme-Course.xlsx")
        return self._programme_course_df
    
    @property
    def travel_times(self) -> Dict[Tuple[str, str], int]:
        """loading and caching inter-campus travel time matrix (minutes)."""
        if self._travel_times is None:
            self._travel_times = {}
            try:
                rc = pd.read_excel(DATA_RAW / "Rooms_and_Room_Types.xlsx",
                                   sheet_name='Room Constraints')
                for _, row in rc.iterrows():
                    cfrom = row.get('Campus From')
                    cto = row.get('Campus To')
                    mins = row.get('Travel time (mins)')
                    if pd.notna(cfrom) and pd.notna(cto) and pd.notna(mins):
                        # Standardise campus names (same logic as rooms/events)
                        cfrom = str(cfrom).strip().title().replace('Bioquarter', 'BioQuarter')
                        cto = str(cto).strip().title().replace('Bioquarter', 'BioQuarter')
                        self._travel_times[(cfrom, cto)] = int(mins)
            except Exception as e:
                print(f"Warning: Could not load travel times: {e}")
        return self._travel_times
    
    def _load_events(self) -> pd.DataFrame:
        """loading and parsing events and timeslot information."""
        df = pd.read_excel(DATA_RAW / "2024-5_Event_Module_Room.xlsx")
        
        # Parse timeslots into day and hour components
        df = self._parse_timeslots(df)
        
        # Calculate end time based on duration
        df['End Hour'] = df.apply(
            lambda r: r['Start Hour'] + (r['Duration (minutes)'] / 60) if pd.notna(r['Start Hour']) else None,
            axis=1
        )
        
        # Standardize campus names
        if 'Campus' in df.columns:
            df['Campus'] = df['Campus'].str.title()
            df['Campus'] = df['Campus'].replace('Bioquarter', 'BioQuarter')
            
        # Compute Effective Size (cap at original room capacity)
        try:
            rooms_df = self.rooms.set_index('Id')
            
            def calc_effective_size(row):
                size = row.get('Event Size', 0)
                if pd.isna(size):
                    size = 0
                
                room = row.get('Room')
                if pd.notna(room) and room in rooms_df.index:
                    cap = rooms_df.loc[room, 'Capacity']
                    if isinstance(cap, pd.Series):
                        cap = cap.iloc[0]
                    if pd.notna(cap):
                        return min(size, cap)
                return size
                
            df['Effective Size'] = df.apply(calc_effective_size, axis=1)
        except Exception as e:
            print(f"Warning: Could not compute Effective Size: {e}")
            df['Effective Size'] = df['Event Size'].fillna(0)
        
        return df
    
    def _parse_timeslots(self, df: pd.DataFrame) -> pd.DataFrame:
        """parsing 'Timeslot' column into 'Day' and 'Start Hour' columns."""
        
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
        """getting all events scheduled at a specific day and hour."""
        events = self.events
        mask = (events['Day'] == day) & (events['Start Hour'] <= hour) & (events['End Hour'] > hour)
        return events[mask]
    
    def get_room_capacity(self, room_name: str) -> Optional[int]:
        """getting capacity for a specific room."""
        rooms = self.rooms
        match = rooms[rooms['Description'] == room_name]
        if len(match) > 0:
            return match.iloc[0]['Capacity']
        return None
    
    def get_student_events_by_student(self, student_id: str) -> pd.DataFrame:
        """getting all events for a specific student."""
        return self.student_events[self.student_events['AnonID'] == student_id]
    
    def get_unique_students(self) -> np.ndarray:
        """getting array of unique student IDs."""
        return self.student_events['AnonID'].unique()
    
    def get_events_by_scenario(self, scenario: str) -> Dict[str, pd.DataFrame]:
        """
        splitting events into 'within_bounds' and 'displaced' based on scenario.
        
        scenarios:
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
    """getting summary statistics for all data files."""
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
    print("loading data...")
    loader = TimetableDataLoader()
    
    print(f"\nevents: {len(loader.events):,} records")
    print(f"rooms: {len(loader.rooms):,} records")
    print(f"student-events: {len(loader.student_events):,} records")
    
    print("\nSample parsed timeslots:")
    print(loader.events[['Timeslot', 'Day', 'Start Hour', 'End Hour', 'Duration (minutes)']].head(10))
    
    print("\nEvents by day:")
    print(loader.events['Day'].value_counts())
