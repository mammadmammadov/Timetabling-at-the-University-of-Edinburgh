"""
Compulsory Clash Detection Module.

Identifies true compulsory clashes - where whole-class events of compulsory 
courses for the same programme-year overlap.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
import re

from data_loader import TimetableDataLoader, DATA_RAW


@dataclass
class CompulsoryClashResult:
    """Result of compulsory clash analysis."""
    total_programme_years: int
    programmes_with_clashes: int
    total_clash_pairs: int  # Number of course-pair clashes
    total_clash_instances: int  # Total overlapping event instances
    clash_details: List[Dict]  # Detailed clash info


class CompulsoryClashDetector:
    """Detects clashes between compulsory whole-class events."""
    
    def __init__(self, scenario: str = 'baseline'):
        self.scenario = scenario
        self.loader = TimetableDataLoader()
        self._dpt_df = None
        self._compulsory_by_programme = None
        
    @property
    def dpt_data(self) -> pd.DataFrame:
        """Load DPT data with programme-course mappings."""
        if self._dpt_df is None:
            self._dpt_df = pd.read_excel(DATA_RAW / "2024-5_DPT_Data.xlsx")
        return self._dpt_df
    
    def _extract_course_code(self, module_code: str) -> str:
        """Extract base course code from module code.
        
        Example: 'ACCN08007_SS1_SEM1_2024/5' -> 'ACCN08007'
        """
        if pd.isna(module_code):
            return None
        # Take everything before the first underscore
        parts = str(module_code).split('_')
        return parts[0] if parts else None
    
    def get_compulsory_courses_by_programme(self) -> Dict[str, Set[str]]:
        """Get compulsory courses for each programme-year combination.
        
        CRITICAL: Filters out courses that have NO actual student enrollments 
        for that programme-year, even if listed as compulsory in DPT.
        """
        if self._compulsory_by_programme is not None:
            return self._compulsory_by_programme
        
        # 1. Get theoretical compulsory courses from DPT
        dpt = self.dpt_data
        compulsory = dpt[dpt['Compulsory/Optional'] == 'Compulsory']
        
        theoretical_map = defaultdict(set)
        for _, row in compulsory.iterrows():
            prog_code = row['Programme Code']
            # Handle different column names for Year
            prog_year = row.get('Programme Year', row.get('ProgYear', ''))
            
            # Construct key matching student data format: CODE_YEAR
            key = f"{prog_code}_{prog_year}"
            course = row['Course Code']
            theoretical_map[key].add(course)
            
        # 2. Get actual enrollments from Student Data
        student_events = self.loader.student_events
        
        # Check if we have the right columns
        if 'Programme Code-Year' not in student_events.columns or 'Course ID' not in student_events.columns:
            print("Warning: Missing columns in student data. Using DPT only.")
            self._compulsory_by_programme = dict(theoretical_map)
            return self._compulsory_by_programme
            
        # Get set of (Programme_Year, Course) that actually exist
        # We use a set lookup for O(1) access
        valid_pairs = set()
        
        # Group by Programme-Year to avoid massive unique check on whole DF
        # This is more efficient: get unique courses per programme-year
        existing_enrollments = student_events[['Programme Code-Year', 'Course ID']].drop_duplicates()
        
        for _, row in existing_enrollments.iterrows():
            p_key = str(row['Programme Code-Year']).strip()
            # Course ID might be full module code, extract base if needed, 
            # but DPT usually has base. Let's assume Course ID in student data matches DPT Course Code 
            # or starts with it. DPT: CMSE11509. Student: CMSE11509...
            c_id = str(row['Course ID'])
            valid_pairs.add((p_key, c_id))
            
        # 3. Intersect
        self._compulsory_by_programme = {}
        
        for key, courses in theoretical_map.items():
            active_courses = set()
            for course in courses:
                # Check if this course (or a variant of it) exists in valid_pairs for this key
                # We check stricter first: exact match
                if (key, course) in valid_pairs:
                    active_courses.add(course)
                else:
                    # Loose check: verify if any course in valid_pairs for this key starts with the DPT course code
                    # (Optimized: we could pre-filter valid_pairs by key, but let's do a generator check here)
                    # This is O(N_enrolled_in_prog)
                    has_enrollment = any(
                        vp_course.startswith(course) 
                        for vp_key, vp_course in valid_pairs 
                        if vp_key == key
                    )
                    if has_enrollment:
                        active_courses.add(course)
                        
            if active_courses:
                self._compulsory_by_programme[key] = active_courses
        
        return self._compulsory_by_programme
    
    def get_whole_class_events(self) -> pd.DataFrame:
        """Get events that are whole-class (lectures, etc.) and parse timeslots."""
        events = self.loader.events.copy()
        
        # Filter to whole-class events OR lectures specifically
        whole_class_mask = (events['WholeClass'] == True) | (events['Event Type'] == 'Lecture')
        wc_events = events[whole_class_mask].copy()
        
        # Extract base course code
        wc_events['Course Code'] = wc_events['Module Code'].apply(self._extract_course_code)
        
        return wc_events
    
    def _events_overlap(self, evt1: pd.Series, evt2: pd.Series) -> bool:
        """Check if two events overlap in time."""
        if evt1['Day'] != evt2['Day']:
            return False
        if pd.isna(evt1['Start Hour']) or pd.isna(evt2['Start Hour']):
            return False
        
        # Check time overlap
        start1, end1 = evt1['Start Hour'], evt1['End Hour']
        start2, end2 = evt2['Start Hour'], evt2['End Hour']
        
        return not (end1 <= start2 or start1 >= end2)
    
    def _is_event_in_scenario(self, event: pd.Series) -> bool:
        """Check if event is within scenario time bounds."""
        day = event.get('Day')
        start_hour = event.get('Start Hour')
        end_hour = event.get('End Hour')
        
        if pd.isna(day) or pd.isna(start_hour):
            return False
        
        if day in ['Saturday', 'Sunday']:
            return False
        
        if self.scenario == 'baseline':
            return 9 <= start_hour and end_hour <= 18
        elif self.scenario == 'scenario_a':
            return 9 <= start_hour and end_hour <= 17
        elif self.scenario == 'scenario_b':
            if day == 'Friday':
                return 9 <= start_hour and end_hour <= 12
            return 9 <= start_hour and end_hour <= 18
        
        return True
    
    def detect_compulsory_clashes(self) -> CompulsoryClashResult:
        """Detect all compulsory clashes across programme-years."""
        compulsory_by_prog = self.get_compulsory_courses_by_programme()
        wc_events = self.get_whole_class_events()
        
        # Filter events within scenario bounds
        wc_events = wc_events[wc_events.apply(self._is_event_in_scenario, axis=1)]
        
        # Build course -> events mapping
        course_events = defaultdict(list)
        for idx, event in wc_events.iterrows():
            course_code = event['Course Code']
            if course_code:
                course_events[course_code].append(event)
        
        # Check each programme-year for clashes
        programmes_with_clashes = 0
        total_clash_pairs = 0
        total_clash_instances = 0
        clash_details = []
        
        for prog_key, courses in compulsory_by_prog.items():
            courses_list = list(courses)
            prog_has_clash = False
            
            # Check all pairs of compulsory courses
            for i, course1 in enumerate(courses_list):
                events1 = course_events.get(course1, [])
                
                for course2 in courses_list[i+1:]:
                    events2 = course_events.get(course2, [])
                    
                    # Check for overlapping events
                    clash_count = 0
                    for evt1 in events1:
                        for evt2 in events2:
                            if self._events_overlap(evt1, evt2):
                                clash_count += 1
                    
                    if clash_count > 0:
                        prog_has_clash = True
                        total_clash_pairs += 1
                        total_clash_instances += clash_count
                        
                        # Store details (limit to avoid memory issues)
                        if len(clash_details) < 100:
                            clash_details.append({
                                'programme': prog_key,
                                'course1': course1,
                                'course2': course2,
                                'clash_count': clash_count
                            })
            
            if prog_has_clash:
                programmes_with_clashes += 1
        
        return CompulsoryClashResult(
            total_programme_years=len(compulsory_by_prog),
            programmes_with_clashes=programmes_with_clashes,
            total_clash_pairs=total_clash_pairs,
            total_clash_instances=total_clash_instances,
            clash_details=clash_details
        )


def analyze_compulsory_clashes(scenario: str = 'baseline') -> Dict:
    """Run compulsory clash analysis for a scenario."""
    print(f"\nAnalyzing compulsory clashes for {scenario}...")
    
    detector = CompulsoryClashDetector(scenario)
    result = detector.detect_compulsory_clashes()
    
    return {
        'scenario': scenario,
        'total_programme_years': result.total_programme_years,
        'programmes_with_clashes': result.programmes_with_clashes,
        'clash_rate': round(result.programmes_with_clashes / result.total_programme_years * 100, 1) if result.total_programme_years > 0 else 0,
        'total_clash_pairs': result.total_clash_pairs,
        'total_clash_instances': result.total_clash_instances,
        'sample_clashes': result.clash_details[:10]
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Compulsory Clash Detection Analysis")
    print("=" * 70)
    
    for scenario in ['baseline', 'scenario_a', 'scenario_b']:
        result = analyze_compulsory_clashes(scenario)
        
        print(f"\n{scenario.upper()}")
        print("-" * 40)
        print(f"  Programme-years analyzed: {result['total_programme_years']}")
        print(f"  Programmes with clashes: {result['programmes_with_clashes']} ({result['clash_rate']}%)")
        print(f"  Total clash pairs (course combinations): {result['total_clash_pairs']}")
        print(f"  Total clash instances (event overlaps): {result['total_clash_instances']}")
        
        if result['sample_clashes']:
            print(f"\n  Sample clashes:")
            for clash in result['sample_clashes'][:5]:
                print(f"    - {clash['programme']}: {clash['course1']} vs {clash['course2']} ({clash['clash_count']} overlaps)")
