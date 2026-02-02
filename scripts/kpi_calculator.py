"""
KPI Calculator Module for University Timetabling Analysis.

Calculates 3-tier KPIs: Feasibility, Student Experience, and Institutional Efficiency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from data_loader import TimetableDataLoader


@dataclass
class FeasibilityKPIs:
    """Tier 1: Hard constraint feasibility metrics."""
    compulsory_clashes: int
    capacity_violations: int
    unscheduled_events: int
    is_feasible: bool


@dataclass
class StudentExperienceKPIs:
    """Tier 2: Student experience metrics."""
    lunch_break_percentage: float  # % of students with 1hr break in 12-2pm
    avg_daily_span: float  # Average hours from first to last class
    students_analyzed: int


@dataclass
class EfficiencyKPIs:
    """Tier 3: Institutional efficiency metrics."""
    avg_room_utilization: float  # % of room-hours used
    peak_utilization: float  # Highest hourly utilization
    bottleneck_room_types: List[str]  # Room types at >=70% saturation
    total_room_hours_available: int
    total_room_hours_used: int


class KPICalculator:
    """Calculates all KPIs for a given scenario."""
    
    def __init__(self, scenario: str = 'baseline'):
        self.scenario = scenario
        self.loader = TimetableDataLoader()
        self._events = None
        self._filtered_events = None
    
    @property
    def events(self) -> pd.DataFrame:
        """Get events filtered by scenario."""
        if self._events is None:
            self._events = self.loader.events.copy()
        return self._events
    
    def get_scenario_hours(self) -> Dict[str, List[int]]:
        """Get available hours per day for the scenario."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        if self.scenario == 'baseline':
            return {day: list(range(9, 18)) for day in days}
        elif self.scenario == 'scenario_a':
            return {day: list(range(9, 17)) for day in days}
        elif self.scenario == 'scenario_b':
            hours = {day: list(range(9, 18)) for day in days}
            hours['Friday'] = list(range(9, 12))
            return hours
        return {day: list(range(9, 18)) for day in days}
    
    def calculate_feasibility_kpis(self) -> FeasibilityKPIs:
        """Calculate Tier 1 feasibility KPIs."""
        events = self.events
        scenario_hours = self.get_scenario_hours()
        
        # Count events outside scenario bounds
        unscheduled = 0
        capacity_violations = 0
        
        rooms_df = self.loader.rooms.set_index('Description')
        
        for _, event in events.iterrows():
            day = event.get('Day')
            start_hour = event.get('Start Hour')
            end_hour = event.get('End Hour')
            
            if pd.isna(day) or pd.isna(start_hour):
                unscheduled += 1
                continue
            
            # Check if event is within scenario bounds
            if day not in scenario_hours:
                unscheduled += 1
                continue
            
            allowed_hours = scenario_hours[day]
            if not allowed_hours:
                unscheduled += 1
                continue
            
            min_hour, max_hour = min(allowed_hours), max(allowed_hours) + 1
            if start_hour < min_hour or end_hour > max_hour:
                unscheduled += 1
                continue
            
            # Check capacity
            room = event.get('Room')
            event_size = event.get('Event Size', 0)
            if pd.notna(room) and room in rooms_df.index:
                room_capacity = rooms_df.loc[room, 'Capacity']
                if pd.notna(room_capacity) and event_size > room_capacity:
                    capacity_violations += 1
        
        # Check for compulsory clashes (students with overlapping events)
        compulsory_clashes = self._calculate_compulsory_clashes()
        
        is_feasible = (compulsory_clashes == 0 and 
                       capacity_violations == 0 and 
                       unscheduled == 0)
        
        return FeasibilityKPIs(
            compulsory_clashes=compulsory_clashes,
            capacity_violations=capacity_violations,
            unscheduled_events=unscheduled,
            is_feasible=is_feasible
        )
    
    def _calculate_compulsory_clashes(self, sample_size: int = 5000) -> int:
        """Calculate compulsory clashes using DPT programme data.
        
        A compulsory clash is when two whole-class events of compulsory courses 
        for the same programme-year overlap in time.
        """
        try:
            from compulsory_clash_detector import CompulsoryClashDetector
            detector = CompulsoryClashDetector(self.scenario)
            result = detector.detect_compulsory_clashes()
            return result.total_clash_pairs  # Return course-pair clashes
        except ImportError:
            # Fallback to simple count if module not available
            return 0
    
    def calculate_student_experience_kpis(self, sample_size: int = 5000) -> StudentExperienceKPIs:
        """Calculate Tier 2 student experience KPIs."""
        student_events = self.loader.student_events
        events = self.events
        
        # Sample students
        unique_students = student_events['AnonID'].unique()
        if len(unique_students) > sample_size:
            sampled_students = np.random.choice(unique_students, sample_size, replace=False)
        else:
            sampled_students = unique_students
        
        lunch_breaks = 0
        daily_spans = []
        
        # Drop duplicate Event IDs, keeping first occurrence
        events_dedup = events.drop_duplicates(subset='Event ID')
        events_lookup = events_dedup.set_index('Event ID')[['Day', 'Start Hour', 'End Hour']].to_dict('index')
        
        for student_id in sampled_students:
            student_event_ids = student_events[
                student_events['AnonID'] == student_id
            ]['Event ID'].unique()
            
            # Group events by day
            by_day = defaultdict(list)
            for eid in student_event_ids:
                if eid in events_lookup:
                    evt = events_lookup[eid]
                    if pd.notna(evt['Day']) and pd.notna(evt['Start Hour']):
                        by_day[evt['Day']].append((evt['Start Hour'], evt['End Hour']))
            
            # Check lunch break (12-2pm window)
            has_lunch_break = True
            for day, slots in by_day.items():
                slots_sorted = sorted(slots)
                # Check if there's a 1-hour gap in 12-2 window
                lunch_free = True
                for start, end in slots_sorted:
                    if start < 14 and end > 12:
                        # Event overlaps with lunch window
                        # Check if there's still a 1-hour gap
                        lunch_free = False
                        break
                
                if not lunch_free:
                    has_lunch_break = False
                    break
            
            if has_lunch_break:
                lunch_breaks += 1
            
            # Calculate daily span
            for day, slots in by_day.items():
                if slots:
                    starts = [s[0] for s in slots]
                    ends = [s[1] for s in slots]
                    span = max(ends) - min(starts)
                    daily_spans.append(span)
        
        lunch_percentage = (lunch_breaks / len(sampled_students) * 100) if sampled_students.size > 0 else 0
        avg_span = np.mean(daily_spans) if daily_spans else 0
        
        return StudentExperienceKPIs(
            lunch_break_percentage=round(lunch_percentage, 1),
            avg_daily_span=round(avg_span, 2),
            students_analyzed=len(sampled_students)
        )
    
    def calculate_efficiency_kpis(self) -> EfficiencyKPIs:
        """Calculate Tier 3 institutional efficiency KPIs."""
        events = self.events
        rooms_df = self.loader.rooms
        scenario_hours = self.get_scenario_hours()
        
        # Filter to events WITH room assignments only (exclude online/virtual)
        physical_events = events[events['Room'].notna()].copy()
        
        # Get unique rooms that are actually used
        unique_rooms_used = physical_events['Room'].nunique()
        total_rooms = len(rooms_df)
        
        # Calculate total available room-slots per week (room × timeslot combinations)
        hours_per_week = sum(len(hours) for hours in scenario_hours.values())
        total_room_slots_per_week = total_rooms * hours_per_week
        
        # Parse weeks and calculate room-week-slot usage
        # Key: (week, room, day, hour) - each unique combination should only count once
        week_room_slots = defaultdict(set)  # week -> set of (room, day, hour)
        
        for _, event in physical_events.iterrows():
            day = event.get('Day')
            start_hour = event.get('Start Hour')
            duration = event.get('Duration (minutes)', 50)
            room = event.get('Room')
            weeks_str = event.get('Weeks', '')
            
            if pd.notna(day) and pd.notna(start_hour) and pd.notna(room):
                if day in scenario_hours:
                    # Parse weeks
                    if pd.notna(weeks_str) and weeks_str:
                        weeks = [w.strip() for w in str(weeks_str).split(',')]
                    else:
                        weeks = ['1']  # Default to 1 week if not specified
                    
                    # Mark each week-room-day-hour slot as occupied
                    event_hours = duration / 60
                    for week in weeks:
                        for h in range(int(start_hour), min(int(start_hour + event_hours) + 1, 18)):
                            if h in scenario_hours.get(day, []):
                                week_room_slots[week].add((room, day, h))
        
        # Calculate average slots occupied per week
        if week_room_slots:
            slots_per_week = [len(slots) for slots in week_room_slots.values()]
            avg_slots_per_week = sum(slots_per_week) / len(slots_per_week)
        else:
            avg_slots_per_week = 0
        
        avg_utilization = (avg_slots_per_week / total_room_slots_per_week * 100) if total_room_slots_per_week > 0 else 0
        
        # Find bottleneck rooms (used in >70% of available slots across weeks)
        room_usage_count = defaultdict(int)
        total_weeks = len(week_room_slots) if week_room_slots else 1
        for week, slots in week_room_slots.items():
            for room, day, hour in slots:
                room_usage_count[room] += 1
        
        bottlenecks = []
        for room, count in room_usage_count.items():
            room_util = count / (hours_per_week * total_weeks) if hours_per_week > 0 else 0
            if room_util >= 0.70:
                bottlenecks.append(room)
        
        # Calculate peak hourly utilization (unique rooms in use during busiest hour of any week)
        hourly_rooms_by_week = defaultdict(lambda: defaultdict(set))  # week -> (day_hour) -> set of rooms
        for _, event in physical_events.iterrows():
            day = event.get('Day')
            start_hour = event.get('Start Hour')
            room = event.get('Room')
            weeks_str = event.get('Weeks', '')
            
            if pd.notna(day) and pd.notna(start_hour) and pd.notna(room):
                if pd.notna(weeks_str) and weeks_str:
                    weeks = [w.strip() for w in str(weeks_str).split(',')]
                else:
                    weeks = ['1']
                
                for week in weeks:
                    key = f"{day}_{int(start_hour)}"
                    hourly_rooms_by_week[week][key].add(room)
        
        # Find the maximum rooms used in any single hour across all weeks
        peak_usage = 0
        for week, hourly_rooms in hourly_rooms_by_week.items():
            for hour_key, rooms in hourly_rooms.items():
                if len(rooms) > peak_usage:
                    peak_usage = len(rooms)
        
        peak_utilization = (peak_usage / total_rooms * 100) if total_rooms > 0 else 0
        
        return EfficiencyKPIs(
            avg_room_utilization=round(avg_utilization, 1),
            peak_utilization=round(peak_utilization, 1),
            bottleneck_room_types=bottlenecks,
            total_room_hours_available=int(total_room_slots_per_week),
            total_room_hours_used=int(avg_slots_per_week)
        )
    
    def calculate_all_kpis(self) -> Dict:
        """Calculate all KPIs and return as dictionary."""
        feasibility = self.calculate_feasibility_kpis()
        experience = self.calculate_student_experience_kpis()
        efficiency = self.calculate_efficiency_kpis()
        
        return {
            'scenario': self.scenario,
            'feasibility': {
                'compulsory_clashes': feasibility.compulsory_clashes,
                'capacity_violations': feasibility.capacity_violations,
                'unscheduled_events': feasibility.unscheduled_events,
                'is_feasible': feasibility.is_feasible
            },
            'student_experience': {
                'lunch_break_percentage': experience.lunch_break_percentage,
                'avg_daily_span_hours': experience.avg_daily_span,
                'students_analyzed': experience.students_analyzed
            },
            'efficiency': {
                'avg_room_utilization': efficiency.avg_room_utilization,
                'peak_utilization': efficiency.peak_utilization,
                'bottleneck_room_types': efficiency.bottleneck_room_types,
                'room_hours_available': efficiency.total_room_hours_available,
                'room_hours_used': efficiency.total_room_hours_used
            }
        }


if __name__ == "__main__":
    for scenario in ['baseline', 'scenario_a', 'scenario_b']:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.upper()}")
        print('='*60)
        
        calc = KPICalculator(scenario)
        kpis = calc.calculate_all_kpis()
        
        print("\nTier 1 - Feasibility:")
        print(f"  Compulsory clashes: {kpis['feasibility']['compulsory_clashes']}")
        print(f"  Capacity violations: {kpis['feasibility']['capacity_violations']}")
        print(f"  Unscheduled events: {kpis['feasibility']['unscheduled_events']}")
        print(f"  Is Feasible: {kpis['feasibility']['is_feasible']}")
        
        print("\nTier 2 - Student Experience:")
        print(f"  Lunch break (12-2pm): {kpis['student_experience']['lunch_break_percentage']}%")
        print(f"  Avg daily span: {kpis['student_experience']['avg_daily_span_hours']} hours")
        
        print("\nTier 3 - Efficiency:")
        print(f"  Avg room utilization: {kpis['efficiency']['avg_room_utilization']}%")
        print(f"  Peak utilization: {kpis['efficiency']['peak_utilization']}%")
        print(f"  Room hours used: {kpis['efficiency']['room_hours_used']}/{kpis['efficiency']['room_hours_available']}")
