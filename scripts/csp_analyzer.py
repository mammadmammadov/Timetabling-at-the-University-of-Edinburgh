"""
CSP Analyzer Module for University Timetabling.

Models the timetabling problem as a Constraint Satisfaction Problem (CSP)
with optimization via Simulated Annealing for displaced events.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import random
import math

from data_loader import TimetableDataLoader


@dataclass
class TimeSlot:
    """Represents a teaching timeslot."""
    day: str
    start_hour: float
    end_hour: float
    
    def overlaps(self, other: 'TimeSlot') -> bool:
        """Check if this timeslot overlaps with another."""
        if self.day != other.day:
            return False
        return not (self.end_hour <= other.start_hour or self.start_hour >= other.end_hour)
    
    def __hash__(self):
        return hash((self.day, self.start_hour, self.end_hour))


@dataclass
class Event:
    """Represents a teaching event."""
    event_id: str
    module_code: str
    duration_minutes: int
    event_size: int
    event_type: str
    is_whole_class: bool
    assigned_slot: Optional[TimeSlot] = None
    assigned_room: Optional[str] = None


@dataclass
class Room:
    """Represents a teaching room."""
    name: str
    capacity: int
    room_type: str
    building: str


@dataclass
class CSPResult:
    """Result of a CSP feasibility check."""
    is_feasible: bool
    events_scheduled: int
    events_unscheduled: int
    capacity_violations: int
    clash_count: int
    binding_constraints: List[str] = field(default_factory=list)


class TimetableCSP:
    """CSP model for timetabling with optimization."""
    
    def __init__(self, scenario: str):
        self.scenario = scenario
        self.loader = TimetableDataLoader()
        self.available_slots: List[TimeSlot] = []
        self.rooms: Dict[str, Room] = {}
        self.events: Dict[str, Event] = {}
        self.room_schedule: Dict[str, Dict[TimeSlot, str]] = defaultdict(dict)  # room -> {slot -> event_id}
        self.student_schedules: Dict[str, List[TimeSlot]] = defaultdict(list)
        
        self._init_available_slots()
        self._init_rooms()
        self._init_events()
    
    def _init_available_slots(self):
        """Initialize available timeslots based on scenario."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        for day in days:
            if self.scenario == 'baseline':
                hours = range(9, 18)  # 9am-6pm
            elif self.scenario == 'scenario_a':
                hours = range(9, 17)  # 9am-5pm
            elif self.scenario == 'scenario_b':
                if day == 'Friday':
                    hours = range(9, 12)  # 9am-12pm on Friday
                else:
                    hours = range(9, 18)  # 9am-6pm Mon-Thu
            else:
                hours = range(9, 18)
            
            for hour in hours:
                self.available_slots.append(TimeSlot(day, hour, hour + 1))
    
    def _init_rooms(self):
        """Initialize room data."""
        rooms_df = self.loader.rooms
        for _, row in rooms_df.iterrows():
            room = Room(
                name=str(row['Description']),
                capacity=int(row['Capacity']) if pd.notna(row['Capacity']) else 0,
                room_type=str(row.get('Room Type', 'Unknown')),
                building=str(row.get('Building', 'Unknown'))
            )
            self.rooms[room.name] = room
    
    def _init_events(self):
        """Initialize events from data."""
        events_df = self.loader.events
        for _, row in events_df.iterrows():
            event_id = str(row['Event ID'])
            event = Event(
                event_id=event_id,
                module_code=str(row['Module Code']),
                duration_minutes=int(row['Duration (minutes)']) if pd.notna(row['Duration (minutes)']) else 50,
                event_size=int(row['Event Size']) if pd.notna(row['Event Size']) else 0,
                event_type=str(row['Event Type']),
                is_whole_class=bool(row['WholeClass']) if pd.notna(row['WholeClass']) else False
            )
            
            # Set current assignment if within scenario bounds
            if pd.notna(row['Day']) and pd.notna(row['Start Hour']):
                slot = TimeSlot(row['Day'], row['Start Hour'], row['End Hour'])
                if self._is_slot_in_bounds(slot):
                    event.assigned_slot = slot
                    event.assigned_room = str(row['Room']) if pd.notna(row['Room']) else None
            
            self.events[event_id] = event
    
    def _is_slot_in_bounds(self, slot: TimeSlot) -> bool:
        """Check if a timeslot is within scenario bounds."""
        if slot.day in ['Saturday', 'Sunday']:
            return False
        
        if self.scenario == 'baseline':
            return 9 <= slot.start_hour and slot.end_hour <= 18
        elif self.scenario == 'scenario_a':
            return 9 <= slot.start_hour and slot.end_hour <= 17
        elif self.scenario == 'scenario_b':
            if slot.day == 'Friday':
                return 9 <= slot.start_hour and slot.end_hour <= 12
            return 9 <= slot.start_hour and slot.end_hour <= 18
        return False
    
    def get_displaced_events(self) -> List[Event]:
        """Get events that need to be rescheduled."""
        return [e for e in self.events.values() if e.assigned_slot is None]
    
    def get_scheduled_events(self) -> List[Event]:
        """Get events that are currently scheduled."""
        return [e for e in self.events.values() if e.assigned_slot is not None]
    
    def check_hard_constraints(self) -> CSPResult:
        """Check all hard constraints and return feasibility result."""
        capacity_violations = 0
        clashes = 0
        binding = []
        
        # Build room schedules
        room_usage = defaultdict(list)  # room -> [(slot, event)]
        
        for event in self.get_scheduled_events():
            if event.assigned_room and event.assigned_slot:
                room_usage[event.assigned_room].append((event.assigned_slot, event))
                
                # Check capacity
                room = self.rooms.get(event.assigned_room)
                if room and event.event_size > room.capacity:
                    capacity_violations += 1
        
        # Check for room double-booking
        for room_name, bookings in room_usage.items():
            for i, (slot1, evt1) in enumerate(bookings):
                for slot2, evt2 in bookings[i+1:]:
                    if slot1.overlaps(slot2):
                        clashes += 1
        
        unscheduled = len(self.get_displaced_events())
        scheduled = len(self.get_scheduled_events())
        
        is_feasible = (capacity_violations == 0 and clashes == 0 and unscheduled == 0)
        
        if capacity_violations > 0:
            binding.append(f"Room capacity violations: {capacity_violations}")
        if clashes > 0:
            binding.append(f"Room double-bookings: {clashes}")
        if unscheduled > 0:
            binding.append(f"Unscheduled events: {unscheduled}")
        
        return CSPResult(
            is_feasible=is_feasible,
            events_scheduled=scheduled,
            events_unscheduled=unscheduled,
            capacity_violations=capacity_violations,
            clash_count=clashes,
            binding_constraints=binding
        )
    
    def find_available_slot(self, event: Event) -> Optional[Tuple[TimeSlot, str]]:
        """Find an available slot and room for an event using greedy search."""
        duration_hours = event.duration_minutes / 60
        
        for slot in self.available_slots:
            # Create extended slot for event duration
            extended_slot = TimeSlot(slot.day, slot.start_hour, slot.start_hour + duration_hours)
            
            if not self._is_slot_in_bounds(extended_slot):
                continue
            
            # Find suitable room
            for room_name, room in self.rooms.items():
                if room.capacity < event.event_size:
                    continue
                
                # Check if room is available
                is_available = True
                for (booked_slot, _) in self.room_schedule.get(room_name, {}).items():
                    if extended_slot.overlaps(booked_slot):
                        is_available = False
                        break
                
                if is_available:
                    return (extended_slot, room_name)
        
        return None
    
    def greedy_reschedule(self) -> int:
        """Attempt to reschedule displaced events using greedy approach."""
        displaced = self.get_displaced_events()
        scheduled_count = 0
        
        # Sort by size (larger events first - harder to place)
        displaced.sort(key=lambda e: -e.event_size)
        
        for event in displaced:
            result = self.find_available_slot(event)
            if result:
                slot, room = result
                event.assigned_slot = slot
                event.assigned_room = room
                self.room_schedule[room][slot] = event.event_id
                scheduled_count += 1
        
        return scheduled_count
    
    def simulated_annealing(self, max_iterations: int = 1000, initial_temp: float = 100.0) -> int:
        """
        Use Simulated Annealing to optimize event placement.
        Returns number of additional events scheduled.
        """
        displaced = self.get_displaced_events()
        if not displaced:
            return 0
        
        initial_unscheduled = len(displaced)
        temp = initial_temp
        cooling_rate = 0.995
        
        for iteration in range(max_iterations):
            # Try to schedule a random displaced event
            if not displaced:
                break
            
            event = random.choice(displaced)
            result = self.find_available_slot(event)
            
            if result:
                slot, room = result
                event.assigned_slot = slot
                event.assigned_room = room
                self.room_schedule[room][slot] = event.event_id
                displaced.remove(event)
            else:
                # With probability based on temperature, try to swap with existing
                if random.random() < math.exp(-1 / max(temp, 0.01)):
                    # Attempt swap logic here (simplified)
                    pass
            
            temp *= cooling_rate
        
        return initial_unscheduled - len(displaced)


def analyze_scenario(scenario: str) -> Dict:
    """Run full CSP analysis for a scenario."""
    print(f"\nAnalyzing {scenario}...")
    
    csp = TimetableCSP(scenario)
    
    # Initial state
    initial_displaced = len(csp.get_displaced_events())
    initial_scheduled = len(csp.get_scheduled_events())
    
    # Check constraints before optimization
    initial_result = csp.check_hard_constraints()
    
    # Try to reschedule displaced events
    greedy_scheduled = csp.greedy_reschedule()
    sa_scheduled = csp.simulated_annealing()
    
    # Final check
    final_result = csp.check_hard_constraints()
    
    return {
        'scenario': scenario,
        'initial_scheduled': initial_scheduled,
        'initial_displaced': initial_displaced,
        'greedy_rescheduled': greedy_scheduled,
        'sa_rescheduled': sa_scheduled,
        'final_scheduled': final_result.events_scheduled,
        'final_unscheduled': final_result.events_unscheduled,
        'is_feasible': final_result.is_feasible,
        'capacity_violations': final_result.capacity_violations,
        'clash_count': final_result.clash_count,
        'binding_constraints': final_result.binding_constraints
    }


if __name__ == "__main__":
    for scenario in ['baseline', 'scenario_a', 'scenario_b']:
        result = analyze_scenario(scenario)
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario}")
        print(f"Initial: {result['initial_scheduled']} scheduled, {result['initial_displaced']} displaced")
        print(f"After optimization: {result['final_scheduled']} scheduled, {result['final_unscheduled']} unscheduled")
        print(f"Feasible: {result['is_feasible']}")
        if result['binding_constraints']:
            print(f"Binding constraints: {result['binding_constraints']}")
