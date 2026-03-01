"""
Debug script to find and display double-booking statistics.

Double bookings are known to be intentional at Edinburgh (Maths studio pods,
ECA project spaces, Chemistry/Engineering labs, Medicine LT block-bookings).
They are treated as a SOFT constraint: a rate is calculated and compared
against DOUBLE_BOOKING_THRESHOLD_PCT to determine acceptability.
"""

# Soft-constraint threshold (mirrors csp_analyzer.py)
DOUBLE_BOOKING_THRESHOLD_PCT: float = 15.0  # % of room-slot assignments that may overlap
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from data_loader import TimetableDataLoader

def parse_weeks(weeks_str):
    """Parse weeks string into a set of week numbers."""
    weeks = set()
    if pd.isna(weeks_str):
        return {1}
    parts = str(weeks_str).split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                s, e = part.split('-')
                weeks.update(range(int(s), int(e) + 1))
            except:
                pass
        else:
            try:
                weeks.add(int(part))
            except:
                pass
    return weeks if weeks else {1}

def find_double_bookings(limit=10):
    """
    Find and display double-booking statistics.

    Reports the double-booking RATE (% of room-slot assignments that are
    shared) and compares it against DOUBLE_BOOKING_THRESHOLD_PCT.
    Treats double bookings as a soft constraint, not a hard failure.
    """
    print("Loading data...")
    loader = TimetableDataLoader()
    events = loader.events.copy()

    # Filter to events with room assignments (not online)
    physical = events[events['Room'].notna() & (events['Online Delivery'] != True)].copy()
    total_room_slots = len(physical)

    print(f"Physical room-assigned events: {total_room_slots:,}")

    # Build index: (Room, Day, Hour) -> list of event dicts
    from collections import defaultdict
    room_day_hour = defaultdict(list)

    for _, row in physical.iterrows():
        room = row['Room']
        day = row.get('Day')
        start = row.get('Start Hour')
        weeks = parse_weeks(row.get('Weeks'))

        if pd.notna(day) and pd.notna(start):
            key = (room, day, int(start))
            room_day_hour[key].append({
                'event_id': row['Event ID'],
                'module': row['Module Code'],
                'weeks': weeks,
                'duration': row.get('Duration (minutes)', 50)
            })

    # Find overlapping pairs
    conflicts = []
    for key, events_list in room_day_hour.items():
        if len(events_list) < 2:
            continue

        for i in range(len(events_list)):
            for j in range(i + 1, len(events_list)):
                e1, e2 = events_list[i], events_list[j]
                if not e1['weeks'].isdisjoint(e2['weeks']):
                    overlapping_weeks = e1['weeks'] & e2['weeks']
                    conflicts.append({
                        'room': key[0],
                        'day': key[1],
                        'hour': key[2],
                        'event1_id': e1['event_id'],
                        'event1_module': e1['module'],
                        'event1_weeks': sorted(e1['weeks']),
                        'event2_id': e2['event_id'],
                        'event2_module': e2['module'],
                        'event2_weeks': sorted(e2['weeks']),
                        'overlapping_weeks': sorted(overlapping_weeks)
                    })

    total_conflicts = len(conflicts)
    double_booking_rate = (
        (total_conflicts / total_room_slots * 100) if total_room_slots > 0 else 0.0
    )

    # --- Summary (soft-constraint framing) ---
    print(f"\n{'='*55}")
    print(f"  Double-Booking Summary (SOFT constraint)")
    print(f"{'='*55}")
    print(f"  Overlapping room-slot pairs : {total_conflicts:,}")
    print(f"  Total room-slot assignments : {total_room_slots:,}")
    print(f"  Double-booking rate         : {double_booking_rate:.2f}%")
    print(f"  Acceptable threshold        : {DOUBLE_BOOKING_THRESHOLD_PCT:.1f}%")

    if double_booking_rate <= DOUBLE_BOOKING_THRESHOLD_PCT:
        print(f"  Status : ✅ WITHIN threshold — treated as intentional sharing")
    else:
        print(f"  Status : ⚠️  ABOVE threshold — review recommended")
    print(f"{'='*55}\n")

    # --- Sample examples ---
    if conflicts:
        print(f"Sample double-bookings (first {limit} of {total_conflicts:,}):")
        for i, c in enumerate(conflicts[:limit]):
            print(f"\n--- Example {i+1} ---")
            print(f"  Room : {c['room']}")
            print(f"  Day  : {c['day']}, Hour: {c['hour']}:00")
            print(f"  Event 1: {c['event1_id']} ({c['event1_module']})  weeks={c['event1_weeks']}")
            print(f"  Event 2: {c['event2_id']} ({c['event2_module']})  weeks={c['event2_weeks']}")
            print(f"  Overlapping weeks: {c['overlapping_weeks']}")

if __name__ == "__main__":
    find_double_bookings(limit=10)
