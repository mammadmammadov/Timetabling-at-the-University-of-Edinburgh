import pandas as pd
from data_loader import TimetableDataLoader
from csp_analyzer import TimetableCSP, TimeSlot

loader = TimetableDataLoader()
csp = TimetableCSP(loader)
csp.scenario = 'scenario_b'
csp._init_available_slots()
csp._init_rooms()
csp._init_events()

print("\nChecking the allowed double-bookings logic for the clustered events:")

event1 = csp.events['E:WRY9EJUZHI']
event2 = csp.events['E:97YJ11EZPP']

print(f"Event 1: {event1.event_id} (Weeks: {event1.weeks})")
print(f"Event 2: {event2.event_id} (Weeks: {event2.weeks})")

print(f"\nAre they allowed to overlap in the csp.allowed_double_bookings map?")
is_allowed = event2.event_id in csp.allowed_double_bookings.get(event1.event_id, set())
print(f"Allowed: {is_allowed}")

print("\nLet's test _is_slot_room_available for Event 2 when Event 1 is already in the room:")
slot = TimeSlot('Monday', 9.0, 10.0)

# Manually assign Event 1
csp._assign_event(event1, slot, '0225_02_2.201')

# Now test if Event 2 is allowed in
result = csp._is_slot_room_available(event2, slot, '0225_02_2.201')
print(f"Is Event 2 allowed to be placed? {result}")
