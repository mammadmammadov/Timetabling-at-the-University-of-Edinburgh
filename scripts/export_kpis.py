import sys
from pathlib import Path

sys.path.insert(0, 'scripts')
from kpi_calculator import KPICalculator

output_path = Path('outputs/kpi_results.txt')
output_path.parent.mkdir(exist_ok=True)

with open(output_path, 'w') as f:
    for scenario in ['raw', 'baseline', 'scenario_a']:
        f.write(f"\n{'='*60}\n")
        f.write(f"Scenario: {scenario.upper()}\n")
        f.write(f"{'='*60}\n")
        
        calc = KPICalculator(scenario)
        kpis = calc.calculate_all_kpis()
        
        f.write("\nTier 1 - Feasibility:\n")
        f.write(f"  Compulsory clashes: {kpis['feasibility']['compulsory_clashes']}\n")
        f.write(f"  Capacity violations: {kpis['feasibility']['capacity_violations']}\n")
        f.write(f"  Unscheduled events: {kpis['feasibility']['unscheduled_events']}\n")
        f.write(f"  Is Feasible: {kpis['feasibility']['is_feasible']}\n")
        
        f.write("\nTier 2 - Student Experience:\n")
        f.write(f"  Lunch break (12-2pm): {kpis['student_experience']['lunch_break_percentage']}%\n")
        f.write(f"  Avg daily span: {kpis['student_experience']['avg_daily_span_hours']} hours\n")
        
        f.write("\nTier 3 - Efficiency:\n")
        f.write(f"  Avg room utilization: {kpis['efficiency']['avg_room_utilization']}%\n")
        f.write(f"  Peak utilization: {kpis['efficiency']['peak_utilization']}%\n")
        f.write(f"  Room hours used: {kpis['efficiency']['room_hours_used']}/{kpis['efficiency']['room_hours_available']}\n")
        
print(f"Successfully wrote all KPIs to {output_path}")
