"""
Report Generator for University Timetabling Analysis.

Generates visualizations and a comprehensive markdown report comparing scenarios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import TimetableDataLoader
from kpi_calculator import KPICalculator


# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_utilization_heatmap(loader: TimetableDataLoader, scenario: str) -> str:
    """Create a heatmap of room utilization by day and hour."""
    events = loader.events
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    hours = list(range(9, 18))
    
    # Create utilization matrix
    util_matrix = np.zeros((len(days), len(hours)))
    
    for _, event in events.iterrows():
        day = event.get('Day')
        start_hour = event.get('Start Hour')
        
        if pd.notna(day) and pd.notna(start_hour) and day in days:
            day_idx = days.index(day)
            hour_idx = int(start_hour) - 9
            if 0 <= hour_idx < len(hours):
                util_matrix[day_idx, hour_idx] += 1
    
    # Normalize by max
    if util_matrix.max() > 0:
        util_matrix = util_matrix / util_matrix.max() * 100
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Mark excluded zones based on scenario
    if scenario == 'scenario_a':
        # 5pm-6pm excluded (hour 17 = index 8)
        for i in range(len(days)):
            util_matrix[i, 8] = -10  # Mark as excluded
    elif scenario == 'scenario_b':
        # Friday 12pm-6pm excluded
        friday_idx = 4
        for h in range(3, 9):  # hours 12-17
            util_matrix[friday_idx, h] = -10
    
    # Custom colormap with grey for excluded
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_under('lightgrey')
    
    im = ax.imshow(util_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f"{h}:00" for h in hours])
    ax.set_yticks(range(len(days)))
    ax.set_yticklabels(days)
    
    ax.set_xlabel('Hour')
    ax.set_ylabel('Day')
    ax.set_title(f'Teaching Density Heatmap - {scenario.replace("_", " ").title()}')
    
    plt.colorbar(im, ax=ax, label='Relative Utilization (%)')
    
    # Save
    filename = f"heatmap_{scenario}.png"
    filepath = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(filepath)


def create_comparison_bar_chart(all_kpis: List[Dict]) -> str:
    """Create bar charts comparing key metrics across scenarios."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scenarios = [k['scenario'] for k in all_kpis]
    x = np.arange(len(scenarios))
    width = 0.6
    
    # Unscheduled events
    ax1 = axes[0, 0]
    unscheduled = [k['feasibility']['unscheduled_events'] for k in all_kpis]
    bars1 = ax1.bar(x, unscheduled, width, color=['#2ecc71', '#e74c3c', '#e74c3c'])
    ax1.set_ylabel('Count')
    ax1.set_title('Events Requiring Rescheduling')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
    ax1.bar_label(bars1, padding=3)
    
    # Lunch break percentage
    ax2 = axes[0, 1]
    lunch = [k['student_experience']['lunch_break_percentage'] for k in all_kpis]
    bars2 = ax2.bar(x, lunch, width, color='#3498db')
    ax2.set_ylabel('Percentage')
    ax2.set_title('Students with Lunch Break (12-2pm)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
    ax2.set_ylim(0, 100)
    ax2.bar_label(bars2, fmt='%.1f%%', padding=3)
    
    # Room utilization
    ax3 = axes[1, 0]
    util = [k['efficiency']['avg_room_utilization'] for k in all_kpis]
    bars3 = ax3.bar(x, util, width, color='#9b59b6')
    ax3.set_ylabel('Percentage')
    ax3.set_title('Average Room Utilization')
    ax3.set_xticks(x)
    ax3.set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
    ax3.axhline(y=65, color='r', linestyle='--', label='Target (65%)')
    ax3.legend()
    ax3.bar_label(bars3, fmt='%.1f%%', padding=3)
    
    # Peak utilization
    ax4 = axes[1, 1]
    peak = [k['efficiency']['peak_utilization'] for k in all_kpis]
    bars4 = ax4.bar(x, peak, width, color='#e67e22')
    ax4.set_ylabel('Percentage')
    ax4.set_title('Peak Hourly Utilization')
    ax4.set_xticks(x)
    ax4.set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
    ax4.axhline(y=70, color='r', linestyle='--', label='Bottleneck threshold (70%)')
    ax4.legend()
    ax4.bar_label(bars4, fmt='%.1f%%', padding=3)
    
    plt.tight_layout()
    
    filename = "scenario_comparison.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(filepath)


def create_displacement_analysis_chart(loader: TimetableDataLoader) -> str:
    """Create chart showing events in cut zones."""
    events = loader.events
    
    # Count events in different time windows
    late_afternoon = len(events[(events['Start Hour'] >= 17) & (events['Start Hour'] < 18)])
    friday_afternoon = len(events[(events['Day'] == 'Friday') & (events['Start Hour'] >= 12)])
    
    # By event type
    event_types = events['Event Type'].value_counts().head(6)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cut zones
    ax1 = axes[0]
    zones = ['5-6pm (All Days)', 'Friday 12-6pm']
    counts = [late_afternoon, friday_afternoon]
    colors = ['#e74c3c', '#f39c12']
    bars = ax1.barh(zones, counts, color=colors)
    ax1.set_xlabel('Number of Events')
    ax1.set_title('Events in Proposed Cut Zones')
    ax1.bar_label(bars, padding=3)
    
    # Event types
    ax2 = axes[1]
    event_types.plot(kind='bar', ax=ax2, color='#3498db')
    ax2.set_ylabel('Count')
    ax2.set_title('Events by Type (Top 6)')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    filename = "displacement_analysis.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(filepath)


def generate_markdown_report(all_kpis: List[Dict], chart_paths: Dict[str, str]) -> str:
    """Generate the final markdown report."""
    
    baseline = next(k for k in all_kpis if k['scenario'] == 'baseline')
    scenario_a = next(k for k in all_kpis if k['scenario'] == 'scenario_a')
    scenario_b = next(k for k in all_kpis if k['scenario'] == 'scenario_b')
    
    report = f"""# University Timetabling Scenario Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Executive Summary

This report analyzes the feasibility of reducing core teaching hours at the University. Three scenarios were evaluated:

| Scenario | Teaching Hours | Key Finding |
|----------|---------------|-------------|
| **Baseline** | Mon-Fri 9am-6pm | Current state |
| **Scenario A** | Mon-Fri 9am-5pm | {scenario_a['feasibility']['unscheduled_events']:,} events need rescheduling |
| **Scenario B** | Mon-Thu 9-6pm, Fri 9am-12pm | {scenario_b['feasibility']['unscheduled_events']:,} events need rescheduling |

---

## Scenario Comparison

![Scenario Comparison]({chart_paths['comparison']})

---

## Tier 1: Feasibility Analysis

### Events Requiring Rescheduling

| Metric | Baseline | Scenario A (9-5) | Scenario B (No Fri PM) |
|--------|----------|-----------------|------------------------|
| Unscheduled Events | {baseline['feasibility']['unscheduled_events']:,} | {scenario_a['feasibility']['unscheduled_events']:,} | {scenario_b['feasibility']['unscheduled_events']:,} |
| Capacity Violations | {baseline['feasibility']['capacity_violations']} | {scenario_a['feasibility']['capacity_violations']} | {scenario_b['feasibility']['capacity_violations']} |
| Compulsory Clashes | {baseline['feasibility']['compulsory_clashes']} | {scenario_a['feasibility']['compulsory_clashes']} | {scenario_b['feasibility']['compulsory_clashes']} |
| **Feasible (as-is)** | {'✅ Yes' if baseline['feasibility']['is_feasible'] else '❌ No'} | {'✅ Yes' if scenario_a['feasibility']['is_feasible'] else '❌ No'} | {'✅ Yes' if scenario_b['feasibility']['is_feasible'] else '❌ No'} |

### Displacement Analysis

![Displacement Analysis]({chart_paths['displacement']})

---

## Tier 2: Student Experience

| Metric | Baseline | Scenario A | Scenario B |
|--------|----------|------------|------------|
| Lunch Break (12-2pm) | {baseline['student_experience']['lunch_break_percentage']}% | {scenario_a['student_experience']['lunch_break_percentage']}% | {scenario_b['student_experience']['lunch_break_percentage']}% |
| Avg Daily Span | {baseline['student_experience']['avg_daily_span_hours']:.1f} hrs | {scenario_a['student_experience']['avg_daily_span_hours']:.1f} hrs | {scenario_b['student_experience']['avg_daily_span_hours']:.1f} hrs |

> **Note**: Lunch break percentage represents students who have at least 1 continuous hour free in the 12pm-2pm window.

---

## Tier 3: Room Utilization

| Metric | Baseline | Scenario A | Scenario B |
|--------|----------|------------|------------|
| Avg Utilization | {baseline['efficiency']['avg_room_utilization']:.1f}% | {scenario_a['efficiency']['avg_room_utilization']:.1f}% | {scenario_b['efficiency']['avg_room_utilization']:.1f}% |
| Peak Utilization | {baseline['efficiency']['peak_utilization']:.1f}% | {scenario_a['efficiency']['peak_utilization']:.1f}% | {scenario_b['efficiency']['peak_utilization']:.1f}% |
| Room-Hours Available | {baseline['efficiency']['room_hours_available']:,} | {scenario_a['efficiency']['room_hours_available']:,} | {scenario_b['efficiency']['room_hours_available']:,} |
| Room-Hours Used | {baseline['efficiency']['room_hours_used']:,} | {scenario_a['efficiency']['room_hours_used']:,} | {scenario_b['efficiency']['room_hours_used']:,} |

### Utilization Heatmaps

**Baseline (Current)**
![Baseline Heatmap]({chart_paths['heatmap_baseline']})

**Scenario A (9am-5pm)**
![Scenario A Heatmap]({chart_paths['heatmap_scenario_a']})

**Scenario B (No Friday PM)**
![Scenario B Heatmap]({chart_paths['heatmap_scenario_b']})

---

## Key Findings & Recommendations

### Scenario A (Mon-Fri 9am-5pm)
- **Impact**: {scenario_a['feasibility']['unscheduled_events']:,} events currently in 5-6pm slot need rescheduling
- **Feasibility**: {'Feasible with optimization' if scenario_a['feasibility']['unscheduled_events'] < 1000 else 'Requires significant rescheduling effort'}
- **Utilization Impact**: Room utilization would increase to ~{scenario_a['efficiency']['avg_room_utilization']:.0f}% (denser schedule)

### Scenario B (No Friday 12pm-6pm)
- **Impact**: {scenario_b['feasibility']['unscheduled_events']:,} events need rescheduling
- **Feasibility**: {'Feasible with optimization' if scenario_b['feasibility']['unscheduled_events'] < 1500 else 'More challenging - larger displacement'}
- **Benefit**: Provides half-day Friday for staff/student activities

### Lunch Break Analysis
- Current lunch break compliance: {baseline['student_experience']['lunch_break_percentage']}%
- Both scenarios maintain similar lunch break availability
- Target of 1-hour break in 12-2pm window is {'achievable' if baseline['student_experience']['lunch_break_percentage'] > 70 else 'challenging'} for majority

---

## Methodology

This analysis used a Constraint Satisfaction Problem (CSP) approach:
1. **Data ingestion**: Parsed {baseline['student_experience']['students_analyzed']:,} student schedules
2. **Constraint validation**: Checked room capacity, double-booking, and student clashes
3. **KPI calculation**: Computed feasibility, student experience, and efficiency metrics

---

*Report generated by University Timetabling Analysis System*
"""
    
    return report


def main():
    """Run the full analysis and generate report."""
    print("=" * 60)
    print("University Timetabling Scenario Analysis")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading data...")
    loader = TimetableDataLoader()
    print(f"  - Events: {len(loader.events):,}")
    print(f"  - Rooms: {len(loader.rooms):,}")
    print(f"  - Student-Events: {len(loader.student_events):,}")
    
    # Calculate KPIs for each scenario
    print("\n[2/5] Calculating KPIs...")
    all_kpis = []
    for scenario in ['baseline', 'scenario_a', 'scenario_b']:
        print(f"  - {scenario}...")
        calc = KPICalculator(scenario)
        kpis = calc.calculate_all_kpis()
        all_kpis.append(kpis)
        print(f"    Unscheduled: {kpis['feasibility']['unscheduled_events']:,}")
        print(f"    Lunch break: {kpis['student_experience']['lunch_break_percentage']}%")
    
    # Generate visualizations
    print("\n[3/5] Generating visualizations...")
    chart_paths = {}
    
    chart_paths['comparison'] = create_comparison_bar_chart(all_kpis)
    print(f"  - Comparison chart: {chart_paths['comparison']}")
    
    chart_paths['displacement'] = create_displacement_analysis_chart(loader)
    print(f"  - Displacement chart: {chart_paths['displacement']}")
    
    for scenario in ['baseline', 'scenario_a', 'scenario_b']:
        key = f'heatmap_{scenario}'
        chart_paths[key] = create_utilization_heatmap(loader, scenario)
        print(f"  - {scenario} heatmap: {chart_paths[key]}")
    
    # Generate report
    print("\n[4/5] Generating markdown report...")
    report = generate_markdown_report(all_kpis, chart_paths)
    
    report_path = OUTPUT_DIR / "analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  - Report saved: {report_path}")
    
    # Summary
    print("\n[5/5] Analysis Complete!")
    print("=" * 60)
    print("\nKey Results:")
    for kpi in all_kpis:
        scenario = kpi['scenario']
        print(f"\n{scenario.upper()}:")
        print(f"  Feasible: {kpi['feasibility']['is_feasible']}")
        print(f"  Unscheduled: {kpi['feasibility']['unscheduled_events']:,}")
        print(f"  Lunch break: {kpi['student_experience']['lunch_break_percentage']}%")
        print(f"  Room util: {kpi['efficiency']['avg_room_utilization']:.1f}%")
    
    print(f"\n[OK] Full report available at: {report_path}")
    
    return all_kpis


if __name__ == "__main__":  
    main()
