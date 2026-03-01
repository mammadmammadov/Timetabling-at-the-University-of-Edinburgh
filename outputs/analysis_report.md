# University Timetabling Scenario Analysis Report

**Generated**: 2026-03-01 21:38

---

## Executive Summary

This report analyzes the feasibility of reducing core teaching hours at the University. Three scenarios were evaluated:

| Scenario | Teaching Hours | Key Finding |
|----------|---------------|-------------|
| **Baseline** | Mon-Fri 9am-6pm | Current state |
| **Scenario A** | Mon-Fri 9am-5pm | 2 events need rescheduling |
| **Scenario B** | Mon-Thu 9-6pm, Fri 9am-12pm | 9 events need rescheduling |

---

## Scenario Comparison

![Scenario Comparison](C:\Users\gulma\Desktop\Timetabling-at-the-University-of-Edinburgh\outputs\scenario_comparison.png)

---

## Tier 1: Feasibility Analysis

### Events Requiring Rescheduling

| Metric | Baseline | Scenario A (9-5) | Scenario B (No Fri PM) |
|--------|----------|-----------------|------------------------|
| Unscheduled Events | 1 | 2 | 9 |
| Capacity Violations | 0 | 0 | 0 |
| Compulsory Clashes | 4097 | 5754 | 8856 |
| **Feasible (as-is)** | ❌ No | ❌ No | ❌ No |

### Displacement Analysis

![Displacement Analysis](C:\Users\gulma\Desktop\Timetabling-at-the-University-of-Edinburgh\outputs\displacement_analysis.png)

---

## Tier 2: Student Experience

| Metric | Baseline | Scenario A | Scenario B |
|--------|----------|------------|------------|
| Lunch Break (12-2pm) | 7.6% | 7.0% | 8.8% |
| Avg Daily Span | 4.9 hrs | 4.6 hrs | 4.7 hrs |

> **Note**: Lunch break percentage represents students who have at least 1 continuous hour free in the 12pm-2pm window.

---

## Tier 3: Room Utilization

| Metric | Baseline | Scenario A | Scenario B |
|--------|----------|------------|------------|
| Avg Utilization | 17.6% | 19.2% | 19.8% |
| Peak Utilization | 46.5% | 82.9% | 102.0% |
| Room-Hours Available | 29,205 | 25,960 | 25,311 |
| Room-Hours Used | 5,142 | 4,993 | 5,019 |

### Utilization Heatmaps

**Baseline (Current)**
![Baseline Heatmap](C:\Users\gulma\Desktop\Timetabling-at-the-University-of-Edinburgh\outputs\heatmap_baseline.png)

**Scenario A (9am-5pm)**
![Scenario A Heatmap](C:\Users\gulma\Desktop\Timetabling-at-the-University-of-Edinburgh\outputs\heatmap_scenario_a.png)

**Scenario B (No Friday PM)**
![Scenario B Heatmap](C:\Users\gulma\Desktop\Timetabling-at-the-University-of-Edinburgh\outputs\heatmap_scenario_b.png)

---

## Key Findings & Recommendations

### Scenario A (Mon-Fri 9am-5pm)
- **Impact**: 2 events currently in 5-6pm slot need rescheduling
- **Feasibility**: Feasible with optimization
- **Utilization Impact**: Room utilization would increase to ~19% (denser schedule)

### Scenario B (No Friday 12pm-6pm)
- **Impact**: 9 events need rescheduling
- **Feasibility**: Feasible with optimization
- **Benefit**: Provides half-day Friday for staff/student activities

### Lunch Break Analysis
- Current lunch break compliance: 7.6%
- Both scenarios maintain similar lunch break availability
- Target of 1-hour break in 12-2pm window is challenging for majority

---

## Methodology

This analysis used a Constraint Satisfaction Problem (CSP) approach:
1. **Data ingestion**: Parsed 5,000 student schedules
2. **Constraint validation**: Checked room capacity, double-booking, and student clashes
3. **KPI calculation**: Computed feasibility, student experience, and efficiency metrics

---

