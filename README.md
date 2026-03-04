# Timetabling at the University of Edinburgh

This repository contains the codebase for optimizing the timetabling at the University of Edinburgh using Simulated Annealing. 


## Getting Started

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/mammadmammadov/Timetabling-at-the-University-of-Edinburgh.git
cd Timetabling-at-the-University-of-Edinburgh
```

### 2. Install Dependencies
Make sure you have the required Python packages installed to handle the dataset processing:
```bash
pip install -r requirements.txt
```

### 3. Running the Optimizer
The core optimizer is located in `scripts/csp_analyzer.py`. Running this script will automatically ingest the raw datasets, apply hard and soft constraints, run the Simulated Annealing logic and generate new Excel spreadsheets with the optimized schedule.

To run it yourself:
```bash
python scripts/csp_analyzer.py
```

## Generated Datasets (Already Completed)
**Note:** You do not actually need to run the python command above if you just want to see the results.

The optimizer (implementing greedy rescheduling and simulated annealing, which are popular algorithms in the domain of AI) has *already* been run on the current datasets. The fully optimized timetable configurations are pre-generated and sitting in the `outputs/` folder:
- `outputs/timetable_baseline.xlsx` (The optimized Baseline scenario)
- `outputs/timetable_scenario_a.xlsx` (The optimized Scenario A scenario, this one is still in progress)

These files contain the exact revised Days, Start Hours, and Rooms assigned to every single event, as well as a "Violations" sheet detailing the events that were mathematically impossible to find a legal room for, therefore, needing manual intervention.
