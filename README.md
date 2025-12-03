# Sugarscape Agent-Based Model

A Python simulation of Epstein & Axtell's Sugarscape model showing how simple agent behaviors create complex social patterns like wealth inequality.

**Student:** Nikita Devi Gurumayum  
**Student ID:** SCSJ2400109

## Quick Run

```bash
# 1. Clone & enter
git clone https://github.com/yourusername/sugarscape.git
cd sugarscape

# 2. Install requirements
pip install numpy pandas matplotlib

# 3. Run everything
python sugarscape_simulation.py
python sugarscape_analysis.py
```

## What You'll See

**Terminal Output:**
```
SUGARSCAPE SIMULATION
================================

Config 1: 100 agents, no reproduction
Step 100/500 - Population: 0 (DIED OUT)

Config 2: 100 agents with reproduction  
Step 500/500 - Population: 510
Wealth Inequality: Gini = 0.264

Config 3: 50 agents with reproduction
Step 500/500 - Population: 474
Wealth Inequality: Gini = 0.249
```

**What It Shows:**
- No reproduction = everyone dies (resource exhaustion)
- Reproduction = sustainable population (~500 agents)
- Emergent wealth inequality (Gini coefficient ~0.25)
- Graphs saved as PNG files automatically

## Files You Get
- `sugarscape_model.py` - Core simulation engine
- `sugarscape_simulation.py` - Runs 3 scenarios
- `sugarscape_analysis.py` - Creates graphs
- `simulation_results/` - CSV data + PNG graphs

## One Command To Run All
```bash
python sugarscape_simulation.py && python sugarscape_analysis.py
```

Done! You'll get all data and graphs in the `simulation_results_*/` folder.
