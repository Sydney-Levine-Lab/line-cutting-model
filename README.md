# Line-cutting simulation model

This repository contains the code used to produce universalization metrics for the project ***"How universal is universalization? Exploring the use of universalization in norm violation across the globe".***

The Julia code in `src/` runs simulations over different maps. Agents must collect water in a world where line-cutting has become generalized and everyone pushes forward selfishly.

The `data/` folder holds scenario, experimental, and simulation CSVs; `analysis/` contains tools to build metrics and compare them to participant judgments.

---

## Repository layout

From the repo root:

- `src/` – **Julia simulation code**
  - `main.jl` – entry point to run simulations.
  - `utils.jl` – small utilities
  - `water_collection_heuristic.jl` – task-specific heuristic used for planning
  - `domain.pddl` – PDDL domain describing the gridworld environment where agents must collect water
  - `maps/` – PDDL files for each individual map
  - `Project.toml`, `Manifest.toml` – Julia environment for the simulation.

- `data/` – **CSV data**
  - `data/scenarios/` – CSVs comparing outcomes in two videos shown to participants (after an agent cuts out of line vs. a line-following baseline).
  - `data/experimental/` – CSVs with participant judgments (evaluations of single line-cutting instance)
  - `data/simulations/` – CSVs comparing outcomes in simulations to the line-following baseline.
- `analysis/` – **Analysis tools**
  - `build_utils.py` – tools to build CSVs with outcome and universalized metrics
  - `analysis_utils.py` – tools for statistical analysis
  - `fit_simulation_to_data.ipynb` – notebook that fits the simulation outputs to participant judgments.

---

## Requirements

- **Julia** ≥ 1.10  
  The examples below assume you can invoke it as `julia +1.10` and that you are in the repository root.

---

## Quick start (simulation only)

From the repository root:

1. **Instantiate and precompile the Julia environment** (one-time):

   ```bash
   julia +1.10 --project=src -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'


2. **Minimal test run**:
   ```bash
    julia +1.10 --project=src src/main.jl

3. **Paper-style run** (to reproduce our results):
   ```bash
    RUN_LABEL=my_replication_run \
    RUNS=50 \
    SAVE_TRAJECTORIES=true \
    WRITE_SNAPSHOT=true \
    JULIA_NUM_THREADS=10 \
    julia +1.10 --project=src src/main.jl

4. **Analysis**
    You can use `analysis/fit_simulation_to_data.ipynb` to fit data from any run to participant judgments. You'll need Python 3 and a Jupyter-capable environment.