# TODO
UPDATE / make this a bit cleaner

# Main command for paper data (From the repo root)
RUN_LABEL=paper_data RUNS=50 SAVE_TRAJECTORIES=false WRITE_SNAPSHOT=true NUM_THREADS=10 julia +1.10 --project=src src/main.jl

# Main command for playing around with it
JULIA_NUM_THREADS=10 julia +1.10 --project=src src/main.jl




# Multi-agent water-collection simulations (Julia)

This repository runs multi-agent water-collection simulations in a gridworld specified in PDDL. Agents act sequentially, plan using A* guided by a task-specific heuristic, and select actions via Boltzmann sampling.

## Repository structure

- `src/main.jl` — entry point (runs the simulation batch)
- `src/utils.jl` — small utilities (ENV parsing, thread-safe printing, reproducibility snapshot)
- `src/water_collection_heuristic.jl` — task-specific heuristic used by A*
- `src/domain.pddl` — PDDL domain definition
- `src/maps/` — PDDL problem instances (maps)
- `data/` — outputs written by runs (created automatically)

## Requirements

- Julia **1.10.x** (recommended)
- Packages are managed via the Julia project in `src/` (`src/Project.toml` + `src/Manifest.toml`)

If you use `juliaup`, you can run with `julia +1.10 ...`.

## Setup

From the repository root:

```bash
julia +1.10 --project=src -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'
p`, you can run with `julia +1.10 ...`.

p`, you can run with `julia +1.10 ...`.
p`, you can run with `julia +1.10 ...`.
## RUNn

##

## Setup



# Main command
julia +1.10 --project=src -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'
RUN_LABEL=Test_FINAL JULIA_NUM_THREADS=10 julia +1.10 --project=src src/main.jl