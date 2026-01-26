# Todo now (to get clean repo for cogsci)
- Delete useless files: some under src/maps, some others at root (.md) files, and a couple of notes.txt here & there
- Double check
- Run final paper_run
- Write clean README.md





### Below: very old notes

# TODO / Roadmap for line-cutting project

_Last updated: 2025-11-21_

## 0. Repo & structure

- [ ] Move simulation outputs from `maps/data/` into `data/raw_sim/` and update paths in code.
- [ ] Move state histories into `data/state_histories/` and ignore them in `.gitignore`.
- [ ] Create `src/` for core Julia code and `analysis/` for notebooks & Python helpers.
- [ ] Add a small `data/examples/` folder with 1–2 maps × conditions so notebooks can run without full sims.

## 1. Analysis pipeline

- [ ] Turn the Jupyter helper functions into a module, e.g. `analysis/python/sim_analysis.py`:
  - [ ] `parse_dict_string`
  - [ ] `load_sim_file` (Julia CSV → runs × agents DataFrame)
  - [ ] `summarize_runs`
  - [ ] `plot_single_file`, `compare_files`
- [ ] Update existing notebook(s) to import from `sim_analysis.py` instead of defining everything inline.
- [ ] Create a Google Sheet summarizing per-map behavior (L0 vs L1 vs paper; later with-line).

### Notes: Compare 

## 2. Baseline with line

- [ ] Export with-line baseline from Logan’s Google Sheet for 2–3 canonical maps.
- [ ] Write `load_withline_csv` to return runs × agents like `load_sim_file`.
- [ ] Extend comparisons to include `with_line` alongside `level_0`, `level_1`, `paper_sim`.

## 3. Simulation improvements

- [ ] Before running again, decide where to save the data (change from maps/data) and what to save (state_histories seem useless)
- [ ] Add random seed(s) in Julia (`Random.seed!`) so runs are reproducible.
- [ ] Standardize output format (map_name, level, noise, run, agent, filled_time, collisions?).
- [ ] Decide which extra info to keep (collisions, full trajectories) and which to drop.

## 4. Level-k reasoning

- [ ] Cleanly parameterize reasoning level (0/1/k) in main sim:
  - [ ] `modify_state_level_0`
  - [ ] `modify_state_level_1`
  - [ ] general `modify_state_level_k` if needed.
- [ ] Sanity-check behavior on 2–3 key maps for k = 0,1,2.

## 5. Longer-term / parked

- [ ] Map classification (funnel / trivial / maybe).
- [ ] Connection to rigid rules / plausible deniability ideas.
- [ ] Decide on codebase direction (polish Julia vs re-implement in Memo).
