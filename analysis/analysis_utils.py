import numpy as np
import pandas as pd
from pathlib import Path

def gini(values):
    """
    Compute Gini coefficient for a 1D array-like of non-negative values.
    Returns np.nan if all values are zero.
    """
    x = np.asarray(values, dtype=float)
    if np.all(x == 0):
        return np.nan
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cum = np.cumsum(x_sorted)
    # 2 * sum(i * x_i) / (n * sum(x)) - (n + 1)/n
    g = (2.0 * np.sum((np.arange(1, n + 1)) * x_sorted) / (n * np.sum(x_sorted))) - (n + 1) / n
    return g

### ---------- Functions to make analysis file for each map and run type ---------- ###
def build_analysis_one_map(
        map_name, 
        run_type, 
        data_root="../data/readable-data", 
        xp_csv="../data/2023_experimental_data.csv", 
        output_root="../data/analysis_per_map"
        ):
    """
    For a given map (e.g., 'no_line_1') and run type (e.g., 'level-0', 'level-1', 'paper_data'),
    build an analysis CSV that includes:
    - all simulation runs for that map & run type
    - the 'line' and '1cut" experimental data from xp_csv
    
    and computes useful metrics, including change in aggregate welfare compared to line scenario, gini, etc.
    Saves to: <output_root>/<run_type>/<map_name>.csv
    """
    data_root = Path(data_root)
    xp_csv = Path(xp_csv)
    output_root = Path(output_root)

    # ---------- 1) Make dataframe with simulation and experimental data ----------
    df_xp = pd.read_csv(xp_csv) # Load experimental data

    sim_csv = data_root / run_type / f"{map_name}.csv"
    if not sim_csv.exists():
        raise FileNotFoundError(f"Simulation file not found: {sim_csv}")
    df_sim = pd.read_csv(sim_csv) # Load simulation data

    # Add columns to simulation data frame, re-order
    df_sim["source"] = "simulation"
    df_sim["condition"] = np.nan
    base_cols = ["source", "run", "condition"]
    other_cols = [c for c in df_sim.columns if c not in base_cols]
    df_sim = df_sim[base_cols + other_cols]

    # Extract the 2-to-4 relevant lines from experimental data, and make df compatible
    df_xp_subset = df_xp[df_xp["map"] == map_name].copy()
    if df_xp_subset.empty:
        raise ValueError(f"No experimental rows found for map '{map_name}' in {xp_csv}")
    df_xp_subset["source"] = "experimental"
    df_xp_subset["run"] = np.nan
    df_xp_subset["noise"] = np.nan
    df_xp_subset = df_xp_subset[base_cols + other_cols]

    # Combine simulation and experimental dfs to get analysis data frame
    df = pd.concat([df_sim, df_xp_subset], ignore_index=True)
    df["run"] = df["run"].astype("Int64") # Restore run index as integer

    # ---------- 2) Compute metrics ----------
    agent_cols = [c for c in df.columns if c.startswith("agent_")] # Identify the agent columns

    # Basic metrics
    df["first"] = df[agent_cols].min(axis=1)
    df["last"]  = df[agent_cols].max(axis=1)
    df["gap"] = (df["last"] - df["first"]) * 1.0
    df["average"] = df[agent_cols].mean(axis=1)
    df["total"]  = df[agent_cols].sum(axis=1)

    # Baseline metrics: those for 'line' condition
    baseline_mask = df["condition"] == "line"
    baseline_welfare = df.loc[ baseline_mask, "total"].mean()
    baseline_last = df.loc[ baseline_mask, "last"].mean() # For normalization

    # Change in aggregate welfare, compared to baseline
    df["aggregate_welfare"] = ( baseline_welfare - df["total"] ) / baseline_last
    df.loc[baseline_mask, "aggregate_welfare"] = 0.0 # make sure baseline welfare stays at 0

    # Normalized gap
    baseline_gap = df.loc[baseline_mask, "gap"].mean()
    df["gap_normalized"] = (df["gap"] - baseline_gap) / baseline_last

    # Gini inequality
    df["gini"] = df[agent_cols].apply(gini, axis=1)

    # ---------- 3) Save to analysis folder ----------
    output_dir = output_root / run_type
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{map_name}.csv"
    df.to_csv(output_path, index=False)

    return f"Result for {map_name} in {run_type} run saved to {output_dir}"

def build_all_analyses(
        data_root="../data/readable-data",
        xp_csv="../data/2023_experimental_data.csv",
        output_root="../data/analysis_per_map"
        ):
    """
    Build analysis CSVs for all maps and run types inside data_root folder.
    Uses build_analysis_one_map().
    """
    data_root = Path(data_root)
    xp_csv = Path(xp_csv)
    output_root = Path(output_root)

    # Deduce run types from subfolders of data_root
    run_types = []
    for p in data_root.iterdir():
        if p.is_dir() and any(p.glob("*.csv")):
            run_types.append(p.name)

    for run_type in run_types:
        sim_dir = data_root / run_type
        map_names = sorted(f.stem for f in sim_dir.glob("*.csv"))

        print(f"\n=== Run type: {run_type} ===")
        for map_name in map_names:
            try:
                build_analysis_one_map(
                    map_name=map_name,
                    run_type=run_type,
                    data_root=data_root,
                    xp_csv=xp_csv,
                    output_root=output_root,
                )
                print(f"  ✔ {map_name}")
            except Exception as e:
                print(f"  ✖ {map_name}: {type(e).__name__}: {e}")
    return

### ---------- Functions to make one big analysis file with summary statistics ---------- ###

# Metrics to summarize
METRIC_COLS = [
    "first",
    "last",
    "gap",
    "average",
    "total",
    "aggregate_welfare",
    "gap_normalized",
    "gini",
]

def build_summary_csv(
    analysis_data_root="../data/analysis_per_map",
    output_csv="../data/summary_statistics.csv"
    ):
    """
    Build one big summary CSV.
    Each line is a map for a specific source of data:
        - 'line'               (baseline experimental line condition)
        - 'paper_data'         (mean over sim runs from paper_data, Joe's 2023 run)
        - 'level-0'            (mean over sim runs from my level-0 run)
        - 'level-1'            (mean over sim runs from my level-1 run) #(TODO modify this)
        - '1cut', '1cut_bad', '1cut_badder', ... (experimental cut variants)

    Columns include map, source, and all metrics under METRIC_COLS.
    It reads from per-map analysis files:
        <analysis_data_root>/<run_type>/<map>.csv

    and writes a single long-format CSV.
    """
    analysis_data_root = Path(analysis_data_root)
    output_csv = Path(output_csv)

    # 1) Detect run_types as subfolders with csv files
    run_types = [
        p.name
        for p in analysis_data_root.iterdir()
        if p.is_dir() and any(p.glob("*.csv"))
    ]
    if not run_types:
        raise RuntimeError(f"No run_type subfolders with CSVs found under {analysis_data_root}")

    # 2) Collect all map names (union across run_types)
    map_names = set()
    for run_type in run_types:
        run_dir = analysis_data_root / run_type
        for f in run_dir.glob("*.csv"):
            map_names.add(f.stem)
    map_names = sorted(map_names)

    all_rows = []

    for map_name in map_names:
        # ---- Experimental-based summaries ('line', '1cut*') ----
        # Use first run_type that has this map
        ref_df = None
        for run_type in run_types:
            path = analysis_data_root / run_type / f"{map_name}.csv"
            if path.exists():
                ref_df = pd.read_csv(path)
                break

        if ref_df is None:
            print(f"[WARN] No analysis file found for map={map_name} in any run_type; skipping exp-based rows.")
        else:
            # Baseline line
            mask_line = (ref_df["source"] == "experimental") & (ref_df["condition"] == "line")
            if mask_line.any():
                line_metrics = ref_df.loc[mask_line, METRIC_COLS].mean()
                row = {"map": map_name, "source": "line"}
                row.update(line_metrics.to_dict())
                all_rows.append(row)
            else:
                print(f"[WARN] No 'line' condition found for map={map_name} in {path}")

            # 1cut variants
            mask_cut = (ref_df["source"] == "experimental") & ref_df["condition"].astype(str).str.startswith("1cut")
            if mask_cut.any():
                for cond, df_cond in ref_df.loc[mask_cut].groupby("condition"):
                    cut_metrics = df_cond[METRIC_COLS].mean()
                    row = {"map": map_name, "source": cond}   # e.g. '1cut', '1cut_bad'
                    row.update(cut_metrics.to_dict())
                    all_rows.append(row)

        # ---- Simulation-based summaries (e.g., paper_data / level-0 / level-1) ----
        for run_type in run_types:
            path = analysis_data_root / run_type / f"{map_name}.csv"
            if not path.exists():
                # Totally fine: some maps may be missing for some run_types
                continue

            df_rt = pd.read_csv(path)
            mask_sim = df_rt["source"] == "simulation"
            if not mask_sim.any():
                print(f"[WARN] No simulation rows for map={map_name}, run_type={run_type}")
                continue

            sim_metrics = df_rt.loc[mask_sim, METRIC_COLS].mean()
            row = {"map": map_name, "source": run_type}  # e.g. 'paper_data', 'level-0', 'level-1'
            row.update(sim_metrics.to_dict())
            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No summary rows created; check your analysis_data_root structure.")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved summary CSV: {output_csv}")

    return df
