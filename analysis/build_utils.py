"""
Tools to build per-map analysis CSVs and the project-wide summary CSV.

Intended usage:
    - call `get_summary_dataframe(recompute=True)` once to regenerate all
      analysis and summary files from raw data;
    - call `get_summary_dataframe(recompute=False)` to load the existing
      summary CSV in notebooks.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------

def gini(values):
    """
    Compute Gini coefficient for a 1D array of non-negative values.
    """
    x = np.asarray(values, dtype=float)
    if np.all(x == 0):
        return np.nan
    
    x_sorted = np.sort(x)
    n = len(x_sorted)
    # 2 * sum(i * x_i) / (n * sum(x_i)) - (n + 1)/n
    g = (2.0 * np.sum(np.arange(1, n + 1) * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n
    return g



# Metrics summarized (mean, sd) in summary csv
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

# ---------------------------------------------------------------------
# Per-map analysis files
# ---------------------------------------------------------------------

def build_analysis_one_map(
        map_name, 
        run_type, 
        data_root="../data/readable-data", 
        xp_csv="../data/2023_experimental_data.csv", 
        output_root="../data/analysis_per_map"
        ):
    """
    Build an analysis CSV for a single map and run type.

    The output file includes:
    - all simulation runs for that map and run type
    - the corresponding experimental row for that map
    - derived metrics (e.g., aggregate_welfare, inequality, etc.)

    Saves to: <output_root>/<run_type>/<map_name>.csv
    """
    data_root = Path(data_root)
    xp_csv = Path(xp_csv)
    output_root = Path(output_root)

    # ---------- 1) Make dataframe with simulation and experimental data ----------
    df_xp = pd.read_csv(xp_csv) # experimental data

    sim_csv = data_root / run_type / f"{map_name}.csv"
    if not sim_csv.exists():
        raise FileNotFoundError(f"Simulation file not found: {sim_csv}")
    df_sim = pd.read_csv(sim_csv) # simulation data

    # Add columns to simulation data frame, re-order
    df_sim["source"] = "simulation"
    df_sim["condition"] = np.nan
    base_cols = ["source", "run", "condition"]
    other_cols = [c for c in df_sim.columns if c not in base_cols]
    df_sim = df_sim[base_cols + other_cols]

    # Extract the relevant experimental rows for this map, and make df compatible
    df_xp_subset = df_xp[df_xp["map"] == map_name].copy()
    if df_xp_subset.empty:
        raise ValueError(f"No experimental rows found for map '{map_name}' in {xp_csv}")
    df_xp_subset["source"] = "experimental"
    df_xp_subset["run"] = np.nan
    df_xp_subset["noise"] = np.nan
    df_xp_subset = df_xp_subset[base_cols + other_cols]

    # Combine simulation and experimental dfs to get analysis data frame
    df = pd.concat([df_sim, df_xp_subset], ignore_index=True)
    df["run"] = df["run"].astype("Int64") # restore run index as integer

    # ---------- 2) Compute metrics ----------
    agent_cols = [c for c in df.columns if c.startswith("agent_")]

    # Basic metrics
    df["first"] = df[agent_cols].min(axis=1)
    df["last"]  = df[agent_cols].max(axis=1)
    df["gap"] = (df["last"] - df["first"]) * 1.0
    df["average"] = df[agent_cols].mean(axis=1)
    df["total"]  = df[agent_cols].sum(axis=1)

    # Use 'line' condition metrics as baseline for other metrics
    baseline_mask = df["condition"] == "line"
    baseline_welfare = df.loc[ baseline_mask, "total"].mean()
    baseline_last = df.loc[ baseline_mask, "last"].mean() # For normalization

    # Change in aggregate welfare, compared to baseline
    df["aggregate_welfare"] = ( baseline_welfare - df["total"] ) / baseline_last
    df.loc[baseline_mask, "aggregate_welfare"] = 0.0 # make sure baseline welfare stays at 0

    # Normalized gap, compared to baseline
    baseline_gap = df.loc[baseline_mask, "gap"].mean()
    df["gap_normalized"] = (df["gap"] - baseline_gap) / baseline_last

    # Gini coefficient (not compared to baseline)
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
    Build per-map analysis CSVs for all run types found under data_root,
    treating each of its subdirectories as a run type.
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


# ---------------------------------------------------------------------
# Summary CSV (across maps & run types)
# ---------------------------------------------------------------------

def build_summary_csv(
    analysis_data_root="../data/analysis_per_map",
    output_csv="../data/summary_statistics.csv"
):
    """
    Build a summary CSV aggregating metrics across all maps and run types.
    """
    analysis_data_root = Path(analysis_data_root)
    output_csv = Path(output_csv)

    run_types = [
        p.name
        for p in analysis_data_root.iterdir()
        if p.is_dir() and any(p.glob("*.csv"))
    ]
    if not run_types:
        raise RuntimeError(f"No run_type subfolders with CSVs found under {analysis_data_root}")

    map_names = set()
    for run_type in run_types:
        run_dir = analysis_data_root / run_type
        for f in run_dir.glob("*.csv"):
            map_names.add(f.stem)
    map_names = sorted(map_names)

    all_rows = []

    for map_name in map_names:
        # Experimental-based metrics ('line', '1cut*')
        ref_df = None
        for run_type in run_types:
            path = analysis_data_root / run_type / f"{map_name}.csv"
            if path.exists():
                ref_df = pd.read_csv(path)
                break

        if ref_df is None:
            print(f"[WARN] No analysis file found for map={map_name} in any run_type; skipping exp-based rows.")
        else:
            mask_line = (ref_df["source"] == "experimental") & (ref_df["condition"] == "line")
            if mask_line.any():
                line_mean = ref_df.loc[mask_line, METRIC_COLS].mean()
                line_sd   = ref_df.loc[mask_line, METRIC_COLS].std(ddof=1)

                row = {"map": map_name, "source": "line"}
                for m in METRIC_COLS:
                    row[m] = line_mean[m]
                    row[m + "_sd"] = line_sd[m]
                all_rows.append(row)
            else:
                print(f"[WARN] No 'line' condition found for map={map_name} in {path}")

            mask_cut = (ref_df["source"] == "experimental") & ref_df["condition"].astype(str).str.startswith("1cut")
            if mask_cut.any():
                for cond, df_cond in ref_df.loc[mask_cut].groupby("condition"):
                    cut_mean = df_cond[METRIC_COLS].mean()
                    cut_sd   = df_cond[METRIC_COLS].std(ddof=1)

                    row = {"map": map_name, "source": cond}
                    for m in METRIC_COLS:
                        row[m] = cut_mean[m]
                        row[m + "_sd"] = cut_sd[m]
                    all_rows.append(row)

        # Simulation-based metrics (e.g., paper_data, level-0, level-1)
        for run_type in run_types:
            path = analysis_data_root / run_type / f"{map_name}.csv"
            if not path.exists():
                continue

            df_rt = pd.read_csv(path)
            mask_sim = df_rt["source"] == "simulation"
            if not mask_sim.any():
                print(f"[WARN] No simulation rows for map={map_name}, run_type={run_type}")
                continue

            sim_mean = df_rt.loc[mask_sim, METRIC_COLS].mean()
            sim_sd   = df_rt.loc[mask_sim, METRIC_COLS].std(ddof=1)
            n_sim    = mask_sim.sum()

            row = {"map": map_name, "source": run_type}
            row["n_runs"] = int(n_sim)
            for m in METRIC_COLS:
                row[m] = sim_mean[m]
                row[m + "_sd"] = sim_sd[m]
            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No summary rows created; check your analysis_data_root structure.")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved summary CSV: {output_csv}")

    return df

# ---------------------------------------------------------------------
# Convenience helper for notebooks
# ---------------------------------------------------------------------

def get_summary_dataframe(
    recompute=False,
    data_root="../data/readable-data",
    xp_csv="../data/2023_experimental_data.csv",
    analysis_data_root="../data/analysis_per_map",
    summary_csv="../data/summary_statistics.csv",
):
    """
    High-level helper for notebooks.

    If recompute is True:
      - Rebuild per-map analysis CSVs from raw simulation + experimental data.
      - Rebuild the summary CSV.
      - Return the fresh summary DataFrame.

    If recompute is False:
      - Read the existing summary CSV and return it.
    """
    if recompute:
        build_all_analyses(
            data_root=data_root,
            xp_csv=xp_csv,
            output_root=analysis_data_root,
        )
        df = build_summary_csv(
            analysis_data_root=analysis_data_root,
            output_csv=summary_csv,
        )
        return df

    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        raise FileNotFoundError(
            f"Summary CSV {summary_csv} not found. "
            f"Either run get_summary_dataframe(recompute=True) "
            f"or ensure the file exists."
        )
    return pd.read_csv(summary_csv)
