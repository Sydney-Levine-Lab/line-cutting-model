"""
Tools to build analysis CSVs for simulations and scenarios.

Intended usage from Notebook:
    - call get_universalization_summary(run_label, recompute=True) once for every type of simulation run,
    to build per-map universalization metrics CSV and load it.
    - call get_outcome_metrics(recompute=True) once to build per-scenario outcome metrics CSV and load it.
    - later, call these functions with recompute=False to load the dataframes without rebuilding them.
    - NEW (DEC 11): use build_display_matrix() to build regression ready display matrix using dfs for universalization metrics, outcome, and experimental data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_gini(values):
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



def compute_metrics(world, line):
    """
    Compute metrics by comparing completion times in world to those in the line baseline.

    line: size 8 array with completion times when agents follow a line.
    world: size 8 array with completion times in an alternate world.

    Returns a dict with outcome measures when world is a scenario where 1 agent cuts the line (1cut, 1cut_bad, 1cut_badder),
    and a univerlizability estimates when world is a simulation run.
    """
    line = pd.Series(line, dtype=float)
    world = pd.Series(world, dtype=float)

    line_last = line.max() # Completion time for last agent in line; used to normalize metrics
    
    # Aggregate welfare (positive: total completion times are lower than in line; ie welfare is improved)
    aggregate_welfare = - ( world.sum() - line.sum() ) / line_last
        
    # Inequality (positive: increase in gap btw first and last)
    gap_world = world.max() - world.min()
    gap_line = line_last - line.min()
    inequality = ( gap_world - gap_line ) / line_last

    # Cardinal harm (always positive; normalized sum of increases in completion time)
    positive_differences = (world - line).clip(lower=0)
    cardinal_harm =  positive_differences.sum() / line_last
    
    # Ordinal harm (always positive; sum of increases in rank)
    ### NOTE: this is a "blind" version. Possible TODO: replace with proper version.
    ### With just completion times, I can't tell if a line-cutter is using a different source or not. Kwon2023 reports a non-blind version, probably done by hand
    world_rank = world.rank(method="first")
    line_rank = line.rank(method="first")
    positive_rank_differences = (world_rank - line_rank).clip(lower=0)
    ordinal_harm_blind = positive_rank_differences.sum()

    # Gini (difference in Gini coefficients)
    gini = compute_gini(world) - compute_gini(line)
    
    return {
        "aggregate_welfare": aggregate_welfare,
        "ordinal_harm_blind": ordinal_harm_blind,
        "cardinal_harm": cardinal_harm,
        "inequality": inequality,
        "gini": gini
    }



METRIC_COLS = [
    "aggregate_welfare",
    "inequality",
    "cardinal_harm",
    "ordinal_harm_blind",
    "gini",
]

# ---------------------------------------------------------------------
# Build analysis CSVs
# ---------------------------------------------------------------------

def build_processed_sim_one_map(
        run_label,
        map_name, 
        sim_root="../data/simulations", 
        line_csv="../data/scenarios/completion_times.csv"
        ):
    """
    Build a processed simulation file, for a run type and map name with
    per-run univerlization metrics,
    by adding colums to the corresponding raw simulation file.

    Assumes run type is a subdirectory of sim_data_root, and that raw data is under raw/.
    Saves to <sim_data_root>/<run_label>/processed/<map_name>.csv
    """
    sim_root = Path(sim_root)
    raw_path = sim_root / run_label / "raw" / f"{map_name}.csv"
    processed_path = sim_root / run_label / "processed" / f"{map_name}.csv"

    df = pd.read_csv(raw_path) # Load raw simulation data
    agent_cols = [c for c in df.columns if c.startswith("agent_")] # Should be valid for both dfs

    # Load completion times when agents follow the line
    line_path = Path(line_csv)
    comp = pd.read_csv(line_path)
    line = comp[(comp["map"] == map_name) & (comp["condition"] == "line")]
    line_times = line.iloc[0][agent_cols].to_numpy(dtype=float)

    for idx, run in df.iterrows():
        world_times = run[agent_cols].to_numpy(dtype=float)
        metrics = compute_metrics(world=world_times, line=line_times)
        for k, v in metrics.items():
            df.loc[idx, k] = v

    df.to_csv(processed_path, index=False)



def build_processed_sim_all_maps(
    run_label,
    sim_root="../data/simulations",
    line_csv="../data/scenarios/completion_times.csv",
):
    """
    Build processed simulation files for every map within a run type folder.

    Uses build_processed_sim_one_map().
    """
    sim_root = Path(sim_root)
    raw_dir = sim_root / run_label / "raw"
    processed_dir = sim_root / run_label / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("*.csv"))
    for raw_path in raw_files:
        map_name = raw_path.stem
        build_processed_sim_one_map(run_label=run_label, map_name=map_name, sim_root=sim_root,line_csv=line_csv)

    print(
        f"Processed {len(raw_files)} maps for run_label='{run_label}' "
        f"and saved to {processed_dir}"
    )



def build_universalization_summary(
    run_label,
    sim_root="../data/simulations",
    output_csv="summary_universalization_metrics.csv"
):
    """
    Build a summary CSV aggregating universalization metrics across all maps
    for a given run type.
    """
    sim_root = Path(sim_root)
    processed_dir = sim_root / run_label / "processed"
    output_path = sim_root / run_label / output_csv

    processed_files = sorted(processed_dir.glob("*.csv"))

    rows = []

    for fpath in processed_files:
        map_name = fpath.stem
        df = pd.read_csv(fpath)

        n_runs = len(df)

        row = {
            "map_name": map_name,
            "n_runs": n_runs
        }

        # compute mean and sd for each metric
        for m in METRIC_COLS:
            mean_val = df[m].mean()
            sd_val = df[m].std(ddof=1)
            row[f"univ_{m}"] = mean_val
            row[f"univ_{m}_sd"] = sd_val

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path, index=False)

    print(
        f"Built universalization summary for run_label='{run_label}' "
        f"over {len(rows)} maps and saved to {output_csv}"
    )
    return summary_df



def build_outcome_metrics(
    completion_csv="../data/scenarios/completion_times.csv",
    lookup_csv="../data/scenarios/scenario_lookup.csv",
    output_csv="../data/scenarios/outcome_metrics.csv",
):
    """
    Build a scenario-level outcome metrics CSV from completion_csv,
    using lookup_csv as a correspondance table.
    """
    completion_path = Path(completion_csv)
    lookup_path = Path(lookup_csv)
    output_path = Path(output_csv)

    comp_df = pd.read_csv(completion_path)
    lookup_df = pd.read_csv(lookup_path)

    agent_cols = [c for c in comp_df.columns if c.startswith("agent_")]
    line_df = comp_df[comp_df["condition"] == "line"].set_index("map") # Line baseline per map

    rows = []

    for _, row in lookup_df.iterrows():
        map_name = row["map_name"]
        cond = row["condition"]
        
        # Baseline line completion times   
        line_times = line_df.loc[map_name, agent_cols].to_numpy(dtype=float) 
        # Scenario after line-cutting completion times
        world_rows = comp_df[(comp_df["map"] == map_name) & (comp_df["condition"] == cond)]
        world_times = world_rows.iloc[0][agent_cols].to_numpy(dtype=float)

        metrics = compute_metrics(world=world_times, line=line_times)

        out = {
            "scenario_label": row.get("preferred_label", None),
            "xp_name": row.get("xp_name", None),
            "map_name": map_name,
            "condition": cond,
            "aggregate_welfare": metrics["aggregate_welfare"],
            "inequality": metrics["inequality"],
            "cardinal_harm": metrics["cardinal_harm"],
            "ordinal_harm_blind": metrics["ordinal_harm_blind"],
            "gini": metrics["gini"],
        }
        rows.append(out)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_path, index=False)

    print(
        f"Built {len(metrics_df)} scenario outcome rows and saved to {output_path}"
    )
    return metrics_df


# ---------------------------------------------------------------------
# Helpers to call from notebooks
# ---------------------------------------------------------------------

def get_universalization_summary(
    run_label,
    sim_root="../data/simulations",
    summary_csv="summary_universalization_metrics.csv",
    line_csv="../data/scenarios/completion_times.csv",
    recompute=False,
):
    """
    Load universalization metrics for run_label.

    If recompute=False: read summary_csv.
    If recompute=True: rebuild it using build_universalization_summary()
    and build_processed_sim_all_maps().
    """
    sim_root=Path(sim_root)

    if recompute:
        raw_dir = sim_root / run_label / "raw"
        raw_files = sorted(raw_dir.glob("*.csv"))

        if not raw_files:
            raise FileNotFoundError(
                f"No raw CSV files found for run_label='{run_label}'.\n"
                f"Expected files under: {raw_dir}"
            )
        # rebuild per-map processed files
        build_processed_sim_all_maps(
            run_label=run_label,
            sim_root=sim_root,
            line_csv=line_csv
        )
        # rebuild summary over maps
        return build_universalization_summary(
            run_label=run_label,
            sim_root=sim_root,
            output_csv=summary_csv
            )
    else:
        summary_path = sim_root / run_label / summary_csv
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Summary CSV not found for run_label='{run_label}'.\n"
                f"Expected: {summary_path}\n"
                f"To rebuild from raw simulation files call get_universalization_summary(run_label, recompute=True)."
            )
        return pd.read_csv(summary_path)



def get_outcome_metrics(
    completion_csv="../data/scenarios/completion_times.csv",
    lookup_csv="../data/scenarios/scenario_lookup.csv",
    outcome_csv="../data/scenarios/outcome_metrics.csv",
    recompute=False,
):
    """
    Load the scenario outcome metrics.

    If recompute=False: read outcome_csv.
    If recompute=True: rebuild it using build_outcome_metrics().
    """
    if recompute:
        return build_outcome_metrics(
            completion_csv=completion_csv,
            lookup_csv=lookup_csv,
            output_csv=outcome_csv
        )
    else:
        outcome_path = Path(outcome_csv)
        return pd.read_csv(outcome_path)
    

def build_design_matrix(
    univ_df,
    out_df,
    xp_df,
    country=None,
):
    """
    Build a regression-ready design matrix (dataframe),
    using universalization, output and experimental data dfs,
    typically retrieved with above helpers.

    Ratings are pooled across countries by default;
    and otherwise limited to those obtained for a specified country.
    """
    # normalize xp column names
    xp_df = xp_df.rename(columns={"map": "xp_name"}).copy()

    if country is not None:
        # restrict to a single country
        xp_df = xp_df[xp_df["country"] == country].copy()
        xp_scenario = xp_df.rename(columns={"rating_mean": "rating_mean_country"})
    else:
        xp_scenario = (
            xp_df.groupby("xp_name", as_index=False)
                 .agg(rating_mean=("rating_mean", "mean"))
        )

    scenario_univ = out_df.merge(univ_df, on="map_name", how="left")

    design = xp_scenario.merge(scenario_univ, on="xp_name", how="inner")
    return design
