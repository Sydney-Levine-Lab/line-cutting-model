# analysis/python/sim_analysis.py

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------
# Core parsing utilities
# ---------------------------------------------------------------------

def parse_dict_string(dict_str: str) -> dict:
    """
    Parse a Julia-style Dict string into a Python dict.

    Example input:
        "Dict(5 => Dict(5 => 75, 4 => 86, ...), 4 => Dict(...), ...)"

    Output:
        {5: {5: 75, 4: 86, ...}, 4: {...}, ...}
    """
    # Replace Dict( with { and ) with }
    dict_str = dict_str.replace('Dict(', '{').replace(')', '}')
    # Replace Julia-style '=>' with Python ':'
    dict_str = dict_str.replace(' => ', ': ')
    return ast.literal_eval(dict_str)


### MODIFIED Nov 24: some of level 0 files had 2 rows, one for a 5 run test and otehr for a 50 run test --- ignore the fist one

def load_sim_file(filepath) -> pd.DataFrame:
    """
    Load a Julia-produced CSV of the form:
        param, data
        0.0001, Dict(1 => Dict(1 => 42, ...), 2 => Dict(...), ...)

    Some files (esp. old ones) may have multiple rows for the same noise
    because results were appended (e.g. a 5-run test, then a 50-run batch).
    In that case, we pick the row whose Dict has the *most* iterations.

    Returns:
        DataFrame with shape [runs/iterations × agents], e.g. 50 × 8, with:
            rows    = Iter_1, Iter_2, ...
            columns = Agent_1, Agent_2, ...
    """
    filepath = Path(filepath)
    df_raw = pd.read_csv(filepath, header=None, names=["param", "data"])

    if df_raw.shape[0] == 0:
        raise ValueError(f"No rows found in sim CSV {filepath}")

    # Parse each row's Dict and track how many iterations it has
    best_idx = None
    best_dict = None
    best_len = -1

    for i, row in df_raw.iterrows():
        dict_str = row["data"]
        if not isinstance(dict_str, str):
            continue
        try:
            data_dict = parse_dict_string(dict_str)
        except Exception:
            continue

        n_iter = len(data_dict)
        if n_iter > best_len:
            best_len = n_iter
            best_idx = i
            best_dict = data_dict

    if best_dict is None:
        raise ValueError(
            f"Could not parse any valid Dict from sim CSV {filepath}"
        )

    iterations = sorted(best_dict.keys())
    agents = sorted(best_dict[iterations[0]].keys())

    data = pd.DataFrame(
        [[best_dict[it][agent] for agent in agents] for it in iterations],
        columns=[f"Agent_{agent}" for agent in agents],
        index=[f"Iter_{it}" for it in iterations],
    )
    return data


## NEW (nov 24): thing to convert into readable csv
import re
from pathlib import Path

import re
from pathlib import Path

def convert_raw_sim_to_clean(raw_path, out_dir=None):
    """
    Convert a raw nested-Dict sim CSV like:
        MAPNAME.pddl_0.0001_50.csv
        MAPNAME.pddl_0.0001.csv
    into a clean CSV:
        MAPNAME.csv

    Output columns:
        run, noise, agent_1..agent_8

    raw_path : path to the original CSV
    out_dir  : directory to write the clean CSV to.
               If None, uses the same directory as raw_path.
    """
    raw_path = Path(raw_path)

    # 1) Load as [runs × Agent_1..Agent_8] using existing helper
    df = load_sim_file(raw_path)   # columns: Agent_1..Agent_8, index: Iter_1..Iter_N

    # 2) Make a copy and reset the index so we don't see Iter_* in Jupyter
    df = df.copy()
    df.reset_index(drop=True, inplace=True)

    # 3) Create run index (1..N)
    df["run"] = range(1, len(df) + 1)

    # 4) Rename Agent_1..Agent_8 → agent_1..agent_8
    rename_map = {col: col.lower() for col in df.columns if col.startswith("Agent_")}
    df = df.rename(columns=rename_map)

    # 5) Parse noise from filename: MAPNAME.pddl_NOISE or MAPNAME.pddl_NOISE_50
    # Example filename: yes_line_10_test.pddl_0.0001_50.csv
    m = re.search(r"\.pddl_([^_]+)", raw_path.name)
    if m:
        noise_str = m.group(1)   # e.g. "0.0001"
    else:
        noise_str = None

    df["noise"] = noise_str

    # 6) Reorder columns: run, noise, agent_1..agent_8
    agent_cols = [c for c in df.columns if c.startswith("agent_")]
    cols = ["run", "noise"] + agent_cols
    df = df[cols]

    # 7) Decide output path: MAPNAME.csv (strip everything from .pddl_ onward)
    stem = raw_path.name.split(".pddl_")[0]    # "yes_line_10_test"
    stem = stem.replace("_test", "")           # → "yes_line_10"

    if out_dir is None:
        out_dir = raw_path.parent
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{stem}.csv"
    df.to_csv(out_path, index=False)

    return df, out_path



# ---------------------------------------------------------------------
# Optional: baseline loader (stub for later with-line CSVs)
# ---------------------------------------------------------------------

def load_withline_csv(filepath,                 ##filepath: str | Path (type hint for new python)
                      run_col: str = "run",
                      agent_prefix: str = "agent_") -> pd.DataFrame:
    """
    Load baseline 'with-line' CSV exported from Google Sheets.

    Expected format (can be adapted later):
        columns: run, agent_1, agent_2, ..., agent_8

    Returns:
        DataFrame with shape [runs × agents], columns Agent_1..Agent_8,
        index Iter_1..Iter_N.

    You can adjust run_col / agent_prefix when you know the exact column names.
    """
    filepath = Path(filepath)
    df_wide = pd.read_csv(filepath)

    # Identify agent columns
    agent_cols = [c for c in df_wide.columns if str(c).startswith(agent_prefix)]
    if not agent_cols:
        raise ValueError(
            f"No columns starting with '{agent_prefix}' found in {filepath}"
        )

    df = df_wide[agent_cols].copy()
    df.columns = [f"Agent_{i+1}" for i in range(len(agent_cols))]
    df.index = [f"Iter_{i+1}" for i in range(len(df))]
    return df


# ---------------------------------------------------------------------
# Summary & plotting helpers
# ---------------------------------------------------------------------

def summarize_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a [runs × agents] DataFrame, return per-agent summary stats:
        mean, sd, sem, ci_low, ci_high (95% CI using normal approx).
    """
    n = df.shape[0]
    mean = df.mean(axis=0)
    sd = df.std(axis=0, ddof=1)
    sem = sd / np.sqrt(n)
    z = 1.96  # 95% CI
    ci_low = mean - z * sem
    ci_high = mean + z * sem

    out = pd.DataFrame({
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci_low": ci_low,
        "ci_high": ci_high,
    })
    return out


def plot_single_file(
    #filepath: str | Path,
    *,
    show_individual: bool = False,
    individual_kind: str = "violin",
    show_average: bool = True,
    error: str = "ci",
    overlay_samples: int = 10,
    jitter_width: float = 0.08,
):
    """
    Visualize completion times for one map/condition.

    Parameters
    ----------
    filepath : str or Path
        Path to a Julia CSV (param, Dict(...) format) OR to a baseline CSV
        if you load it via load_withline_csv and adapt accordingly.
    show_individual : bool
        Whether to show per-run distributions per agent (violin or box plot).
    individual_kind : {"violin", "box"}
        Type of distribution plot.
    show_average : bool
        Whether to show mean ± error bars per agent.
    error : {"ci", "sem", "sd"}
        Type of error bar: 95% CI, standard error, or standard deviation.
    overlay_samples : int
        Number of individual run points to scatter per agent on the mean plot.
    jitter_width : float
        Horizontal jitter width for overlayed points.
    """
    data = load_sim_file(filepath)
    filename = Path(filepath).name

    # Configure style (can be moved elsewhere if you prefer)
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)

    # Decide number of subplot columns
    ncols = (1 if not show_individual else 2) if show_average else 1
    fig, axes = plt.subplots(1, ncols, figsize=(18 if ncols == 2 else 12, 6))
    if ncols == 1:
        axes = [axes]
    ax_i = 0

    # A) Distributions per agent (cleaner than bars for many runs)
    if show_individual:
        ax = axes[ax_i]
        df_long = data.reset_index().melt(
            id_vars="index", var_name="agent", value_name="value"
        )
        if individual_kind == "violin":
            sns.violinplot(
                data=df_long, x="agent", y="value", inner="quartile",
                ax=ax, cut=0
            )
        else:
            sns.boxplot(
                data=df_long, x="agent", y="value",
                ax=ax, showfliers=False
            )
        ax.set_title(f"Per-run distribution — {filename}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Agent")
        ax.set_ylabel("Completion time")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
        ax_i += 1

    # B) Mean + error bars
    if show_average:
        ax = axes[ax_i]
        summary = summarize_runs(data)
        x = np.arange(len(summary))
        means = summary["mean"].values

        if error == "ci":
            yerr = (summary["ci_high"] - summary["mean"]).values
            err_label = "95% CI"
        elif error == "sem":
            yerr = summary["sem"].values
            err_label = "SEM"
        else:
            yerr = summary["sd"].values
            err_label = "SD"

        bars = ax.bar(
            x, means, yerr=yerr, capsize=5, alpha=0.8,
            color="steelblue", ecolor="black"
        )
        ax.set_title(f"Average ± {err_label} — {filename}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Agent")
        ax.set_ylabel("Average completion time")
        ax.set_xticks(x)
        ax.set_xticklabels(summary.index, rotation=45)
        ax.grid(axis="y", alpha=0.3)

        # Optional: overlay a few individual runs for intuition
        if overlay_samples and overlay_samples > 0:
            rng = np.random.default_rng(42)
            for j, agent in enumerate(data.columns):
                vals = data[agent].values
                take = min(overlay_samples, len(vals))
                idxs = rng.choice(len(vals), size=take, replace=False)
                xs = j + rng.uniform(-jitter_width, jitter_width, size=take)
                ax.scatter(xs, vals[idxs], s=18, alpha=0.5)

        # Value labels (mean ± error)
        for bar, mu, err in zip(bars, means, yerr):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                mu + err,
                f"{mu:.2f}±{err:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()

    return data


def compare_files(filepaths, labels=None, error: str = "ci"):
    """
    Compare average completion times (± error) for multiple files
    on the same map (e.g., level-0 vs level-1 vs paper vs with-line).

    Parameters
    ----------
    filepaths : list[str or Path]
        Paths to CSVs to compare. All must correspond to the same map,
        and have 8 agents in the same order.
    labels : list[str], optional
        Labels to use in the legend; defaults to filenames.
    error : {"ci", "sem", "sd"}
        Error bar type.
    """
    if labels is None:
        labels = [Path(fp).name for fp in filepaths]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    n_agents = 8  # could also infer from data
    x = np.arange(n_agents)
    width = 0.8 / len(filepaths)

    for i, (filepath, label) in enumerate(zip(filepaths, labels)):
        data = load_sim_file(filepath)
        summary = summarize_runs(data)
        mu = summary["mean"].values

        if error == "ci":
            yerr = (summary["ci_high"] - summary["mean"]).values
            err_label = "95% CI"
        elif error == "sem":
            yerr = summary["sem"].values
            err_label = "SEM"
        else:
            yerr = summary["sd"].values
            err_label = "SD"

        offset = (i - len(filepaths) / 2 + 0.5) * width
        ax.bar(
            x + offset, mu, width, yerr=yerr, capsize=3,
            label=label, alpha=0.85
        )

    ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel("Average completion time", fontsize=12)
    ax.set_title(f"Comparison of averages ± {err_label}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Agent_{i+1}" for i in range(n_agents)], rotation=0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_folders(folder1, folder2, pattern="*.csv", exclude_collision=True):
    """
    Compare matching files between two folders, e.g. level-0 vs level-1.

    It will:
      - find common filenames in both dirs (matching `pattern`)
      - optionally exclude files starting with 'collision'
      - call compare_files() on each common filename.

    Useful for scanning level-0 vs level-1 differences across many maps.
    """
    folder1_path = Path(folder1)
    folder2_path = Path(folder2)

    files1 = {f.name: f for f in folder1_path.glob(pattern)}
    files2 = {f.name: f for f in folder2_path.glob(pattern)}

    if exclude_collision:
        files1 = {k: v for k, v in files1.items() if not k.startswith("collision")}
        files2 = {k: v for k, v in files2.items() if not k.startswith("collision")}

    common_files = set(files1.keys()) & set(files2.keys())
    print(f"Found {len(common_files)} matching files between folders")

    for filename in sorted(common_files):
        print("\n" + "=" * 80)
        print(f"Comparing: {filename}")
        print("=" * 80)
        compare_files(
            [files1[filename], files2[filename]],
            [f"{folder1}/{filename}", f"{folder2}/{filename}"],
        )
