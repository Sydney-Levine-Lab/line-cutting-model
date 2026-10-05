"""
logan_individual.py
===================

Loader for Logan's cross-cultural INDIVIDUAL-TRIAL export
(`cross-cultrual-ind-trials.csv`: trial_id, participant_id, country,
stimulus, answer on a 1-5 scale).

What this unlocks vs the old aggregated file (`country_agg_judgment.csv`):
`ExperimentalDataset.long_df` is populated, so `bootstrap_over_participants`
/ `bootstrap_delta_r2` / `bootstrap_pairwise` work on the cross-cultural
data — CIs over Logan's participants, pooled or per country.

Usage
-----
    from logan_individual import load_logan_individual

    pooled, by_country = load_logan_individual()          # default paths
    datasets["logan_moral"] = pooled                      # drop-in replacement
    # per-country fits:
    for c, exp in by_country.items():
        ev = evaluate_fits(dm_logan, exp)                 # same design mats
"""

from pathlib import Path
import pandas as pd

try:
    from new_comparison_utils import ExperimentalDataset, DEFAULT_PATHS
except ImportError:                       # standalone use / testing
    ExperimentalDataset = None
    DEFAULT_PATHS = {"experimental_dir": Path("../data/experimental"),
                     "lookup_csv": Path("../data/scenarios/scenario_lookup.csv")}

MIN_TRIALS_PER_PARTICIPANT = 10   # drop participants with fewer scored trials


def load_logan_individual(csv_file="cross-cultrual-ind-trials.csv",
                          experimental_dir=None, lookup_csv=None,
                          min_participants_per_country=20):
    """
    Returns (pooled, by_country):
      pooled     : ExperimentalDataset over all countries (with long_df)
      by_country : {country: ExperimentalDataset} (each with long_df)
    """
    experimental_dir = Path(experimental_dir or DEFAULT_PATHS["experimental_dir"])
    lookup_csv = Path(lookup_csv or DEFAULT_PATHS["lookup_csv"])

    df = pd.read_csv(experimental_dir / csv_file)

    # -- clean ---------------------------------------------------------
    df = df[~df["stimulus"].str.startswith("demo")].copy()

    # name-mapping through the canonical lookup, when available
    if lookup_csv.exists():
        lookup = pd.read_csv(lookup_csv)
        if "cogsci 26" in lookup.columns:
            label_map = dict(zip(lookup["cogsci 26"], lookup["preferred_label"]))
            df["stimulus"] = df["stimulus"].map(label_map).fillna(df["stimulus"])
    else:
        print(f"  NOTE: {lookup_csv} not found — using raw stimulus names "
              f"(they already follow preferred_label conventions).")

    # drop participants with too few trials (partial sessions)
    counts = df.groupby("participant_id")["answer"].size()
    keep = counts[counts >= MIN_TRIALS_PER_PARTICIPANT].index
    dropped = df["participant_id"].nunique() - len(keep)
    if dropped:
        print(f"  Dropped {dropped} participants with "
              f"<{MIN_TRIALS_PER_PARTICIPANT} trials.")
    df = df[df["participant_id"].isin(keep)]

    # participant_id values REPEAT across countries (hash collisions /
    # per-country id spaces): 521 raw uniques vs ~1,900 real participants.
    # Pooled analyses must use a country-qualified id or bootstrap
    # resampling silently merges different people.
    df["pid_global"] = df["country"].astype(str) + ":" + df["participant_id"].astype(str)

    def _make(sub, name, desc, pcol):
        agg = (sub.groupby("stimulus", as_index=False)
                  .agg(judgment_mean=("answer", "mean")))
        return ExperimentalDataset(
            name=name, agg_df=agg, dv_col="judgment_mean", long_df=sub,
            participant_col=pcol, judgment_col="answer",
            stimulus_col="stimulus", description=desc)

    pooled = _make(df, "Logan moral (pooled, ind)",
                   f"Cross-cultural moral judgments, individual trials; "
                   f"{df['pid_global'].nunique()} participants, "
                   f"{df['country'].nunique()} countries", "pid_global")

    by_country = {}
    for c, sub in df.groupby("country"):
        n = sub["participant_id"].nunique()
        if n < min_participants_per_country:
            print(f"  Skipping {c}: only {n} participants.")
            continue
        by_country[c] = _make(sub.copy(), f"Logan moral ({c})",
                              f"{n} participants", "participant_id")

    return pooled, by_country
