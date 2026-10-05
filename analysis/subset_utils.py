"""
subset_utils.py — restrict the analysis to a flagged subset of stimuli
(e.g. Logan's D-optimal set, column `in_d_optimal` in scenario_lookup.csv).

Usage in the poster notebook, right after datasets / dm are built:

    from subset_utils import load_stimulus_flag, restrict_all

    KEEP = load_stimulus_flag()                      # set of 20 preferred_labels
    datasets_d, dm_d = restrict_all(datasets, dm, KEEP)

    ev_univ_d = evaluate_fits(dm_d["joe_univ"], datasets_d["joe_univ"])
    ci, boot_d = bootstrap_over_participants(dm_d["joe_univ"], datasets_d["joe_univ"],
                                             n_boot=N_BOOT_POSTER, seed=BOOT_SEED)
    ...

Everything downstream (evaluate_fits, bootstrap_*, split_half_all, plots) is
unchanged: it just receives filtered objects. The filter is applied to the
design matrices AND to each dataset's agg_df + long_df, so participant
bootstraps resample only trials on the kept stimuli.
"""

import pandas as pd
from new_comparison_utils import ExperimentalDataset, DEFAULT_PATHS


def load_stimulus_flag(lookup_csv=None, col="in_d_optimal", value=True):
    """Return the set of preferred_labels whose `col` == value in scenario_lookup."""
    lookup = pd.read_csv(lookup_csv or DEFAULT_PATHS["lookup_csv"])
    if col not in lookup.columns:
        raise KeyError(f"'{col}' not in {list(lookup.columns)} — add the flag column first.")
    flag = lookup[col].astype(str).str.lower().isin({"true", "1", "yes"}) if value else \
           ~lookup[col].astype(str).str.lower().isin({"true", "1", "yes"})
    return set(lookup.loc[flag, "preferred_label"])


def restrict_dataset(experiment, keep, suffix=" [subset]"):
    """Copy of an ExperimentalDataset with agg_df and long_df limited to `keep`."""
    agg = experiment.agg_df[experiment.agg_df["stimulus"].isin(keep)].copy()
    long = None
    if experiment.long_df is not None:
        scol = experiment.stimulus_col
        long = experiment.long_df[experiment.long_df[scol].isin(keep)].copy()
    return ExperimentalDataset(
        name=experiment.name + suffix,
        agg_df=agg,
        dv_col=experiment.dv_col,
        long_df=long,
        participant_col=experiment.participant_col,
        judgment_col=experiment.judgment_col,
        stimulus_col=experiment.stimulus_col,
        description=experiment.description + suffix,
    )


def restrict_design_mats(dm, keep):
    """dm: dict[tag -> design matrix] for ONE experiment."""
    return {tag: df[df["stimulus"].isin(keep)].copy() for tag, df in dm.items()}


def restrict_all(datasets, dm, keep, suffix=" [subset]", verbose=True):
    """
    datasets: dict[exp_key -> ExperimentalDataset]
    dm:       dict[exp_key -> dict[tag -> design matrix]]
    Returns (datasets_sub, dm_sub) with the same keys.
    """
    datasets_sub, dm_sub = {}, {}
    for key, exp in datasets.items():
        datasets_sub[key] = restrict_dataset(exp, keep, suffix)
        if key in dm:
            dm_sub[key] = restrict_design_mats(dm[key], keep)
        if verbose:
            n_before = exp.agg_df["stimulus"].nunique()
            n_after = datasets_sub[key].agg_df["stimulus"].nunique()
            n_maps = ""
            if key in dm_sub and dm_sub[key]:
                first = next(iter(dm_sub[key].values()))
                n_maps = f", {first['map'].nunique()} maps"
            print(f"  {key:15s} {n_before} -> {n_after} stimuli{n_maps}")
    return datasets_sub, dm_sub
