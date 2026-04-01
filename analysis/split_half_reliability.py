"""
Split-half reliability analysis for Joe's participant data.

What this does:
1. Takes the raw per-participant judgments (each participant rated each stimulus)
2. Randomly splits participants into two halves (e.g., 21 vs 22 for N=43)
3. Computes the mean judgment per stimulus for each half
4. Correlates the two sets of stimulus means
5. Repeats 1000 times with different random splits
6. Reports the average correlation (split-half r) and Spearman-Brown correction

Why it matters:
- If split-half r is high (>0.9), participants agree on which stimuli are bad/good
- This sets a ceiling: no model can correlate better than sqrt(reliability) with the means
- If r is low, the data is too noisy to distinguish between models

Spearman-Brown correction:
- Split-half r underestimates reliability because each half has only N/2 participants
- The correction estimates what reliability would be with the full N:
  r_corrected = 2 * r_half / (1 + r_half)

Usage:
    python split_half_reliability.py

Expects files in ../data/experimental/:
    joe_univ_judgments_long.csv
    joe_moral_judgments_long.csv
"""

import pandas as pd
import numpy as np
import os

# ── Config ──
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "experimental")
N_SPLITS = 1000
SEED = 42

# ── Load data ──
print("=" * 60)
print("SPLIT-HALF RELIABILITY ANALYSIS")
print("=" * 60)

univ_file = os.path.join(DATA_DIR, "joe_univ_judgments_long.csv")
moral_file = os.path.join(DATA_DIR, "joe_moral_judgments_long.csv")

univ = pd.read_csv(univ_file)
moral = pd.read_csv(moral_file)

# ── Sanity checks ──
print("\n--- Data overview ---")
for name, df, col in [("Universalizability", univ, "univ_judgment"),
                       ("Moral", moral, "moral_judgment")]:
    n_part = df["participant_id"].nunique()
    n_stim = df["stimulus"].nunique()
    n_rows = len(df)
    expected = n_part * n_stim
    
    print(f"\n{name}:")
    print(f"  Participants: {n_part}")
    print(f"  Stimuli:      {n_stim}")
    print(f"  Rows:         {n_rows}  (expected {expected}, "
          f"{'OK' if n_rows == expected else 'MISSING ' + str(expected - n_rows)})")
    print(f"  Judgment range: [{df[col].min()}, {df[col].max()}]")
    print(f"  Judgment mean:  {df[col].mean():.1f}, sd: {df[col].std():.1f}")
    
    # Check for excluded participants
    if "excluded" in df.columns:
        n_excl = df[df["excluded"]]["participant_id"].nunique()
        print(f"  Excluded participants: {n_excl}")
    
    # Check: do all participants rate all stimuli?
    counts = df.groupby("participant_id")["stimulus"].nunique()
    if counts.min() < n_stim:
        print(f"  WARNING: some participants rated only {counts.min()} stimuli")


# ── Split-half function ──
def split_half(df, participant_col, stimulus_col, judgment_col,
               n_splits=1000, seed=42):
    """
    Compute split-half reliability.
    
    For each of n_splits random splits:
    1. Divide participants randomly into two halves
    2. Compute mean judgment per stimulus for each half
    3. Correlate the two sets of means
    
    Returns dict with results.
    """
    rng = np.random.RandomState(seed)
    participants = df[participant_col].unique()
    n = len(participants)
    
    correlations = []
    
    for i in range(n_splits):
        # Shuffle and split
        perm = rng.permutation(participants)
        half_a = set(perm[:n // 2])
        half_b = set(perm[n // 2:])
        
        # Mean per stimulus for each half
        means_a = (df[df[participant_col].isin(half_a)]
                   .groupby(stimulus_col)[judgment_col].mean())
        means_b = (df[df[participant_col].isin(half_b)]
                   .groupby(stimulus_col)[judgment_col].mean())
        
        # Align stimuli
        common = means_a.index.intersection(means_b.index)
        r = np.corrcoef(means_a[common].values, means_b[common].values)[0, 1]
        correlations.append(r)
    
    correlations = np.array(correlations)
    r_mean = np.mean(correlations)
    r_sb = 2 * r_mean / (1 + r_mean)  # Spearman-Brown
    
    return {
        "n_participants": n,
        "n_stimuli": df[stimulus_col].nunique(),
        "n_splits": n_splits,
        "r_half_mean": r_mean,
        "r_half_sd": np.std(correlations),
        "r_half_min": np.min(correlations),
        "r_half_max": np.max(correlations),
        "r_half_ci95_lo": np.percentile(correlations, 2.5),
        "r_half_ci95_hi": np.percentile(correlations, 97.5),
        "r_spearman_brown": r_sb,
        "noise_ceiling": np.sqrt(r_sb),
        "all_correlations": correlations,
    }


# ── Run analysis ──
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

results = {}
for name, df, col in [("Joe universalizability", univ, "univ_judgment"),
                       ("Joe moral", moral, "moral_judgment")]:
    # Exclude if flagged
    if "excluded" in df.columns:
        df_clean = df[~df["excluded"]].copy()
    else:
        df_clean = df.copy()
    
    res = split_half(df_clean, "participant_id", "stimulus", col,
                     n_splits=N_SPLITS, seed=SEED)
    results[name] = res
    
    print(f"\n{name} (N={res['n_participants']}, {res['n_stimuli']} stimuli):")
    print(f"  Split-half r:     {res['r_half_mean']:.3f}  "
          f"(sd={res['r_half_sd']:.3f}, range=[{res['r_half_min']:.3f}, {res['r_half_max']:.3f}])")
    print(f"  95% CI:           [{res['r_half_ci95_lo']:.3f}, {res['r_half_ci95_hi']:.3f}]")
    print(f"  Spearman-Brown:   {res['r_spearman_brown']:.3f}")
    print(f"  Noise ceiling:    {res['noise_ceiling']:.3f}")


# ── Sanity check: is this too good? ──
print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

print("""
Q: Is r = 0.97 suspiciously high?
A: Not necessarily. Here's why:

1. The stimuli are VERY different from each other. Some maps have clear
   line violations (yes_line_*), others are ambiguous (maybe_*), others
   have no violation (no_line_*). Participants all agree that cutting
   is bad and not-cutting is fine. The between-stimulus variance is huge.

2. Split-half r is inflated by between-stimulus variance. If stimuli
   span a wide range and participants roughly agree on the ordering,
   you get high correlations even with noisy individual judgments.

3. The real question is whether the DATA DISTINGUISHES BETWEEN MODELS.
   With r = 0.97, the means are stable. But your best model only
   correlates at r = 0.84 with univ judgments, well below the ceiling.
""")

# Show the between-stimulus vs within-stimulus variance
for name, df, col in [("Joe universalizability", univ, "univ_judgment"),
                       ("Joe moral", moral, "moral_judgment")]:
    if "excluded" in df.columns:
        df_clean = df[~df["excluded"]].copy()
    else:
        df_clean = df.copy()
    
    stim_means = df_clean.groupby("stimulus")[col].mean()
    stim_sds = df_clean.groupby("stimulus")[col].std()
    
    between_var = stim_means.var()
    within_var = (stim_sds ** 2).mean()
    
    print(f"\n{name}:")
    print(f"  Between-stimulus variance (of means): {between_var:.1f}")
    print(f"  Within-stimulus variance (mean of sds²): {within_var:.1f}")
    print(f"  Ratio (between/within): {between_var / within_var:.1f}")
    print(f"  → High ratio means stimuli are very spread out relative to")
    print(f"    individual noise, which explains the high split-half r.")

    # Also show the actual stimulus means to see the spread
    print(f"\n  Stimulus means (sorted):")
    for stim, mean in stim_means.sort_values().items():
        sd = stim_sds[stim]
        n = df_clean[df_clean["stimulus"] == stim].shape[0]
        se = sd / np.sqrt(n)
        print(f"    {stim:20s}  mean={mean:7.1f}  sd={sd:5.1f}  se={se:5.1f}  n={n}")


# ── Save results ──
out_rows = []
for name, res in results.items():
    out_rows.append({
        "measure": name,
        "n_participants": res["n_participants"],
        "n_stimuli": res["n_stimuli"],
        "split_half_r": round(res["r_half_mean"], 4),
        "split_half_ci_lo": round(res["r_half_ci95_lo"], 4),
        "split_half_ci_hi": round(res["r_half_ci95_hi"], 4),
        "spearman_brown": round(res["r_spearman_brown"], 4),
        "noise_ceiling": round(res["noise_ceiling"], 4),
    })

out_df = pd.DataFrame(out_rows)
out_path = os.path.join(os.path.dirname(__file__), "figures", "split_half_reliability.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
out_df.to_csv(out_path, index=False)
print(f"\nSaved results to {out_path}")
print(out_df.to_string(index=False))
