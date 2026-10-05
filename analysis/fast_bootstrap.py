"""
fast_bootstrap.py
=================

Drop-in accelerator for bootstrap_significance.

WHY: `_resample_means` rebuilds the long dataframe participant-by-participant
on every iteration:

    frames = [long[long[pcol] == p] for p in counts.index for _ in ...]
    boot_long = pd.concat(frames)

That is O(n_participants x len(long)) PER ITERATION. Fine for Joe (43
participants, ~1.7k rows). For Logan's individual trials (1,924
participants, ~39k rows) one iteration does ~75M row-scans, so 1,000
iterations hang for hours — the lag you saw on the third line of the
"verdict sentence" cell and throughout the by-country section.

FIX: precompute one participant x stimulus matrix of judgments (once per
dataset), then each bootstrap iteration is: sample row indices with
replacement, nanmean over rows. Identical resampling draws (same rng
stream), same stimulus means (participants rate each stimulus at most
once in these datasets), ~500x faster on Logan.

USE (one cell, before any bootstrap call):

    import fast_bootstrap; fast_bootstrap.install()

Then bootstrap_over_participants / bootstrap_delta_r2 / bootstrap_pairwise
run unchanged, seeds reproduce.
"""

import numpy as np
import pandas as pd

_CACHE = {}


def _matrix_for(long, pcol, jcol, scol):
    key = (id(long), pcol, jcol, scol, len(long))
    if key not in _CACHE:
        mat = long.pivot_table(index=pcol, columns=scol, values=jcol,
                               aggfunc="mean")
        _CACHE[key] = mat
    return _CACHE[key]


def _fast_resample_means(long, participants, pcol, jcol, scol, rng):
    """Same signature, same rng draws, same result as the original."""
    mat = _matrix_for(long, pcol, jcol, scol)
    resampled = rng.choice(participants, size=len(participants), replace=True)
    sub = mat.loc[resampled].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(sub, axis=0)
    return pd.Series(means, index=mat.columns)


def install():
    import bootstrap_significance as bs
    bs._resample_means = _fast_resample_means
    print("fast_bootstrap installed: _resample_means vectorized "
          "(cached participant x stimulus matrix).")
