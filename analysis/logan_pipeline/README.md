# Logan-pipeline model comparison

Evaluates each simulation model's universalization score (U_AW) as a predictor of
trial-level moral judgments, using Logan Walls's hierarchical ordinal model
(`line_cutting` package, repo `Sydney-Levine-Lab/line-cutting`, branch `cleanup`)
**unchanged**. Only three small scripts live here; nothing in Logan's package is
modified or copied.

## Layout

| file | role |
|---|---|
| `export_model_scores.py` | simulation repo → `model_scores.csv` (one row per stimulus, one column per model, plus Joe's 2023 score and the D-optimal flag) |
| `make_anon_export.py` | wraps the May-2026 cross-cultural trial CSV into Logan's anonymized parquet format (stopgap until Logan sends his own export of the full dataset; see docstring for the reconstructed session/repeat structure) |
| `fit_judgment_model.py` | Logan's `scripts/judgment_model.py` + `--model`, `--subset`, saved pointwise log-likelihood, `summary.json` |
| `model_scores.csv` | output of `export_model_scores.py` for the final poster runs (`*_ghost_flat`, `*_fixed_flat`, 10 runs/map) |

## Environment

Logan's `uv.lock` (Python 3.13, jax, numpyro, arviz ≥ 1.1, polars). From his repo
root: `uv sync`. All commands below run from **his** repo root because
`read_map_data()` uses relative paths (`data/per-stimulus-measures.csv`, …).

## Commands

```bash
# 1. scores (from the simulation repo root)
python analysis/logan_pipeline/export_model_scores.py \
    --joe-measures ../logan-line-cutting/data/per-stimulus-measures.csv

# 2. data in Logan's format (from his repo root)
uv run python <sim-repo>/analysis/logan_pipeline/make_anon_export.py \
    <sim-repo>/data/experimental/cross-cultrual-ind-trials.csv data/anon/crosscultural_may2026
# the Sept-26 pilot parquet files are already in his format: put them in data/anon/pilot_sept26/
# as participants.parquet + trials.parquet

# 3. one fit (≈10 min pilot, ≈1–2 h full data, on 2 CPU cores)
uv run python <sim-repo>/analysis/logan_pipeline/fit_judgment_model.py \
    data/anon/crosscultural_may2026 output/full_L1h \
    --scores <sim-repo>/analysis/logan_pipeline/model_scores.csv --model L1h --subset all
```

`--model joe --subset all` reproduces Logan's analysis exactly (asserted in the
script). Models: `joe L0 L1h L2h realL1 realL2`; subsets: `all` (39 stimuli),
`dopt` (the 20 D-optimal stimuli).

## Outputs per fit

`samples.nc` (ArviZ DataTree incl. `log_likelihood`), `trace_plots.png`,
`model-diagram.svg`, `summary.json` (slope posterior of the universalization
predictor per country and averaged; divergences; max R-hat; PSIS-LOO ELPD).

## Known issues

* `arviz_base` must see `numpyro` imported before `arviz` (lazy-loader bug);
  the driver does this explicitly.
* PSIS-LOO: with per-participant slopes on 13 trials/person, a few percent of
  observations have Pareto k > 0.7. Compare ELPD with that caveat, or switch to
  moment-matched LOO / K-fold if needed.
* Reproduction target for the pilot (Joe's model, Sept-26 pilot, 20 stimuli):
  U_AW slope MX .48 [.31,.66], UK .32 [.16,.49], US .63 [.44,.82];
  mean over countries .48 [.37,.58]; 1 divergence / 8000; all R-hat 1.00.
