"""
Fit Logan Walls's hierarchical ordinal judgment model with a chosen simulation
model's universalization score as the `univ_agg_welfare` predictor.

This is `scripts/judgment_model.py` from the line-cutting repo (branch
`cleanup`) with three additions, marked `# ADDED`:
  --model    which U_AW column of model_scores.csv replaces Joe's univ_agg_welfare
             (`joe` reproduces Logan's analysis exactly: the column is Joe's own)
  --subset   all | dopt — restrict trials to the D-optimal stimuli (lookup flag)
  pointwise log-likelihood saved to the netCDF for PSIS-LOO model comparison.
Everything else — data loading, exclusions, standardization, model, sampler
settings, output format — is imported from `line_cutting` and unchanged.

Run from Logan's repo root (it reads data/per-stimulus-measures.csv etc. via
relative paths, as his script does), with his environment:

    uv run python <path>/fit_judgment_model.py <anon_dir> <out_dir> \
        --scores <path>/model_scores.csv --model L1h --subset all
"""

import json
import sys
import typing
from pathlib import Path

# numpyro must be imported before arviz: arviz_base loads numpyro lazily and, if
# arviz comes first, its proxy lacks `numpyro.infer.initialization`, which
# `az.from_numpyro` needs (AttributeError inside to_inference_data). Logan's
# script avoids this only because line_cutting.analysis.io imports numpyro first.
import numpyro.infer.initialization  # noqa: F401  (import order matters, see above)
import arviz as az
import jax
import numpy as np
import polars as pl
import xarray as xr
from cyclopts import App
from numpyro.infer import log_likelihood
from rich.console import Console

from line_cutting.analysis.anonymize import read_anonymized
from line_cutting.analysis.io import read_map_data
from line_cutting.analysis.stimulus_selection import DEFAULT_DESIGN_FACTORS
from line_cutting.models import util
from line_cutting.models.inference_data import PolarsInferenceData
from line_cutting.models.ordinal import build_inputs, hierarchical_judgment_model
from line_cutting.models.pipeline import (
    configure_jax_runtime,
    run_judgment_mcmc,
    save_population_trace_plot,
    to_inference_data,
)

console = Console()
app = App(console=console)

UNIV_COL = "univ_agg_welfare"
MODELS = ("joe", "L0", "L1h", "L2h", "realL1", "realL2")


@app.default
def main(
    path: Path,
    out_path: Path,
    scores: Path,                                             # ADDED
    model: typing.Literal[*MODELS] = "joe",                   # ADDED
    subset: typing.Literal["all", "dopt"] = "all",            # ADDED
    seed: int = 1234,
    num_samples: int = 2000,
    n_chains: int = 4,
    device: str = "cpu",
    chain_method: typing.Literal["parallel", "sequential"] = "parallel",
    show_progress: bool = True,
    debug: bool = False,
):
    assert path.is_dir(), "path should be a directory holding an anonymized dataset"
    pl.set_random_seed(seed)
    n_chains, chain_method = configure_jax_runtime(device, n_chains, chain_method, debug)
    if not out_path.exists():
        out_path.mkdir(parents=True)

    stimulus_factors = list(DEFAULT_DESIGN_FACTORS)
    all_predictors = stimulus_factors

    map_data = read_map_data().select("stimulus", *stimulus_factors).collect()

    # ADDED: substitute the universalization predictor and (optionally) restrict
    # stimuli. Substitution happens before standardization, as in the original.
    score_table = pl.read_csv(scores)
    assert model in score_table.columns, f"{model!r} not in {scores}"
    sub = score_table.select("stimulus", "in_d_optimal", pl.col(model).alias(UNIV_COL))
    n_before = map_data.height
    map_data = map_data.drop(UNIV_COL).join(sub, on="stimulus", how="inner")
    assert map_data.height == n_before, "score table does not cover every stimulus in map data"
    if model == "joe":
        orig = read_map_data().select("stimulus", UNIV_COL).collect()
        chk = orig.join(map_data.select("stimulus", UNIV_COL), on="stimulus")
        assert np.allclose(chk[UNIV_COL], chk[f"{UNIV_COL}_right"], atol=1e-6), \
            "model=joe should reproduce Logan's univ_agg_welfare exactly"
    if subset == "dopt":
        map_data = map_data.filter(pl.col("in_d_optimal"))
    map_data = map_data.drop("in_d_optimal")
    console.print(f"[bold]model={model} subset={subset}[/bold]: {map_data.height} stimuli")

    data = read_anonymized(path).filter_excluded()
    model_df = (
        data.trials.filter(pl.col("trial_type") == "main", ~pl.col("is_repeat_trial"))
        .join(map_data, on="stimulus", how="inner")   # inner (was left): drops stimuli outside the subset
        .with_columns(
            *(util.standardize_values(pl.col(p)) for p in stimulus_factors),
            pl.col("answer") - 1,
        )
    )
    console.print(f"trials={model_df.height} participants={model_df['participant_uid'].n_unique()} "
                  f"countries={model_df['country'].n_unique()} stimuli={model_df['stimulus'].n_unique()}")

    inputs, metadata = build_inputs(df=model_df, predictor_names=all_predictors, judgment_col="answer")

    mcmc, prior_predictive, posterior_predictive = run_judgment_mcmc(
        hierarchical_judgment_model, inputs, out_path, seed=seed, num_samples=num_samples,
        n_chains=n_chains, chain_method=chain_method, show_progress=show_progress,
    )
    az_idata = to_inference_data(mcmc, inputs, metadata, prior=prior_predictive,
                                 posterior_predictive=posterior_predictive)

    # ADDED: pointwise log-likelihood (chain, draw, obs) for PSIS-LOO.
    ll = log_likelihood(hierarchical_judgment_model, mcmc.get_samples(group_by_chain=True),
                        inputs, batch_ndims=2)["judgment"]
    az_idata["log_likelihood"] = xr.Dataset(
        {"judgment": (("chain", "draw", "obs"), np.asarray(ll))},
        coords={"chain": az_idata["posterior"].dataset.chain,
                "draw": az_idata["posterior"].dataset.draw,
                "obs": metadata.obs_ids_N.to_numpy()},
    )

    idata = PolarsInferenceData(az_idata)
    idata.save(out_path / "samples.nc")
    save_population_trace_plot(az_idata, out_path / "trace_plots.png")

    # ADDED: compact summary next to the samples, so a run can be read without xarray.
    post = az_idata["posterior"].dataset
    mu = post["mu_slopes"].sel(predictor=UNIV_COL)
    pooled = mu.mean(dim="country").values.ravel()
    ss = az_idata["sample_stats"].dataset
    loo = az.loo(az_idata, var_name="judgment", pointwise=False)
    summary = {
        "model": model, "subset": subset, "seed": seed,
        "n_trials": int(model_df.height), "n_participants": int(model_df["participant_uid"].n_unique()),
        "n_stimuli": int(model_df["stimulus"].n_unique()), "n_countries": int(model_df["country"].n_unique()),
        "divergences": int(ss["diverging"].sum()),
        "max_rhat_population": float(az.rhat(az_idata, var_names=["cutpoints", "sigma_delta", "mu_slopes", "sigma_slopes"]).dataset.to_array().max()),
        "univ_slope_mean_over_countries": {"mean": float(pooled.mean()),
                                           "ci95": [float(x) for x in np.percentile(pooled, [2.5, 97.5])]},
        "univ_slope_by_country": {str(c): {"mean": float(mu.sel(country=c).mean()),
                                           "ci95": [float(x) for x in np.percentile(mu.sel(country=c).values.ravel(), [2.5, 97.5])]}
                                  for c in post.country.values},
        "elpd_loo": float(loo["elpd"]), "elpd_loo_se": float(loo["se"]), "p_loo": float(loo["p"]),
        "pareto_k_gt_0.7": (int((np.asarray(loo.pareto_k) > 0.7).sum())
                            if getattr(loo, "pareto_k", None) is not None else None),
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))
    console.print(json.dumps({k: v for k, v in summary.items() if k != "univ_slope_by_country"}, indent=2))


if __name__ == "__main__":
    try:
        app()
    except Exception:
        console.print_exception()
        sys.exit(1)
