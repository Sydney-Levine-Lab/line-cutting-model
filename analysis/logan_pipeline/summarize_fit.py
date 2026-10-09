"""
Build (or rebuild) summary.json for a finished fit from its samples.nc, and
optionally compute paired ELPD differences between fits.

    uv run python summarize_fit.py <fit_dir> [<fit_dir> ...]
    uv run python summarize_fit.py --compare <fit_dir_a> <fit_dir_b> ...   # pairwise table

Used when a run was interrupted after sampling, and for model comparison tables.
"""

import json
import sys
from pathlib import Path

import numpyro.infer.initialization  # noqa: F401  (must precede arviz; see fit_judgment_model.py)
import arviz as az
import numpy as np

UNIV_COL = "univ_agg_welfare"
POP = ["cutpoints", "sigma_delta", "mu_slopes", "sigma_slopes"]


def summarize(fit_dir: Path) -> dict:
    idata = az.from_netcdf(str(fit_dir / "samples.nc"))
    post = idata["posterior"].dataset
    ss = idata["sample_stats"].dataset
    loo = az.loo(idata, var_name="judgment", pointwise=True)
    mu = post["mu_slopes"].sel(predictor=UNIV_COL)
    pooled = mu.mean(dim="country").values.ravel()
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]
    prev = {}
    if (fit_dir / "summary.json").exists():
        prev = json.loads((fit_dir / "summary.json").read_text())
    out = {
        **{k: prev[k] for k in ("model", "subset", "seed", "num_warmup", "num_samples", "n_chains") if k in prev},
        "fit_dir": str(fit_dir),
        "n_trials": int(post.sizes.get("obs", idata["observed_data"].dataset.sizes["obs"])) if "obs" in post.sizes else int(idata["observed_data"].dataset.sizes["obs"]),
        "n_participants": int(post.sizes["individual"]),
        "n_countries": int(post.sizes["country"]),
        "n_draws": int(post.sizes["chain"] * post.sizes["draw"]),
        "divergences": int(ss["diverging"].sum()),
        "max_rhat_population": float(az.rhat(idata, var_names=POP).dataset.to_array().max()),
        "univ_slope_mean_over_countries": {"mean": float(pooled.mean()), "ci95": ci(pooled)},
        "univ_slope_by_country": {str(c): {"mean": float(mu.sel(country=c).mean()),
                                           "ci95": ci(mu.sel(country=c).values.ravel())}
                                  for c in post.country.values},
        "elpd_loo": float(loo["elpd"]), "elpd_loo_se": float(loo["se"]), "p_loo": float(loo["p"]),
        "pareto_k_gt_0.7": int((np.asarray(loo.pareto_k) > 0.7).sum()),
    }
    (fit_dir / "summary.json").write_text(json.dumps(out, indent=2))
    np.save(fit_dir / "elpd_i.npy", np.asarray(loo.elpd_i))
    return out


def compare(fit_dirs: list[Path]) -> None:
    e = {}
    for d in fit_dirs:
        f = d / "elpd_i.npy"
        if not f.exists():
            summarize(d)
        e[d.name] = np.load(f)
    names = list(e)
    print(f"{'':34s}" + "".join(f"{n[-14:]:>16s}" for n in names))
    for a in names:
        row = []
        for b in names:
            if a == b:
                row.append(f"{'—':>16s}")
            else:
                d = e[a] - e[b]
                row.append(f"{d.sum():+8.1f}±{np.sqrt(len(d) * d.var(ddof=1)):5.1f}")
        print(f"{a:34s}" + "".join(row))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--compare":
        compare([Path(a) for a in args[1:]])
    else:
        for a in args:
            s = summarize(Path(a))
            u = s["univ_slope_mean_over_countries"]
            print(f"{a}: slope={u['mean']:.3f} [{u['ci95'][0]:.3f},{u['ci95'][1]:.3f}] "
                  f"elpd={s['elpd_loo']:.1f}±{s['elpd_loo_se']:.1f} div={s['divergences']} rhat={s['max_rhat_population']:.3f} k>.7={s['pareto_k_gt_0.7']}")
