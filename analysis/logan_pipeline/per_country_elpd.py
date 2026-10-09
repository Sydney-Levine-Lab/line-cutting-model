"""Per-country decomposition of paired ELPD differences. Usage: per_country_elpd.py <anon_dir> <fit_A> <fit_B> <out_csv>"""
import sys
from pathlib import Path
import numpyro.infer.initialization  # noqa
import arviz as az, numpy as np, polars as pl
anon, fa, fb, out = map(Path, sys.argv[1:5])
ia = az.from_netcdf(str(fa/"samples.nc")); obs = ia["observed_data"].dataset["judgment"].obs.values
ea = np.load(fa/"elpd_i.npy"); eb = np.load(fb/"elpd_i.npy")
trials = pl.read_parquet(anon/"trials.parquet").select("trial_id","country")
df = pl.DataFrame({"trial_id": obs, "d": ea-eb}).with_columns(pl.col("trial_id").cast(trials["trial_id"].dtype)).join(trials, on="trial_id", how="left")
per = df.group_by("country").agg(n_trials=pl.len(), delta_elpd=pl.col("d").sum(), se=(pl.col("d").std(ddof=1)*pl.len().sqrt())).sort("delta_elpd")
per.write_csv(out); print(f"total {df['d'].sum():+.1f} -> {out}")
