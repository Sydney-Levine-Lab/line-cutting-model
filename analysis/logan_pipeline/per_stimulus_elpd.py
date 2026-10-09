"""Per-stimulus decomposition of paired ELPD differences between two fits on the same data.
Usage: uv run python per_stimulus_elpd.py <anon_dir> <fit_A> <fit_B> <out_csv>
Each fit dir must have samples.nc (obs coord = trial_id) and elpd_i.npy (from summarize_fit.py)."""
import sys
from pathlib import Path
import numpyro.infer.initialization  # noqa
import arviz as az, numpy as np, polars as pl
anon, fa, fb, out = map(Path, sys.argv[1:5])
ia = az.from_netcdf(str(fa/"samples.nc")); ib = az.from_netcdf(str(fb/"samples.nc"))
obs_a = ia["observed_data"].dataset["judgment"].obs.values; obs_b = ib["observed_data"].dataset["judgment"].obs.values
assert np.array_equal(obs_a, obs_b), "fits are not on the same trials"
ea = np.load(fa/"elpd_i.npy"); eb = np.load(fb/"elpd_i.npy")
trials = pl.read_parquet(anon/"trials.parquet").select("trial_id","stimulus","country")
df = pl.DataFrame({"trial_id": obs_a, "d": ea-eb}).with_columns(pl.col("trial_id").cast(trials["trial_id"].dtype)).join(trials, on="trial_id", how="left")
assert df["stimulus"].null_count()==0
per = df.group_by("stimulus").agg(n_trials=pl.len(), delta_elpd=pl.col("d").sum(), mean_per_trial=pl.col("d").mean(),
                                  se=(pl.col("d").std(ddof=1)*pl.len().sqrt())).sort("delta_elpd")
per.write_csv(out); print(f"total ΔELPD = {df['d'].sum():+.1f} over {df.height} trials, {per.height} stimuli -> {out}")
