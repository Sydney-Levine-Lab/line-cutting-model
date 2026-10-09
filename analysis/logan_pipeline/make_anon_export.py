"""
Wrap the May-2026 cross-cultural trial CSV (Logan's export of individual trials,
columns: trial_id, participant_id, country, stimulus, answer) into the anonymized
parquet pair that `line_cutting.analysis.anonymize.read_anonymized` expects.

What the CSV lacks and how it is reconstructed here
---------------------------------------------------
* No session / participant structure beyond `participant_id`, which collides
  across AND within countries (521 raw ids; 2,059 sessions). Every session is
  exactly 20 trials: 2 x demo_bad, 2 x yes_line_7_0cut (no-cut control), 13
  distinct real scenarios, 3 repeats of real scenarios. Sessions are therefore
  recovered as blocks of 20 consecutive trial_ids within (country, participant_id).
  A block that does not have that structure (two sessions interleaved) cannot be
  separated and the whole (country, participant_id) is dropped.
* No trial_type / is_repeat_trial flags. Within a session, the first trial of a
  stimulus is the main trial; later ones are repeats (the pilot's `norm_repeat`).
* No exclusion flags: exclusions are assumed already applied -> excluded = False.
* demo_bad and yes_line_7_0cut have no row in per-stimulus-measures.csv; they are
  kept but typed as "demo" so the model script's `trial_type == "main"` filter
  removes them (the script left-joins design factors and cannot take nulls).

Replace this export with Logan's own anonymized export of the full dataset as
soon as it is available; the model script is unchanged either way.

Usage:
    python make_anon_export.py <in.csv> <out_dir>
"""

import argparse
from pathlib import Path

import polars as pl

CONTROL_STIMULI = {"demo_bad", "yes_line_7_0cut"}
SESSION_LEN = 20
N_MAIN_EXPECTED = 13


def main(in_csv: Path, out_dir: Path) -> None:
    raw = (
        pl.read_csv(in_csv)
        .with_columns(uid=pl.col("country") + ":" + pl.col("participant_id").cast(pl.String))
        .sort("uid", "trial_id")
        .with_columns(session=(pl.int_range(pl.len()).over("uid") // SESSION_LEN))
    )
    is_ctrl = pl.col("stimulus").is_in(CONTROL_STIMULI)
    sess = raw.group_by("uid", "session").agg(
        n=pl.len(),
        n_demo=(pl.col("stimulus") == "demo_bad").sum(),
        n_ctrl=(pl.col("stimulus") == "yes_line_7_0cut").sum(),
        n_main=pl.col("stimulus").filter(~is_ctrl).n_unique(),
    ).with_columns(
        ok=(pl.col("n") == SESSION_LEN) & (pl.col("n_demo") == 2) & (pl.col("n_ctrl") == 2)
        & (pl.col("n_main") == N_MAIN_EXPECTED)
    )
    bad_uids = sess.filter(~pl.col("ok"))["uid"].unique()
    print(f"sessions found: {sess.height}; well-formed: {sess['ok'].sum()}; "
          f"dropping {bad_uids.len()} ids whose sessions interleave "
          f"({sess.filter(pl.col('uid').is_in(bad_uids)).height} sessions)")

    trials = (
        raw.filter(~pl.col("uid").is_in(bad_uids))
        .with_columns(participant_uid=pl.col("uid") + "/s" + pl.col("session").cast(pl.String))
        .with_columns(
            occurrence=pl.int_range(pl.len()).over("participant_uid", "stimulus"),
        )
        .with_columns(
            is_repeat_trial=pl.col("occurrence") > 0,
            trial_type=pl.when(is_ctrl).then(pl.lit("demo")).otherwise(pl.lit("main")),
            iso2=pl.lit(None, dtype=pl.String),
            iso3=pl.lit(None, dtype=pl.String),
            language=pl.lit(None, dtype=pl.String),
            data_batch=pl.lit("may2026_csv"),
            excluded=pl.lit(False),
        )
        .sort("participant_uid", "trial_id")
        .drop("trial_id", "participant_id", "uid", "session", "occurrence")
        .with_row_index("trial_id")
        .select(
            "trial_id", "participant_uid", "stimulus", "answer", "trial_type",
            "is_repeat_trial", "country", "iso2", "iso3", "language", "data_batch", "excluded",
        )
    )
    participants = (
        trials.group_by("participant_uid")
        .agg(pl.col("country").first(), pl.col("iso2").first(), pl.col("iso3").first(),
             pl.col("language").first(), pl.col("data_batch").first())
        .with_columns(
            complete=pl.lit(True), passed_comprehension=pl.lit(True),
            passed_attention=pl.lit(True), excluded=pl.lit(False),
        )
        .sort("participant_uid")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    participants.write_parquet(out_dir / "participants.parquet")
    trials.write_parquet(out_dir / "trials.parquet")

    main_trials = trials.filter(pl.col("trial_type") == "main", ~pl.col("is_repeat_trial"))
    per = main_trials.group_by("participant_uid").len()["len"]
    print(f"participants (sessions) kept: {participants.height}; countries: {participants['country'].n_unique()}")
    print(f"main non-repeat trials: {main_trials.height}; stimuli: {main_trials['stimulus'].n_unique()}; "
          f"per participant min {per.min()} max {per.max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args()
    main(a.in_csv, a.out_dir)
