"""
Evaluate stored minutes projections and lineups against actuals, split by data-freshness regime.

Regimes are defined by when upstream inputs went stale during the 2025-26 season:
  A  fresh injuries + fresh box scores   (season start .. 2026-01-19)
  B  stale injuries + fresh box scores   (2026-02-22 .. 2026-03-25)
  C  stale injuries + stale box scores   (2026-03-26 ..)
Pass --regimes to override, e.g. --regimes "A:2025-10-01:2026-01-19,B:2026-02-22:2026-03-25".
"""

import argparse
import io
import warnings

import boto3
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

BUCKET = "nba-prediction-ibracken"
MINUTES_MODELS = ["complex_position_overlap", "direct_position_only", "formula_c_baseline"]
FP_MODELS = ["current", "fp_per_min", "barebones"]
DEFAULT_REGIMES = "A:2025-10-01:2026-01-19,B:2026-02-22:2026-03-25,C:2026-03-26:2026-12-31"

s3 = boto3.client("s3")


def load(key):
    return pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()))


def parse_regimes(spec):
    out = []
    for part in spec.split(","):
        name, start, end = part.split(":")
        out.append((name, pd.Timestamp(start), pd.Timestamp(end)))
    return out


def label_regime(dates, regimes):
    labels = pd.Series("other", index=dates.index)
    for name, start, end in regimes:
        labels[(dates >= start) & (dates <= end)] = name
    return labels


def minutes_report(regimes):
    rows = []
    for model in MINUTES_MODELS:
        df = load(f"model_comparison/{model}/minutes_projections.parquet")
        df["DATE"] = pd.to_datetime(df["DATE"])
        df["REGIME"] = label_regime(df["DATE"], regimes)
        for regime, g in df.groupby("REGIME"):
            projected_to_play = g[g.PROJECTED_MIN > 10]
            dnp = projected_to_play.ACTUAL_MIN.isna() | (projected_to_play.ACTUAL_MIN == 0)
            scored = g.dropna(subset=["ACTUAL_MIN", "PROJECTED_MIN"])
            scored = scored[(scored.PROJECTED_MIN > 0) & (scored.ACTUAL_MIN > 0)]
            err = scored.PROJECTED_MIN - scored.ACTUAL_MIN
            rows.append({
                "regime": regime, "model": model, "dates": g.DATE.nunique(),
                "scored": len(scored),
                "MAE": err.abs().mean(), "bias": err.mean(),
                "DNP_rate": dnp.mean() if len(projected_to_play) else np.nan,
                "n_proj>10": len(projected_to_play),
            })
    out = pd.DataFrame(rows).sort_values(["regime", "model"])
    print("\n=== MINUTES PROJECTIONS: accuracy by regime (players who actually played) ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("DNP_rate = share of players projected >10 MIN who logged 0 / no minutes")


def lineup_frames():
    frames = {}
    for m in MINUTES_MODELS:
        for f in FP_MODELS:
            frames[f"{m}|{f}"] = f"model_comparison/{m}/fp_{f}/daily_lineups.parquet"
    frames["DFF"] = "model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet"
    # Pre-FP-model lineups (Jan 6 archive + Jan 17 legacy file) cover the fresh-injury window
    for m in MINUTES_MODELS:
        frames[f"{m}|legacy"] = [
            f"model_comparison/{m}/archive/daily_lineups_pre_fp_models_2026-01-06.parquet",
            f"model_comparison/{m}/daily_lineups.parquet",
        ]
    return frames


def lineup_report(regimes):
    rows = []
    for name, keys in lineup_frames().items():
        keys = keys if isinstance(keys, list) else [keys]
        parts = []
        for k in keys:
            try:
                parts.append(load(k))
            except Exception:
                pass
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.drop_duplicates(subset=["DATE", "SLOT", "PLAYER"])
        df["REGIME"] = label_regime(df["DATE"], regimes)
        g = df.groupby(["REGIME", "DATE"]).agg(
            n=("PLAYER", "size"), n_actual=("ACTUAL_FP", lambda s: s.notna().sum()),
            actual=("ACTUAL_FP", "sum"), proj=("PROJECTED_FP", "sum"),
            zeros=("ACTUAL_FP", lambda s: (s.fillna(0) == 0).sum()),
        ).reset_index()
        complete = g[(g.n == 8) & (g.n_actual == 8)]
        for regime, cg in complete.groupby("REGIME"):
            rows.append({
                "regime": regime, "variant": name, "slates": len(cg),
                "median_actual": cg.actual.median(), "mean_actual": cg.actual.mean(),
                "mean_proj": cg.proj.mean(), "bias": (cg.proj - cg.actual).mean(),
                "zero_FP_players_per_slate": cg.zeros.mean(),
            })
    out = pd.DataFrame(rows).sort_values(["regime", "median_actual"], ascending=[True, False])
    print("\n=== LINEUPS: realized FP by regime (complete 8-player lineups with all actuals) ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("Legacy = pre-FP-model lineups from the fresh-injury window (Jan 2026)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default=DEFAULT_REGIMES)
    args = ap.parse_args()
    regimes = parse_regimes(args.regimes)
    print("Regimes:", ", ".join(f"{n} {s.date()}..{e.date()}" for n, s, e in regimes))
    minutes_report(regimes)
    lineup_report(regimes)
