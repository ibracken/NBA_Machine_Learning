"""
Score the LLM components against actuals.

L2 (shadow-mode adjustments): per judge model, did adjusting beat not adjusting on the adjusted players?
    Go/no-go: at least 30 slates with valid adjustments AND the 95% bootstrap CI of (post - pre) absolute
    error is entirely below 0. Anything else means it does not ship.
    Precision (share of adjustments a human endorses): --export-review writes unlabeled valid adjustments
    to a CSV with an empty ENDORSED column; fill y/n and pass it back with --labels.

L3 (head-to-head): LLM minutes vs complex_position_overlap and formula_c_baseline on the same player-dates,
    overall and split by whether the player was mentioned in that night's research briefing ("with news").
    Realized lineup FP vs our lineups and DFF on the same slates.

Usage: python scripts/evaluate_llm.py [--export-review review.csv] [--labels review.csv]
"""

import argparse
import io
import json

import boto3
import numpy as np
import pandas as pd

BUCKET = "nba-prediction-ibracken"
MIN_SLATES_TO_SHIP = 30
s3 = boto3.client("s3")


def load(key):
    try:
        return pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()))
    except s3.exceptions.NoSuchKey:
        return pd.DataFrame()


def load_json(key):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None


def actual_minutes():
    box = load("data/box_scores/current.parquet")
    box["DATE"] = pd.to_datetime(box["GAME_DATE"])
    return box[["DATE", "PLAYER", "MIN"]].rename(columns={"MIN": "ACTUAL_MIN"})


def bootstrap_ci(values, groups, n=2000):
    """95% CI of the mean, resampling whole slates (errors within a slate are correlated)."""
    df = pd.DataFrame({"v": values, "g": groups})
    by_slate = df.groupby("g")["v"].agg(["sum", "size"])
    rng = np.random.default_rng(0)
    means = []
    for _ in range(n):
        pick = by_slate.iloc[rng.integers(0, len(by_slate), len(by_slate))]
        means.append(pick["sum"].sum() / pick["size"].sum())
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def l2_report(export_review, labels_path):
    log = load("llm/adjustments/log.parquet")
    print("\n=== L2: shadow-mode adjustments ===")
    if log.empty:
        print("no adjustments logged yet")
        return
    log["DATE"] = pd.to_datetime(log["DATE"])
    log = log.merge(actual_minutes(), on=["DATE", "PLAYER"], how="left")

    for judge, g in log.groupby("JUDGE_MODEL"):
        valid = g[g["VALID"]]
        print(f"\n-- judge {judge}: {len(g)} proposed over {g['DATE'].nunique()} slates, {len(valid)} valid, "
              f"{int(g['CLAMPED'].sum())} clamped")
        print("   rejected:", g.loc[~g["VALID"], "REJECT_REASON"].value_counts().to_dict())
        scored = valid.dropna(subset=["ACTUAL_MIN"])
        scored = scored[scored["ACTUAL_MIN"] > 0]
        dnp = valid["ACTUAL_MIN"].isna().sum()
        if scored.empty:
            print(f"   no adjusted players with actuals yet (DNP/unknown: {dnp})")
            continue
        pre_err = (scored["PRE_MIN"] - scored["ACTUAL_MIN"]).abs()
        post_err = (scored["POST_MIN"] - scored["ACTUAL_MIN"]).abs()
        diff = post_err - pre_err
        lo, hi = bootstrap_ci(diff.values, scored["DATE"].values)
        direction = np.sign(scored["POST_MIN"] - scored["PRE_MIN"]) == np.sign(scored["ACTUAL_MIN"] - scored["PRE_MIN"])
        slates = scored["DATE"].nunique()
        print(f"   scored {len(scored)} player-games on {slates} slates (DNP/unknown excluded: {dnp})")
        print(f"   MAE pre {pre_err.mean():.3f} -> post {post_err.mean():.3f}  "
              f"(diff {diff.mean():+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}])")
        print(f"   bias pre {(scored['PRE_MIN'] - scored['ACTUAL_MIN']).mean():+.2f} -> "
              f"post {(scored['POST_MIN'] - scored['ACTUAL_MIN']).mean():+.2f}; "
              f"moved in the right direction {direction.mean():.0%}")
        verdict = ("SHIP CANDIDATE" if slates >= MIN_SLATES_TO_SHIP and hi < 0 else
                   f"DO NOT SHIP ({'needs ' + str(MIN_SLATES_TO_SHIP) + '+ slates' if slates < MIN_SLATES_TO_SHIP else 'does not beat its absence'})")
        print(f"   verdict: {verdict}")

    if labels_path:
        labels = pd.read_csv(labels_path)
        labels = labels[labels["ENDORSED"].astype(str).str.lower().isin(["y", "n"])]
        if not labels.empty:
            labels["ENDORSED"] = labels["ENDORSED"].str.lower() == "y"
            print("\n   human-endorsed precision:",
                  labels.groupby("JUDGE_MODEL")["ENDORSED"].agg(["mean", "size"]).round(3).to_dict("index"))
    if export_review:
        cols = ["DATE", "JUDGE_MODEL", "PLAYER", "TEAM", "PRE_MIN", "POST_MIN", "PCT_APPLIED", "CONFIDENCE",
                "REASON", "QUOTE", "SOURCE_URL"]
        review = log[log["VALID"]][cols].copy()
        review["ENDORSED"] = ""
        review.to_csv(export_review, index=False)
        print(f"\n   wrote {len(review)} valid adjustments to {export_review}; fill ENDORSED with y/n")


def l3_report():
    print("\n=== L3: LLM head-to-head ===")
    h2h = load("model_comparison/llm_head_to_head/minutes_projections.parquet")
    if h2h.empty:
        print("no head-to-head projections yet")
        return
    frames = {"llm_head_to_head": h2h}
    for m in ["complex_position_overlap", "formula_c_baseline"]:
        frames[m] = load(f"model_comparison/{m}/minutes_projections.parquet")
    for m, df in frames.items():
        df["DATE"] = pd.to_datetime(df["DATE"])
    dates = sorted(h2h["DATE"].unique())

    joined = h2h[["DATE", "PLAYER", "PROJECTED_MIN", "LLM_STATUS"]].rename(columns={"PROJECTED_MIN": "llm"})
    for m in ["complex_position_overlap", "formula_c_baseline"]:
        joined = joined.merge(frames[m][["DATE", "PLAYER", "PROJECTED_MIN"]].rename(columns={"PROJECTED_MIN": m}),
                              on=["DATE", "PLAYER"], how="inner")
    joined = joined.merge(actual_minutes(), on=["DATE", "PLAYER"], how="inner")
    joined = joined[joined["ACTUAL_MIN"] > 0]

    # "With news": named in that night's research briefing
    mentioned = set()
    for d in dates:
        research = load_json(f"llm/research/{pd.Timestamp(d).date()}.json") or {}
        text = " ".join(g.get("briefing", "") for g in research.get("games", [])).lower()
        mentioned |= {(pd.Timestamp(d), p) for p in joined.loc[joined["DATE"] == d, "PLAYER"] if p in text}
    joined["WITH_NEWS"] = [(d, p) in mentioned for d, p in zip(joined["DATE"], joined["PLAYER"])]

    print(f"player-games scored (all three models, played): {len(joined)} over {joined['DATE'].nunique()} slates")
    for subset, g in [("all", joined), ("with news", joined[joined["WITH_NEWS"]]),
                      ("no news", joined[~joined["WITH_NEWS"]])]:
        if g.empty:
            continue
        stats = {m: f"MAE {(g[m] - g['ACTUAL_MIN']).abs().mean():.3f} bias {(g[m] - g['ACTUAL_MIN']).mean():+.2f}"
                 for m in ["llm", "complex_position_overlap", "formula_c_baseline"]}
        print(f"  {subset:9} (n={len(g)}): " + " | ".join(f"{k}: {v}" for k, v in stats.items()))
    print("  prior stated in the roadmap: LLM loses on aggregate MAE and wins on the with-news subset")

    rows = []
    variants = {f"{m}|{fp}": f"model_comparison/{m}/fp_{fp}/daily_lineups.parquet"
                for m in ["llm_head_to_head", "complex_position_overlap", "formula_c_baseline"]
                for fp in ["current", "fp_per_min", "barebones"]}
    variants["DFF"] = "model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet"
    llm_dates = set()
    lineup_frames = {}
    for name, key in variants.items():
        df = load(key)
        if df.empty:
            continue
        df["DATE"] = pd.to_datetime(df["DATE"])
        g = df.groupby("DATE").agg(n=("PLAYER", "size"), n_actual=("ACTUAL_FP", lambda s: s.notna().sum()),
                                   actual=("ACTUAL_FP", "sum"), proj=("PROJECTED_FP", "sum"))
        g = g[(g["n"] == 8) & (g["n_actual"] == 8)]
        lineup_frames[name] = g
        if name.startswith("llm_head_to_head"):
            llm_dates |= set(g.index)
    for name, g in lineup_frames.items():
        g = g[g.index.isin(llm_dates)]
        if not g.empty:
            rows.append({"variant": name, "slates": len(g), "mean_actual": g["actual"].mean(),
                         "mean_proj": g["proj"].mean(), "bias": (g["proj"] - g["actual"]).mean()})
    if rows:
        print("\n  realized lineup FP on slates with a complete LLM lineup:")
        print(pd.DataFrame(rows).sort_values("mean_actual", ascending=False).to_string(
            index=False, float_format=lambda x: f"{x:.1f}"))


def usage_report():
    print("\n=== LLM spend (token cost; web searches billed separately) ===")
    paginator = s3.get_paginator("list_objects_v2")
    rows = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix="llm/usage/"):
        for obj in page.get("Contents", []):
            day = load_json(obj["Key"])
            for run in day["runs"]:
                rows.append({"date": obj["Key"].split("/")[-1][:-5], "action": run["action"],
                             "cost_usd": run["cost_usd"], "web_searches": run["web_search_requests"]})
    if not rows:
        print("no usage recorded yet")
        return
    df = pd.DataFrame(rows)
    print(df.groupby("action")[["cost_usd", "web_searches"]].agg(["sum", "mean"]).round(3).to_string())
    print(f"total ${df['cost_usd'].sum():.2f} over {df['date'].nunique()} days")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-review", help="write valid L2 adjustments to this CSV for human labeling")
    ap.add_argument("--labels", help="labeled review CSV (ENDORSED = y/n) to compute precision")
    args = ap.parse_args()
    l2_report(args.export_review, args.labels)
    l3_report()
    usage_report()
