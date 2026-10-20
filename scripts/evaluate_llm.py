"""
Score the LLM components against actuals.

L2 (shadow-mode adjustments): per judge model, did adjusting beat not adjusting on the adjusted players?
    Go/no-go: at least 30 slates with valid adjustments AND the 95% bootstrap CI of (post - pre) absolute
    error is entirely below 0. Anything else means it does not ship.
    Precision (share of adjustments a human endorses): --export-review writes unlabeled valid adjustments
    to a CSV with an empty ENDORSED column; fill y/n and pass it back with --labels.

AI lineup (replaced the L3 head-to-head minutes, 2026-10-18): realized FP of the AI-picked lineup vs our
    learned-minutes lineup and DFF's, on the same graded slates, with bootstrap CIs.

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


def ai_lineup_report():
    """AI lineup vs our learned-minutes lineup and DFF's, on the slates where all three are complete and graded."""
    print("\n=== AI lineup ===")
    keys = {"ai_lineup": "model_comparison/ai_lineup/daily_lineups.parquet",
            "learned_minutes": "model_comparison/learned_minutes/daily_lineups.parquet",
            "DFF": "model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet"}
    totals = {}
    for name, key in keys.items():
        df = load(key)
        if df.empty:
            print(f"no {name} lineups yet")
            return
        df["DATE"] = pd.to_datetime(df["DATE"])
        g = df.groupby("DATE").agg(n=("PLAYER", "size"), n_actual=("ACTUAL_FP", lambda s: s.notna().sum()),
                                   actual=("ACTUAL_FP", "sum"))
        totals[name] = g[(g["n"] == 8) & (g["n_actual"] == 8)]["actual"]
    common = totals["ai_lineup"].index.intersection(totals["learned_minutes"].index).intersection(totals["DFF"].index)
    if common.empty:
        print("no slate with all three lineups graded yet")
        return
    print(f"{len(common)} slates; realized FP: " + ", ".join(f"{k} {v[common].mean():.1f}" for k, v in totals.items()))
    for other in ("learned_minutes", "DFF"):
        diff = totals["ai_lineup"][common] - totals[other][common]
        lo, hi = bootstrap_ci(diff.values, diff.index)
        print(f"  AI lineup minus {other}: {diff.mean():+.1f} per slate (95% CI {lo:+.1f}..{hi:+.1f}), "
              f"AI ahead on {(diff > 0).mean():.0%} of slates")
    if len(common) < MIN_SLATES_TO_SHIP:
        print(f"  fewer than {MIN_SLATES_TO_SHIP} slates: too early to decide")
    print("  kill rule (roadmap): drop it if it doesn't beat our lineup after 30+ slates")


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
    ai_lineup_report()
    usage_report()
