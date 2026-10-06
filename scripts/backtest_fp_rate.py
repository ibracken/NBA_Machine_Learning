"""
Roadmap M1 backtest: FP-per-minute model vs the three production FP models. Read-only; writes nothing to S3.

Train every model on 2022-23..2024-25 box scores, exactly as supervised-learning prepares them
(the old models with the production +-8 minutes noise, seeded). Test on 2025-26: for each slate with
stored minutes projections, build serving features as of that night with minutes-projection's own
calculate_fp_features, feed each FP model the minutes we actually projected (complex, formula C) and
the actual minutes (oracle), and score against actual FP. Then rebuild lineups with the production
optimizer and score their realized FP (a DNP scores 0).

Also measures the two training issues: R2 on noisy actual minutes vs projected minutes, and the
production 80/20 split (deployed model never trained on the newest 20%) vs training on everything.

Usage: python scripts/backtest_fp_rate.py [--seeds 3] [--no-lineups]
"""

import argparse
import io
import logging
import sys
import warnings
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lambda" / "minutes-projection"))
from lineup_optimizer import calculate_fp_features, optimize_lineup  # noqa: E402
from config import CAREER_RATE_MAX_PRIOR_GAMES  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)
pd.set_option("display.width", 200)

BUCKET = "nba-prediction-ibracken"
TRAIN_SEASONS = ["2022-23", "2023-24", "2024-25"]
TEST_SEASON = "2025-26"
TEST_START, TEST_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2026-04-12")
FRESH_END = pd.Timestamp("2026-01-19")  # injury feed fresh until here (see evaluate_projections.py)
MINUTES_MODELS = {"complex": "complex_position_overlap", "formula_c": "formula_c_baseline"}
GBR = dict(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)

OLD_FEATURES = {
    "current": ["Last3_FP_Avg", "Last7_FP_Avg", "Season_FP_Avg", "Career_FP_Avg", "Games_Played_Career",
                "CLUSTER", "MIN", "Last7_MIN_Avg", "Season_MIN_Avg", "Career_MIN_Avg", "IS_HOME", "REST_DAYS"],
    "fp_per_min": ["Last3_FP_Avg", "Last7_FP_Avg", "Season_FP_Avg", "Career_FP_Avg", "Games_Played_Career",
                   "CLUSTER", "MIN", "Last7_MIN_Avg", "Season_MIN_Avg", "Career_MIN_Avg", "IS_HOME", "REST_DAYS",
                   "FP_PER_MIN"],
    "barebones": ["FP_PER_MIN", "MIN", "REST_DAYS", "CLUSTER"],
}
# New model: per-minute rate from history only; no MIN feature
RATE_FEATURES = ["FP_PER_MIN", "L7_RATE", "CAREER_RATE", "Last3_FP_Avg", "Last7_FP_Avg", "Season_FP_Avg",
                 "Career_FP_Avg", "Season_MIN_Avg", "Last7_MIN_Avg", "Career_MIN_Avg", "Games_Played_Career",
                 "IS_HOME", "REST_DAYS", "CLUSTER"]

s3 = boto3.client("s3")


def load(key):
    return pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()))


def season_of(d):
    return f"{d.year}-{str(d.year + 1)[-2:]}" if d.month >= 10 else f"{d.year - 1}-{str(d.year)[-2:]}"


def safe_div(a, b):
    return np.where(b > 0, a / b.where(b > 0, 1), 0)


def add_rates(df):
    df["L7_RATE"] = safe_div(df["Last7_FP_Avg"], df["Last7_MIN_Avg"])
    df["CAREER_RATE"] = safe_div(df["Career_FP_Avg"], df["Career_MIN_Avg"])
    return df


def training_frame(box):
    """supervised-learning's preprocessing, minus the minutes noise (added per model below)."""
    df = box[box["MIN"] != 0].copy()
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
    df["CLUSTER"] = df["CLUSTER"].fillna("CLUSTER_NAN")
    df["IS_HOME"] = df["MATCHUP"].astype(str).str.contains(" vs. ").astype(int)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER", "GAME_DATE"])
    df["REST_DAYS"] = (df["GAME_DATE"] - df.groupby("PLAYER")["GAME_DATE"].shift(1)).dt.days.fillna(3).clip(0, 30)
    for c in ["Last3_FP_Avg", "Last7_FP_Avg"]:
        df[c] = df[c].fillna(df["Season_FP_Avg"])
    df["Last7_MIN_Avg"] = df["Last7_MIN_Avg"].fillna(df["Season_MIN_Avg"])
    cols = ["Last3_FP_Avg", "Last7_FP_Avg", "Season_FP_Avg", "Career_FP_Avg", "Games_Played_Career",
            "Last7_MIN_Avg", "Season_MIN_Avg", "Career_MIN_Avg"]
    df[cols] = df[cols].replace([np.inf, -np.inf], 0).fillna(0)
    df["SEASON"] = df["GAME_DATE"].apply(season_of)
    df["SEASON_GAME_NUM"] = df.groupby(["PLAYER", "SEASON"]).cumcount() + 1
    df["FP_PER_MIN"] = np.where(df["SEASON_GAME_NUM"] <= 2,
                                safe_div(df["Career_FP_Avg"], df["Career_MIN_Avg"]),
                                safe_div(df["Season_FP_Avg"], df["Season_MIN_Avg"]))
    return add_rates(df)


def encode(df, features, columns=None):
    X = pd.get_dummies(df[features], columns=["CLUSTER"]) if "CLUSTER" in features else df[features].copy()
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X.astype(float)


def train_old(train, name, seed):
    df = train.copy()
    rng = np.random.default_rng(seed)
    df["MIN"] = (df["MIN"] + rng.uniform(-8, 8, size=len(df))).clip(lower=0)
    X = encode(df, OLD_FEATURES[name])
    return GradientBoostingRegressor(**GBR).fit(X, df["FP"]), list(X.columns)


def train_rate(train):
    df = train[train["MIN"] > 0].copy()
    X = encode(df, RATE_FEATURES)
    model = GradientBoostingRegressor(**GBR).fit(X, df["FP"] / df["MIN"], sample_weight=df["MIN"])
    return model, list(X.columns)


def base_name(name):
    """'barebones@80%' / 'current#2' -> the feature set they were trained with."""
    return name.split("@")[0].split("#")[0]


def serving_frame(features, players, home):
    """Features as of the slate the way minutes-projection builds them (calculate_fp_features + fills)."""
    df = players.merge(features, on="PLAYER", how="left")
    df["SEASON_GAMES_PLAYED"] = df["SEASON_GAMES_PLAYED"].fillna(0).astype(int)
    df["FP_PER_MIN"] = np.where(df["SEASON_GAMES_PLAYED"] <= CAREER_RATE_MAX_PRIOR_GAMES,
                                safe_div(df["Career_FP_Avg"].fillna(0), df["Career_MIN_Avg"].fillna(0)),
                                safe_div(df["Season_FP_Avg"].fillna(0), df["Season_MIN_Avg"].fillna(0)))
    for c in ["Last3_FP_Avg", "Last7_FP_Avg", "Season_FP_Avg", "Career_FP_Avg", "Games_Played_Career",
              "Last7_MIN_Avg", "Season_MIN_Avg", "Career_MIN_Avg"]:
        df[c] = df[c].fillna(0)
    df["IS_HOME"] = df["TEAM"].map(home).fillna(0).astype(int)
    df["REST_DAYS"] = df["REST_DAYS"].fillna(3).clip(0, 30)
    df["CLUSTER"] = df["CLUSTER"].fillna("CLUSTER_NAN")
    df["NO_HISTORY"] = (df["Season_FP_Avg"] == 0) & (df["Career_FP_Avg"] == 0)
    return add_rates(df)


def predict(models, df, minutes):
    out = {}
    for name, (model, cols) in models.items():
        X_df = df.assign(MIN=minutes.fillna(0))
        if base_name(name) == "rate":
            p = model.predict(encode(X_df, RATE_FEATURES, cols)).clip(min=0) * minutes.fillna(0).values
        else:
            p = model.predict(encode(X_df, OLD_FEATURES[base_name(name)], cols))
        out[name] = np.where(df["NO_HISTORY"], 0, np.round(p, 1))
    return out


def bootstrap_ci(per_slate_diff, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(per_slate_diff)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="noise seeds for the old models")
    ap.add_argument("--no-lineups", action="store_true")
    args = ap.parse_args()

    print("Loading box scores...")
    box = pd.concat([load(f"data/box_scores/{s}.parquet") for s in TRAIN_SEASONS + [TEST_SEASON]], ignore_index=True)
    box["GAME_DATE"] = pd.to_datetime(box["GAME_DATE"])
    frame = training_frame(box)
    train = frame[frame["GAME_DATE"] < TEST_START]
    test_rows = frame[(frame["GAME_DATE"] >= TEST_START) & (frame["GAME_DATE"] <= TEST_END)]
    print(f"train rows {len(train)} ({train.GAME_DATE.min().date()}..{train.GAME_DATE.max().date()}), "
          f"test rows {len(test_rows)}")

    # ---- Issue 2: R2 on noisy actual minutes (what training reports) vs actual minutes, per model
    print("\nTraining models...")
    seeds = list(range(args.seeds))
    old_by_seed = {}
    for s in seeds:
        old_by_seed[s] = {}
        for n in OLD_FEATURES:
            old_by_seed[s][n] = train_old(train, n, s)
            print(f"  trained {n} (noise seed {s})", flush=True)
    rate = train_rate(train)
    print("  trained rate", flush=True)
    models = {**old_by_seed[0], "rate": rate}

    rng = np.random.default_rng(99)
    noisy = test_rows["MIN"] + rng.uniform(-8, 8, size=len(test_rows))
    print("\n=== Test R2 on 2025-26 box-score rows (actual MIN / production-style noisy MIN) ===")
    for name, (m, cols) in models.items():
        r2s = []
        for mins in (test_rows["MIN"], noisy.clip(lower=0)):
            X = test_rows.assign(MIN=mins)
            p = (m.predict(encode(X, RATE_FEATURES, cols)).clip(min=0) * mins.values if name == "rate"
                 else m.predict(encode(X, OLD_FEATURES[name], cols)))
            r2s.append(r2_score(test_rows["FP"], p))
        print(f"  {name:11s} R2 actual MIN {r2s[0]:.3f} | noisy MIN {r2s[1]:.3f}")

    # ---- Issue 1: production 80/20 split vs all training data (barebones and rate)
    cut = train["GAME_DATE"].sort_values().iloc[int(len(train) * 0.8)]
    split_models = {"barebones@80%": train_old(train[train.GAME_DATE < cut], "barebones", 0),
                    "rate@80%": train_rate(train[train.GAME_DATE < cut])}
    print(f"\nProduction-style 80/20 cutoff on this training window: {cut.date()}")

    # ---- Slates
    daily = load("data/daily_predictions/current.parquet")
    daily["GAME_DATE"] = pd.to_datetime(daily["GAME_DATE"]).dt.date
    box_test = box[box["GAME_DATE"] >= TEST_START].copy()
    actual = box_test.groupby(["GAME_DATE", "PLAYER"]).agg(ACT_FP=("FP", "sum"), ACT_MIN=("MIN", "sum")).reset_index()
    home_by_day = {d: dict(zip(g["TEAM_ABBREVIATION"], g["MATCHUP"].str.contains(" vs. ").astype(int)))
                   for d, g in box_test.groupby("GAME_DATE")}

    proj = {}
    for key, folder in MINUTES_MODELS.items():
        p = load(f"model_comparison/{folder}/minutes_projections.parquet")
        p["DATE"] = pd.to_datetime(p["DATE"]).dt.normalize()
        proj[key] = p[(p.DATE >= TEST_START) & (p.DATE <= TEST_END) & p.DATE.isin(home_by_day.keys())]
    days = sorted(set(proj["complex"].DATE) & set(proj["formula_c"].DATE))
    print(f"Slates with stored projections and box scores: {len(days)} ({days[0].date()}..{days[-1].date()})")

    player_rows, lineup_rows = [], []
    seed_variants = {f"{n}#{s}": old_by_seed[s][n] for s in seeds[1:] for n in OLD_FEATURES}
    all_models = {**models, **split_models, **seed_variants}
    for i, day in enumerate(days):
        acts = actual[actual.GAME_DATE == day].set_index("PLAYER")
        features = calculate_fp_features(box_test[box_test["GAME_DATE"] < day], day.date()).drop_duplicates("PLAYER")
        for mkey, p in proj.items():
            players = p[(p.DATE == day) & (p.PROJECTED_MIN > 0)][["PLAYER", "TEAM", "POSITION", "PROJECTED_MIN"]]
            players = players.drop_duplicates("PLAYER").reset_index(drop=True)
            df = serving_frame(features, players, home_by_day[day])
            df["ACT_FP"] = df["PLAYER"].map(acts["ACT_FP"])
            df["ACT_MIN"] = df["PLAYER"].map(acts["ACT_MIN"])
            for msrc, minutes in (("projected", df["PROJECTED_MIN"]), ("actual", df["ACT_MIN"])):
                preds = predict(all_models, df, minutes)
                played = df["ACT_MIN"].fillna(0) > 0
                for name, pr in preds.items():
                    player_rows.append(pd.DataFrame({"DATE": day, "MINUTES_MODEL": mkey, "MIN_SOURCE": msrc,
                                                     "FP_MODEL": name, "PRED": pr[played], "ACT": df.loc[played, "ACT_FP"]}))
                    if msrc != "projected" or args.no_lineups or "@" in name or "#" in name:
                        continue
                    lin_in = df[["PLAYER", "TEAM", "PROJECTED_MIN"]].assign(PROJECTED_FP=pr)
                    lin_in = lin_in.merge(daily[daily.GAME_DATE == day.date()][["PLAYER", "POSITION"]]
                                          .drop_duplicates("PLAYER"), on="PLAYER", how="left")
                    lineup = optimize_lineup(lin_in, daily, day.date())
                    if len(lineup) == 8:
                        realized = lineup["PLAYER"].map(acts["ACT_FP"]).fillna(0)
                        lineup_rows.append({"DATE": day, "MINUTES_MODEL": mkey, "FP_MODEL": name,
                                            "PROJ": lineup["PROJECTED_FP"].sum(), "REAL": realized.sum(),
                                            "DNP": int(lineup["PLAYER"].map(acts["ACT_MIN"]).fillna(0).eq(0).sum())})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(days)} slates")

    pr = pd.concat(player_rows, ignore_index=True)
    pr["ERR"] = pr["PRED"] - pr["ACT"]
    pr["WINDOW"] = np.where(pr["DATE"] <= FRESH_END, "fresh", "stale")

    print("\n=== Player FP error (players who played; seed-0 models) ===")
    t = pr[~pr.FP_MODEL.str.contains("#")].groupby(["MIN_SOURCE", "MINUTES_MODEL", "FP_MODEL"]).agg(
        n=("ERR", "size"), MAE=("ERR", lambda e: e.abs().mean()), bias=("ERR", "mean")).round(3)
    print(t.to_string())

    print("\n=== Decision: rate vs best old model, projected minutes, per-slate MAE difference (negative = rate better) ===")
    proj_only = pr[pr.MIN_SOURCE == "projected"]
    for mkey in MINUTES_MODELS:
        g = proj_only[proj_only.MINUTES_MODEL == mkey]
        slate_mae = g.groupby(["DATE", "FP_MODEL"])["ERR"].apply(lambda e: e.abs().mean()).unstack()
        best_old = min(OLD_FEATURES, key=lambda n: slate_mae[n].mean())
        mean, lo, hi = bootstrap_ci(slate_mae["rate"] - slate_mae[best_old])
        print(f"  {mkey:9s} best old = {best_old:10s} diff {mean:+.3f} FP  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"({len(slate_mae)} slates)")
        for w in ("fresh", "stale"):
            gw = g[g.WINDOW == w]
            sm = gw.groupby(["DATE", "FP_MODEL"])["ERR"].apply(lambda e: e.abs().mean()).unstack()
            if len(sm) > 1:
                m_, l_, h_ = bootstrap_ci(sm["rate"] - sm[best_old])
                print(f"      {w:5s} window: diff {m_:+.3f} [{l_:+.3f}, {h_:+.3f}] ({len(sm)} slates)")

    print("\n=== Issue 1: trained on first 80% (production) vs all training data, projected minutes ===")
    for a, b in (("barebones@80%", "barebones"), ("rate@80%", "rate")):
        g = proj_only.groupby(["DATE", "MINUTES_MODEL", "FP_MODEL"])["ERR"].apply(lambda e: e.abs().mean()).unstack()
        mean, lo, hi = bootstrap_ci(g[b] - g[a])
        print(f"  {b:9s} all-data minus 80%-only: {mean:+.3f} FP MAE  95% CI [{lo:+.3f}, {hi:+.3f}]")

    if len(seeds) > 1:
        print("\n=== Old models across minutes-noise seeds (projected minutes): MAE; rate shown for reference ===")
        g = proj_only.copy()
        g["BASE"] = g["FP_MODEL"].map(base_name)
        g["SEED"] = g["FP_MODEL"].str.extract(r"#(\d+)")[0].fillna("0")
        g = g[~g["FP_MODEL"].str.contains("@")]
        print(g.groupby(["MINUTES_MODEL", "BASE", "SEED"])["ERR"].apply(lambda e: e.abs().mean())
              .unstack("SEED").round(3).to_string())

    if lineup_rows:
        lr = pd.DataFrame(lineup_rows)
        print("\n=== Lineups rebuilt with the production optimizer (realized FP; DNP = 0) ===")
        s = lr.groupby(["MINUTES_MODEL", "FP_MODEL"]).agg(slates=("REAL", "size"), median_real=("REAL", "median"),
                                                         mean_real=("REAL", "mean"), mean_proj=("PROJ", "mean"),
                                                         dnp_per_slate=("DNP", "mean")).round(1)
        print(s.to_string())
        for mkey in MINUTES_MODELS:
            w = lr[lr.MINUTES_MODEL == mkey].pivot(index="DATE", columns="FP_MODEL", values="REAL").dropna()
            best_old = max(OLD_FEATURES, key=lambda n: w[n].mean())
            mean, lo, hi = bootstrap_ci(w["rate"] - w[best_old])
            print(f"  {mkey:9s} rate minus best old ({best_old}): {mean:+.1f} realized FP/slate  "
                  f"95% CI [{lo:+.1f}, {hi:+.1f}]  ({len(w)} slates)")


if __name__ == "__main__":
    main()
