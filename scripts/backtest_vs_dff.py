"""
Head-to-head with DFF over every 2025-26 regular-season slate we have DFF projections for, from the day the
early-season pause would have lifted (Oct 30) to the end of the regular season (Apr 12). Read-only.

DFF history exists only for 2025-26 (data/daily_predictions/current.parquet, Oct 24 onward; nothing older was
kept). The box scores have no playoff games, so playoff slates can't be scored.

Both sides choose from DFF's slate (every player DFF projected, with DK salary and position) and use the
production optimizer. DFF projects from its own numbers. We project from data/replay/learned_2025_26.parquet
(scripts/injury_response.py --predict-only: trained on 2022-25, pre-tip injury report as of 30 min before the
day's first tip). Our inputs come from the archived reports, not production's live feed, so the Jan-Apr feed
outage doesn't affect this test. DFF's numbers are as of our scrape (about 20 min before the main slate).

Our pool: slate players matched to a replay row that day, not listed Out/Doubtful, with a season FP-per-minute
rate (at least one game this season). Variants:
  learned min x season rate     the candidate
  learned min x learned rate
  Formula C min x season rate   control: how much comes from the learned minutes rather than the setup
DFF names are matched to box-score names per day with lambda/shared/player_names.py. Actual FP comes from box
scores through the same matching, so a name mismatch can't silently score a player as 0.

Usage: python scripts/backtest_vs_dff.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import backtest_shrinkage as bs  # noqa: E402
sys.path.insert(0, str(ROOT / "lambda" / "shared"))
from player_names import key, match_names  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)
REPLAY = ROOT / "data" / "replay"
START, END = "2025-10-30", "2026-04-12"
FEED_BROKEN = "2026-01-14"


def build_slates():
    daily = bs.load("data/daily_predictions/current.parquet")
    daily["D"] = pd.to_datetime(daily["GAME_DATE"]).dt.normalize()
    daily["GAME_DATE"] = daily["D"].dt.date
    daily = daily[(daily.D >= START) & (daily.D <= END)]
    slate = daily.drop_duplicates(["D", "PLAYER"])[["D", "PLAYER", "SALARY", "POSITION", "PPG_PROJECTION"]].copy()
    r_all = pd.read_parquet(REPLAY / "replay.parquet")
    r = r_all[r_all.SEASON == "2025-26"][["D", "PLAYER", "TEAM", "FC_MIN", "ACT_MIN", "ACT_FP", "PLAYED"]]
    known, rosters = set(r_all.PLAYER), r.groupby("D").PLAYER.apply(list)
    keys = []
    for day, g in slate.groupby("D"):
        mapping, _ = match_names(g.PLAYER, rosters.get(day, []), known)
        keys.append(g.PLAYER.map(mapping).rename("KEY"))
    slate["KEY"] = pd.concat(keys)          # box-score name, or None
    learned = pd.read_parquet(REPLAY / "learned_2025_26.parquet").drop(columns="TEAM")

    s = slate.merge(r.rename(columns={"PLAYER": "KEY"}), on=["D", "KEY"], how="left")
    s = s.merge(learned.rename(columns={"PLAYER": "KEY"}), on=["D", "KEY"], how="left")
    # Some stored slates are filed under the wrong date (e.g. Dec 24, when no games are played): drop slates where
    # most known players' teams don't play that day, then drop the remaining rows for teams not playing
    known_keys = {key(n) for n in known}
    s["KNOWN"] = s.KEY.notna() | s.PLAYER.map(key).isin(known_keys)   # a real NBA player, not a never-played two-way
    bad = s[s.KNOWN].groupby("D").apply(lambda d: 1 - d.TEAM.notna().mean())
    bad_days = bad[bad > 0.3].index
    print(f"Dropped {len(bad_days)} misdated slates: {[str(d.date()) for d in bad_days]}; "
          f"{(s.KNOWN & s.TEAM.isna() & ~s.D.isin(bad_days)).sum()} other rows for players not playing that day")
    s = s[~s.D.isin(bad_days) & ~(s.KNOWN & s.TEAM.isna())].copy()
    s["ACT"] = s.ACT_FP.fillna(0.0)
    s["MATCHED"] = s.TEAM.notna()
    s["OURS_OK"] = s.LEARNED_MIN.notna() & s.S_RATE.notna() & (s.S_RATE > 0)
    s["LxS"] = np.where(s.OURS_OK, s.LEARNED_MIN * s.S_RATE, np.nan)
    s["LxL"] = np.where(s.OURS_OK, s.LEARNED_MIN * s.LEARNED_RATE, np.nan)
    s["FCxS"] = np.where(s.OURS_OK, s.FC_MIN * s.S_RATE, np.nan)
    return daily, s


def lineup(day_rows, daily, col, minutes):
    d = day_rows[day_rows[col].notna()]
    if len(d) < 8:
        return None
    inp = d[["PLAYER", "POSITION"]].assign(TEAM=d.TEAM.fillna("UNK").values, PROJECTED_MIN=d[minutes].fillna(0).values,
                                          PROJECTED_FP=d[col].values)
    lu = bs.optimize_lineup(inp, daily, d.D.iloc[0].date())
    return d[d.PLAYER.isin(lu.PLAYER)] if len(lu) == 8 else None


def main():
    daily, s = build_slates()
    days = sorted(s.D.unique())
    unmatched = s[~s.MATCHED]
    print(f"{len(days)} slates, {len(s):,} DFF slate rows. Not matched to a box-score player: {len(unmatched)} "
          f"({unmatched.SALARY.mean():.0f} avg salary): {unmatched.PLAYER.value_counts().head(8).to_dict()}")
    print(f"No projection from us (listed Out, or no game yet this season): {(~s.OURS_OK).sum():,} rows, "
          f"of which DFF projected 15+: {((~s.OURS_OK) & (s.PPG_PROJECTION >= 15)).sum()}")

    both = s[s.OURS_OK & s.MATCHED]
    print("\n=== Player projections, same players (actual FP; did not play = 0) ===")
    for label, d in (("all", both), ("DFF projected 20+", both[both.PPG_PROJECTION >= 20]),
                     ("salary $6k+", both[both.SALARY >= 6000])):
        parts = [f"{n} MAE {(d[c] - d.ACT).abs().mean():.2f} bias {(d[c] - d.ACT).mean():+.2f}"
                 for n, c in (("DFF", "PPG_PROJECTION"), ("learned x season", "LxS"), ("learned x learned", "LxL"),
                              ("Formula C x season", "FCxS"))]
        print(f"  {label:18s} n={len(d):6,d} | " + " | ".join(parts))
    corr = both.groupby("D").apply(lambda d: pd.Series({c: d[c].corr(d.ACT, method="spearman")
                                                        for c in ("PPG_PROJECTION", "LxS", "FCxS")}))
    print(f"  per-slate rank correlation with actual: DFF {corr.PPG_PROJECTION.mean():.3f}, "
          f"learned x season {corr.LxS.mean():.3f}, Formula C x season {corr.FCxS.mean():.3f}")

    variants = {"DFF": ("PPG_PROJECTION", "LEARNED_MIN"), "learned x season": ("LxS", "LEARNED_MIN"),
                "learned x learned": ("LxL", "LEARNED_MIN"), "Formula C x season": ("FCxS", "FC_MIN")}
    rows, picks = [], []
    for day in days:
        g = s[s.D == day]
        row = {"D": day}
        for name, (col, minutes) in variants.items():
            lu = lineup(g if name == "DFF" else g[g.OURS_OK], daily, col, minutes)
            row[name] = np.nan if lu is None else lu.ACT.sum()
            if lu is not None:
                picks.append(lu.assign(VARIANT=name))
        rows.append(row)
    R = pd.DataFrame(rows).set_index("D").dropna()
    P = pd.concat(picks)

    print("\n=== Lineups (realized FP per slate) ===")
    windows = {"full season": R, f"before {FEED_BROKEN}": R[R.index < FEED_BROKEN],
               f"from {FEED_BROKEN}": R[R.index >= FEED_BROKEN]}
    for label, W in windows.items():
        print(f"{label}: {len(W)} slates, DFF {W.DFF.mean():.1f}")
        for name in list(variants)[1:]:
            m, lo, hi = bs.bootstrap(W[name] - W.DFF)
            print(f"  {name:20s} {W[name].mean():6.1f}  vs DFF {m:+5.1f} ({lo:+5.1f}..{hi:+5.1f})  "
                  f"beats DFF on {(W[name] > W.DFF).mean():.0%} of slates")
    print("\nBy month (learned x season minus DFF):")
    print((R["learned x season"] - R.DFF).groupby(R.index.to_period("M")).agg(["size", "mean"]).round(1).to_string())
    print("\nPicked players:")
    print(P.groupby("VARIANT").agg(picks=("PLAYER", "size"), actual=("ACT", "mean"), salary=("SALARY", "mean"),
                                   dnp=("ACT", lambda x: (x == 0).mean()),
                                   unmatched=("MATCHED", lambda x: 1 - x.mean())).round(3).to_string())
    R.to_parquet(REPLAY / "vs_dff_2025_26.parquet")


if __name__ == "__main__":
    main()
