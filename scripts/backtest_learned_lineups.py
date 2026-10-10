"""
Do the learned minutes / per-minute models build better lineups? 2025-26 slates, production optimizer. Read-only.

Pool: the production pool (data/replay/pool_2025_26.parquet, scripts/compare_lineups_dff.py). Learned predictions
come from scripts/injury_response.py (trained on 2022-25, pre-tip injury report out set). Variants:
  production            barebones FP on the complex model's stored minutes (what ran)
  production, report    same, minus players listed Out on the pre-tip injury report (2025-26's feed was stale;
                        fixed since) - the fair baseline
  barebones x learned   barebones FP regenerated on learned minutes
  learned x learned     learned minutes x learned FP per minute
  learned x season      learned minutes x season FP per minute (LEARNED_MIN: with report-status inputs)
  learned v1 x season   same with the first learned minutes model (no status inputs)
All variants except "production" drop listed-Out players. Players without a learned projection (no game yet this
season) keep the production projection. DFF's own lineup is the yardstick. Also scored on the slates before production's injury feed broke (Jan 14, 2026),
where the complex model's inputs were sound apart from the injury-name bugs fixed 2026-10-09. Per-slate realized FP
is saved to data/replay/learned_lineups_2025_26.parquet.

Usage: python scripts/backtest_learned_lineups.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import backtest_shrinkage as bs  # noqa: E402

logging.getLogger().setLevel(logging.ERROR)
REPLAY = ROOT / "data" / "replay"
FEED_BROKEN = "2026-01-14"   # production's injury feed timed out from here on (2026 outage)


def dff_lineups(daily, box):
    slate = daily.drop_duplicates(["D", "PLAYER"])
    team = box.drop_duplicates(["D", "PLAYER"]).set_index(["D", "PLAYER"])["TEAM_ABBREVIATION"]
    act = box.groupby(["D", "PLAYER"])["FP"].sum()
    rows = []
    for day, s in slate.groupby("D"):
        s = s.assign(TEAM=[team.get((day, p), "UNK") for p in s.PLAYER])
        lu = bs.optimize_lineup(s[["PLAYER", "TEAM", "POSITION"]].assign(PROJECTED_MIN=0, PROJECTED_FP=s.PPG_PROJECTION.values),
                                daily, day.date())
        if len(lu) == 8:
            rows.append({"D": day, "REAL": sum(act.get((day, p), 0.0) for p in lu.PLAYER),
                         "PLAYERS": tuple(sorted(lu.PLAYER))})
    return pd.DataFrame(rows).set_index("D")


def main():
    pool = pd.read_parquet(REPLAY / "pool_2025_26.parquet")
    learned = pd.read_parquet(REPLAY / "learned_2025_26.parquet")
    reports = pd.read_parquet(REPLAY / "injury_reports.parquet")
    out = reports[reports.STATUS.isin(["Out", "Doubtful"]) & ~reports.GLEAGUE]
    listed = set(zip(out.D, out.PLAYER))
    pool["LISTED_OUT"] = [(d, p) in listed for d, p in zip(pool.D, pool.PLAYER)]
    pool = pool.merge(learned[["D", "PLAYER", "LEARNED_MIN", "LEARNED_MIN_V1", "LEARNED_RATE", "S_RATE"]], on=["D", "PLAYER"], how="left")
    print(f"Pool {len(pool):,} rows; listed Out pre-tip {pool.LISTED_OUT.sum()} "
          f"(of whom played {pool[pool.LISTED_OUT].PLAYED.fillna(False).sum()}); "
          f"no learned projection {pool.LEARNED_MIN.isna().sum()}")

    daily = bs.load("data/daily_predictions/current.parquet")
    daily["GAME_DATE"] = pd.to_datetime(daily["GAME_DATE"]).dt.date
    daily["D"] = pd.to_datetime(daily["GAME_DATE"])
    box = bs.load("data/box_scores/2025-26.parquet")
    box["D"] = pd.to_datetime(box["GAME_DATE"]).dt.normalize()

    fair = pool[~pool.LISTED_OUT].copy()
    has = fair.LEARNED_MIN.notna()
    regen = fair.assign(PROJECTED_MIN=np.where(has, fair.LEARNED_MIN, fair.PROJECTED_MIN))
    fair["BB_LEARNED"] = bs.regenerate_fp(regen, box).values
    fair["LxL"] = np.where(has, fair.LEARNED_MIN * fair.LEARNED_RATE, fair.FP_HAT)
    fair["LxS"] = np.where(has & fair.S_RATE.notna(), fair.LEARNED_MIN * fair.S_RATE, fair.FP_HAT)
    fair["V1xS"] = np.where(has & fair.S_RATE.notna(), fair.LEARNED_MIN_V1 * fair.S_RATE, fair.FP_HAT)
    v1_min = np.where(has, fair.LEARNED_MIN_V1, fair.PROJECTED_MIN)

    variants = {
        "production": (pool, "FP_HAT", "PROJECTED_MIN"),
        "production, report": (fair, "FP_HAT", "PROJECTED_MIN"),
        "barebones x learned": (fair.assign(PROJECTED_MIN=regen.PROJECTED_MIN.values), "BB_LEARNED", "PROJECTED_MIN"),
        "learned x learned": (fair.assign(PROJECTED_MIN=regen.PROJECTED_MIN.values), "LxL", "PROJECTED_MIN"),
        "learned v1 x season": (fair.assign(PROJECTED_MIN=v1_min), "V1xS", "PROJECTED_MIN"),
        "learned x season": (fair.assign(PROJECTED_MIN=regen.PROJECTED_MIN.values), "LxS", "PROJECTED_MIN"),
    }
    res = {name: bs.lineups(df, daily, col).set_index("D") for name, (df, col, _) in variants.items()}
    dff = dff_lineups(daily, box)

    base = res["production, report"]
    common = dff.index
    for lu in res.values():
        common = common.intersection(lu.index)
    windows = {"full season": common,
               f"injury feed working (before {FEED_BROKEN})": common[common < pd.Timestamp(FEED_BROKEN)]}
    for label, days in windows.items():
        print(f"\n{label}: {len(days)} slates. DFF lineup realized {dff.loc[days, 'REAL'].mean():.1f}")
        print(f"{'variant':22s} realized  vs fair baseline (95% CI)   gap to DFF   player MAE  slates beating DFF")
        for name, lu in res.items():
            df, col, _ = variants[name]
            d = df[df.D.isin(days)]
            e = (d[col] - d.ACT).abs().mean()
            diff = lu.loc[days, "REAL"] - base.loc[days, "REAL"]
            m, lo, hi = bs.bootstrap(diff)
            gap = (dff.loc[days, "REAL"] - lu.loc[days, "REAL"])
            print(f"{name:22s} {lu.loc[days, 'REAL'].mean():7.1f}   {m:+5.1f} ({lo:+5.1f}..{hi:+5.1f})        "
                  f"{gap.mean():+5.1f}      {e:6.3f}      {(gap < 0).mean():.0%}")
    fair.to_parquet(REPLAY / "learned_pool_2025_26.parquet", index=False)
    slates = pd.concat({name: lu.REAL for name, lu in res.items()} | {"DFF": dff.REAL}, axis=1)
    slates.to_parquet(REPLAY / "learned_lineups_2025_26.parquet")


if __name__ == "__main__":
    main()
