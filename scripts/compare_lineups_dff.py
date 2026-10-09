"""
Where do our lineups lose points to DFF's? Slate-by-slate comparison over 2025-26. Read-only.

Our lineup: production pool (players complex projected > 0 and on the DFF slate), barebones FP regenerated as in
scripts/backtest_shrinkage.py. DFF's lineup: the full DFF slate, optimized on DFF's own projection. Both use the
production optimizer. Players in both lineups cancel, so (DFF-only actual FP) - (ours-only actual FP) is exactly the
per-slate points gap. DFF is used only as a yardstick here; nothing from it feeds our projections.

Caches the regenerated pool at data/replay/pool_2025_26.parquet (delete it to rebuild).
Usage: python scripts/compare_lineups_dff.py
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
CACHE = ROOT / "data" / "replay" / "pool_2025_26.parquet"


def pool_cached():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    pool = bs.build_pool()
    pool.to_parquet(CACHE, index=False)
    return pool


def main():
    pool = pool_cached()
    daily = bs.load("data/daily_predictions/current.parquet")
    daily["GAME_DATE"] = pd.to_datetime(daily["GAME_DATE"]).dt.date
    daily["D"] = pd.to_datetime(daily["GAME_DATE"])
    slate = daily.drop_duplicates(["D", "PLAYER"])
    box = bs.load("data/box_scores/2025-26.parquet")
    box["D"] = pd.to_datetime(box["GAME_DATE"]).dt.normalize()
    act = box.groupby(["D", "PLAYER"]).agg(ACT=("FP", "sum"), ACT_MIN=("MIN", "sum"))
    team = box.drop_duplicates(["D", "PLAYER"]).set_index(["D", "PLAYER"])["TEAM_ABBREVIATION"]

    rows, picks = [], []
    for day in sorted(pool.D.unique()):
        ours_in = pool[pool.D == day]
        ours = bs.optimize_lineup(ours_in[["PLAYER", "TEAM", "POSITION", "PROJECTED_MIN"]]
                                  .assign(PROJECTED_FP=ours_in.FP_HAT.values), daily, day.date())
        s = slate[slate.D == day].copy()
        s["TEAM"] = [team.get((day, p), "UNK") for p in s.PLAYER]
        dff = bs.optimize_lineup(s[["PLAYER", "TEAM", "POSITION"]].assign(PROJECTED_MIN=0,
                                 PROJECTED_FP=s.PPG_PROJECTION.values), daily, day.date())
        if len(ours) != 8 or len(dff) != 8:
            continue
        o, f = set(ours.PLAYER), set(dff.PLAYER)
        real = lambda ps: sum(act["ACT"].get((day, p), 0.0) for p in ps)
        rows.append({"D": day, "OURS": real(o), "DFF": real(f), "SHARED": len(o & f)})
        info = ours_in.set_index("PLAYER")
        dffp = s.set_index("PLAYER")["PPG_PROJECTION"]
        for side, ps in (("ours only", o - f), ("DFF only", f - o)):
            for p in ps:
                inpool = p in info.index
                picks.append({
                    "D": day, "SIDE": side, "PLAYER": p, "IN_OUR_POOL": inpool,
                    "SALARY": s.set_index("PLAYER").SALARY.get(p),
                    "OUR_FP": info.FP_HAT.get(p) if inpool else np.nan,
                    "OUR_MIN": info.PROJECTED_MIN.get(p) if inpool else np.nan,
                    "DFF_FP": dffp.get(p), "ACT": act["ACT"].get((day, p), 0.0),
                    "ACT_MIN": act["ACT_MIN"].get((day, p), 0.0),
                    "GROUP": info.GROUP.get(p) if inpool else "not in our pool"})
    R, P = pd.DataFrame(rows), pd.DataFrame(picks)
    gap = R.DFF - R.OURS
    v = np.random.default_rng(0).choice(gap.values, (4000, len(gap))).mean(axis=1)
    print(f"{len(R)} slates. Realized: ours {R.OURS.mean():.1f}, DFF {R.DFF.mean():.1f}; gap {gap.mean():+.1f} per slate "
          f"(95% CI {np.percentile(v, 2.5):+.1f}..{np.percentile(v, 97.5):+.1f}); shared players {R.SHARED.mean():.1f}/8")
    print(f"DFF wins {(gap > 0).mean():.0%} of slates\n")

    P["OUR_ERR"] = P.OUR_FP - P.ACT
    P["DFF_ERR"] = P.DFF_FP - P.ACT
    P["OUR_RATE"] = P.OUR_FP / P.OUR_MIN
    P["ACT_RATE"] = np.where(P.ACT_MIN > 0, P.ACT / P.ACT_MIN.where(P.ACT_MIN > 0, 1), np.nan)
    print("Swapped players (per slate there are 8 - shared on each side):")
    print(P.groupby("SIDE").agg(picks=("PLAYER", "size"), salary=("SALARY", "mean"), actual=("ACT", "mean"),
                                our_proj=("OUR_FP", "mean"), dff_proj=("DFF_FP", "mean"),
                                our_min=("OUR_MIN", "mean"), act_min=("ACT_MIN", "mean"),
                                dnp=("ACT_MIN", lambda x: (x == 0).mean()),
                                not_in_pool=("IN_OUR_POOL", lambda x: 1 - x.mean())).round(2).to_string())

    print("\nOurs-only picks by risk group (actual vs both projections):")
    o = P[P.SIDE == "ours only"]
    print(o.groupby("GROUP").agg(picks=("PLAYER", "size"), actual=("ACT", "mean"), our_proj=("OUR_FP", "mean"),
                                 dff_proj=("DFF_FP", "mean"), our_min=("OUR_MIN", "mean"),
                                 act_min=("ACT_MIN", "mean")).round(2).to_string())

    d = P[(P.SIDE == "DFF only") & P.IN_OUR_POOL]
    print(f"\nDFF-only picks that WERE in our pool ({len(d)}): we projected {d.OUR_FP.mean():.1f}, DFF {d.DFF_FP.mean():.1f}, "
          f"actual {d.ACT.mean():.1f}; our minutes {d.OUR_MIN.mean():.1f} vs actual {d.ACT_MIN.mean():.1f}")
    n = P[(P.SIDE == "DFF only") & ~P.IN_OUR_POOL]
    print(f"DFF-only picks NOT in our pool ({len(n)}): actual {n.ACT.mean():.1f}, DFF projected {n.DFF_FP.mean():.1f}, "
          f"salary {n.SALARY.mean():.0f}")

    # Where does the per-slate gap come from: minutes vs per-minute, on each side
    both = P[P.IN_OUR_POOL & (P.ACT_MIN > 0)].copy()
    both["MIN_PART"] = both.OUR_RATE * (both.OUR_MIN - both.ACT_MIN)
    both["RATE_PART"] = both.ACT_MIN * (both.OUR_RATE - both.ACT_RATE)
    print("\nOur projection error on swapped players who played (minutes part vs per-minute part):")
    print(both.groupby("SIDE")[["OUR_ERR", "MIN_PART", "RATE_PART"]].mean().round(2).to_string())
    P.to_parquet(ROOT / "data" / "replay" / "swaps_2025_26.parquet", index=False)


if __name__ == "__main__":
    main()
