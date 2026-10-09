"""
Learned injury response: who absorbs an absent teammate's minutes, and how much each teammate's FP per minute rises.
Fit walk-forward on the replay table (data/replay/replay.parquet) plus the pre-tip injury reports
(data/replay/injury_reports.parquet, scripts/fetch_injury_history.py). Read-only; writes
data/replay/injury_features.parquet and data/replay/learned_2025_26.parquet (2025-26 predictions, trained on 2022-25).

Tonight's out set (teammates whose minutes are up for grabs), two versions:
  realized (hindsight)  on the roster and did not play tonight
  report (pre-tip)      listed Out on the last injury report before the day's first tip (G League excluded)
Both keep only teammates with a pre-game season average >= 10 min who missed <= 10 team games.

Per player per game (report versions carry an _R suffix):
  OUT_MIN_FRESH / OUT_MIN_ONGOING   season minutes of out teammates who played the team's previous game / did not
  OUT_USG_FRESH / OUT_USG_ONGOING   their usage-minutes (season usage per minute x season minutes)
  OUT_MIN_SAME / OUT_MIN_ADJ        fresh out minutes at his position / an overlapping position (complex's rule)
  WOWY_MIN, WOWY_RATE, WOWY_N       his minutes / FP-per-minute change in earlier games this season when tonight's
                                    out teammates sat vs played, summed over them, each shrunk by n / (n + 3)
History for the with/without terms is always realized (earlier games are known before tip).

Minutes models (gradient boosting), three ways:
  hindsight   realized out set, trained and scored on players who played (the first, leaky setup)
  own only    no teammate-absence inputs at all
  pre-tip     report out set, trained and scored on everyone not listed Out (DNPs count as 0 minutes)
Rate model: actual FP per minute, players who played 8+ minutes, weighted by minutes.

Usage: python scripts/injury_response.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lambda" / "minutes-projection"))
from projection_models import position_overlap_complex  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_injury_history import name_key  # noqa: E402

REPLAY = ROOT / "data" / "replay" / "replay.parquet"
REPORTS = ROOT / "data" / "replay" / "injury_reports.parquet"
POOL = ROOT / "data" / "replay" / "pool_2025_26.parquet"
OUT = ROOT / "data" / "replay" / "injury_features.parquet"
PRED_OUT = ROOT / "data" / "replay" / "learned_2025_26.parquet"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
OUT_MIN_SEASON_AVG, OUT_MAX_MISSED, SHRINK_N = 10, 10, 3
GBR = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=0)

OWN_FEATS = ["S_MIN", "L7_MIN", "PREV_MIN", "GP", "TEAM_GAMES_MISSED"]
ABS_FEATS = ["OUT_MIN_FRESH", "OUT_MIN_ONGOING", "OUT_MIN_SAME", "OUT_MIN_ADJ", "WOWY_MIN", "WOWY_N"]
RATE_OWN = ["S_RATE", "L7_RATE", "C_RATE", "S_USG", "S_MIN"]
RATE_ABS = ["OUT_USG_FRESH", "OUT_USG_ONGOING", "OUT_MIN_FRESH", "WOWY_RATE", "WOWY_N"]
FEAT_NAMES = ["OUT_MIN_FRESH", "OUT_MIN_ONGOING", "OUT_USG_FRESH", "OUT_USG_ONGOING", "OUT_MIN_SAME", "OUT_MIN_ADJ",
              "WOWY_MIN", "WOWY_RATE", "WOWY_N"]


def suffixed(cols, sfx):
    return [c + sfx if c in FEAT_NAMES else c for c in cols]


def team_season_features(ts, tonight_cols):
    """Absence and with/without features for one team-season (rows = roster player x team game), one set per
    tonight-out column in tonight_cols ({suffix: boolean column})."""
    ts = ts.sort_values("TEAM_GAME_NO")
    games = np.sort(ts.TEAM_GAME_NO.unique())
    players = ts.PLAYER.unique()
    gi = {g: k for k, g in enumerate(games)}
    pi = {p: k for k, p in enumerate(players)}
    G, N = len(games), len(players)
    gidx, pidx = ts.TEAM_GAME_NO.map(gi).values, ts.PLAYER.map(pi).values

    def grid(col, fill=0.0, dtype=float):
        a = np.full((G, N), fill, dtype)
        a[gidx, pidx] = ts[col].values
        return a
    on = np.zeros((G, N), bool)
    on[gidx, pidx] = True
    played = grid("PLAYED", False, bool)
    mins, fp = grid("ACT_MIN"), grid("ACT_FP")
    absent = on & ~played

    # Prior-game sums (exclusive of tonight) for every (teammate i, other player x) pair
    def prior(a):
        c = np.cumsum(a, axis=0)
        return np.concatenate([np.zeros((1,) + a.shape[1:]), c[:-1]], axis=0)
    pi_ = played[:, :, None].astype(float)
    ax, px = absent[:, None, :].astype(float), played[:, None, :].astype(float)
    m_i, f_i = mins[:, :, None], fp[:, :, None]
    n_abs, n_pre = prior(pi_ * ax), prior(pi_ * px)
    min_abs, min_pre = prior(m_i * pi_ * ax), prior(m_i * pi_ * px)
    fp_abs, fp_pre = prior(f_i * pi_ * ax), prior(f_i * pi_ * px)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_min = np.where((n_abs > 0) & (n_pre > 0), min_abs / n_abs - min_pre / n_pre, 0.0)
        d_rate = np.where((min_abs > 0) & (min_pre > 0), fp_abs / min_abs - fp_pre / min_pre, 0.0)
    w = n_abs / (n_abs + SHRINK_N)

    pos = ts.drop_duplicates("PLAYER").set_index("PLAYER").POSITION.to_dict()
    same = np.array([[pos[a] == pos[b] for b in players] for a in players], float)
    adj = np.array([[position_overlap_complex(pos[a], pos[b]) and pos[a] != pos[b] for b in players]
                    for a in players], float)
    smin, susg = np.zeros((G, N)), np.zeros((G, N))
    smin[gidx, pidx] = ts.S_MIN.fillna(0).values
    susg[gidx, pidx] = ts.S_USG.fillna(0).values
    missed = np.full((G, N), 99.0)
    missed[gidx, pidx] = ts.TEAM_GAMES_MISSED.fillna(99).values

    out = ts[["SEASON", "D", "TEAM", "PLAYER"]].copy()
    for sfx, col in tonight_cols.items():
        tonight = grid(col, False, bool)
        is_out = tonight & (smin >= OUT_MIN_SEASON_AVG) & (missed <= OUT_MAX_MISSED)
        fresh, ongoing = is_out & (missed == 0), is_out & (missed > 0)
        own = lambda a: a.sum(1, keepdims=True) - a   # teammates only
        feats = {
            "OUT_MIN_FRESH": own(fresh * smin),
            "OUT_MIN_ONGOING": own(ongoing * smin),
            "OUT_USG_FRESH": own(fresh * smin * susg),
            "OUT_USG_ONGOING": own(ongoing * smin * susg),
            "OUT_MIN_SAME": (fresh * smin) @ same.T - fresh * smin,
            "OUT_MIN_ADJ": (fresh * smin) @ adj.T,
            "WOWY_MIN": np.einsum("gix,gx->gi", w * d_min, is_out.astype(float)),
            "WOWY_RATE": np.einsum("gix,gx->gi", w * d_rate, is_out.astype(float)),
            "WOWY_N": np.einsum("gix,gx->gi", n_abs, is_out.astype(float)),
        }
        for k, v in feats.items():
            out[k + sfx] = v[gidx, pidx]
    return out


def build_features(r, reports):
    # Match on letters only, ignoring suffixes: box-score spellings change across seasons ("jimmy butler iii")
    rep = reports[(reports.STATUS == "Out") & ~reports.GLEAGUE].assign(KEY=lambda d: d.PLAYER.map(name_key))
    rep = rep.drop_duplicates(["D", "KEY"])[["D", "KEY"]].assign(LISTED_OUT=True)
    r = r.assign(KEY=r.PLAYER.map(name_key)).merge(rep, on=["D", "KEY"], how="left").drop(columns="KEY")
    r["LISTED_OUT"] = r.LISTED_OUT.fillna(False).astype(bool)
    r["HAS_REPORT"] = r.D.isin(set(reports.D))
    r["NOT_PLAYED"] = ~r.PLAYED
    parts = [team_season_features(ts, {"": "NOT_PLAYED", "_R": "LISTED_OUT"}) for _, ts in r.groupby(["SEASON", "TEAM"])]
    r = r.merge(pd.concat(parts, ignore_index=True), on=["SEASON", "D", "TEAM", "PLAYER"], how="left")
    safe = lambda a, b: np.where(b > 0, a / b.where(b > 0, 1), np.nan)
    r["S_RATE"] = safe(r.S_FP, r.S_MIN)
    r["L7_RATE"] = safe(r.L7_FP, r.L7_MIN)
    r["C_RATE"] = safe(r.C_FP, r.C_MIN)
    r["ACT_RATE"] = safe(r.ACT_FP, r.ACT_MIN)
    return r


def fitted_formula(train):
    X = lambda d: pd.DataFrame({"S": d.S_MIN, "L7": d.L7_MIN, "PREV": d.PREV_MIN, "FEW": (d.GP < 4).astype(float),
                                "RET": (d.TEAM_GAMES_MISSED >= 10).astype(float), "MISS": d.TEAM_GAMES_MISSED.clip(upper=10)})
    m = LinearRegression().fit(X(train), train.ACT_MIN)
    return lambda d: np.clip(m.predict(X(d)), 0, 40)


def gbr(train, feats, target, weight=None):
    return GradientBoostingRegressor(**GBR).fit(train[feats].fillna(0), train[target], sample_weight=weight)


def report(name, d, cols):
    out = []
    for label, col in cols:
        e = d[col] - d.ACT_MIN
        out.append(f"{label} {np.abs(e).mean():.3f} ({e.mean():+.2f})")
    print(f"  {name:36s} n={len(d):6,d} | " + " | ".join(out))


def main():
    reports = pd.read_parquet(REPORTS)
    r = build_features(pd.read_parquet(REPLAY), reports)
    r.to_parquet(OUT, index=False)
    print(f"Report coverage: {r.HAS_REPORT.mean():.1%} of rows; listed Out and played anyway: "
          f"{(r.LISTED_OUT & r.PLAYED).sum()} of {r.LISTED_OUT.sum()}")
    rot = r[(r.S_MIN >= OUT_MIN_SEASON_AVG) & (r.TEAM_GAMES_MISSED <= OUT_MAX_MISSED) & r.HAS_REPORT]
    sat = rot[~rot.PLAYED]
    print(f"Rotation players who sat: {len(sat):,}; listed Out before tip {sat.LISTED_OUT.mean():.1%}, "
          f"Doubtful/Questionable/other {1 - sat.LISTED_OUT.mean():.1%}\n")

    base = r[r.S_MIN.notna() & r.HAS_REPORT]
    played = base[base.PLAYED]
    eligible = base[~base.LISTED_OUT]           # what production projects: everyone not listed Out
    own, hind, pre = OWN_FEATS, OWN_FEATS + ABS_FEATS, OWN_FEATS + suffixed(ABS_FEATS, "_R")

    print("=== Minutes: average minutes off (bias), walk-forward ===")
    preds = None
    for i in range(1, len(SEASONS)):
        tr_p, te_p = played[played.SEASON.isin(SEASONS[:i])], played[played.SEASON == SEASONS[i]].copy()
        tr_e, te_e = eligible[eligible.SEASON.isin(SEASONS[:i])], eligible[eligible.SEASON == SEASONS[i]].copy()
        te_p["HIND"] = np.clip(gbr(tr_p, hind, "ACT_MIN").predict(te_p[hind].fillna(0)), 0, 48)
        te_p["OWN"] = np.clip(gbr(tr_p, own, "ACT_MIN").predict(te_p[own].fillna(0)), 0, 48)
        te_p["FITTED"] = fitted_formula(tr_p)(te_p)
        m_pre, m_own = gbr(tr_e, pre, "ACT_MIN"), gbr(tr_e, own, "ACT_MIN")
        te_e["PRE"] = np.clip(m_pre.predict(te_e[pre].fillna(0)), 0, 48)
        te_e["OWN"] = np.clip(m_own.predict(te_e[own].fillna(0)), 0, 48)
        te_e["FITTED"] = fitted_formula(tr_e)(te_e)
        print(f"test {SEASONS[i]}:")
        cols = [("Formula C", "FC_MIN"), ("fitted", "FITTED"), ("own only", "OWN"), ("hindsight", "HIND")]
        report("[players who played] all", te_p, cols)
        report("[played] fresh 25+ min actually sat", te_p[te_p.OUT_MIN_FRESH >= 25], cols)
        cols = [("Formula C", "FC_MIN"), ("fitted", "FITTED"), ("own only", "OWN"), ("pre-tip", "PRE")]
        report("[not listed Out] all, DNPs = 0", te_e, cols)
        report("[not listed] fresh 25+ min listed Out", te_e[te_e.OUT_MIN_FRESH_R >= 25], cols)
        if SEASONS[i] == SEASONS[-1]:
            preds = te_e[["D", "PLAYER", "TEAM", "PRE", "OWN"]].rename(columns={"PRE": "LEARNED_MIN", "OWN": "OWN_MIN"})

    print("\n=== 2025-26 production pool: learned (pre-tip, trained 2022-25) vs the complex model's stored projections ===")
    pool = pd.read_parquet(POOL).merge(preds, on=["D", "PLAYER"], how="left", suffixes=("", "_L"))
    pool = pool.merge(r[["D", "PLAYER", "ACT_MIN", "FC_MIN", "OUT_MIN_FRESH_R"]], on=["D", "PLAYER"], how="left")
    pool["ACT_MIN"] = pool.ACT_MIN.fillna(0)
    ok = pool[pool.LEARNED_MIN.notna()]
    print(f"pool rows {len(pool):,}, with a learned projection {len(ok):,}")
    cols = [("complex (prod)", "PROJECTED_MIN"), ("Formula C", "FC_MIN"), ("own only", "OWN_MIN"), ("pre-tip", "LEARNED_MIN")]
    report("pool, all", ok, cols)
    report("pool, fresh 25+ min listed Out", ok[ok.OUT_MIN_FRESH_R >= 25], cols)
    report("pool, complex added 0.5+ min", ok[ok.GROUP == "injury boost"], cols)

    print("\n=== FP per minute, players who played 8+ min (minutes-weighted error) ===")
    rated = played[(played.ACT_MIN >= 8) & played.S_RATE.notna()]
    rate_pre = RATE_OWN + suffixed(RATE_ABS, "_R")
    for i in range(1, len(SEASONS)):
        tr, te = rated[rated.SEASON.isin(SEASONS[:i])], rated[rated.SEASON == SEASONS[i]].copy()
        te["HIND"] = gbr(tr, RATE_OWN + RATE_ABS, "ACT_RATE", tr.ACT_MIN).predict(te[RATE_OWN + RATE_ABS].fillna(0))
        te["OWN"] = gbr(tr, RATE_OWN, "ACT_RATE", tr.ACT_MIN).predict(te[RATE_OWN].fillna(0))
        m = gbr(tr, rate_pre, "ACT_RATE", tr.ACT_MIN)
        te["PRE"] = m.predict(te[rate_pre].fillna(0))
        print(f"test {SEASONS[i]}:")
        for name, d in (("all", te), ("fresh usage listed Out", te[te.OUT_USG_FRESH_R >= 10])):
            parts = [f"{lab} {np.average(np.abs(d[c] - d.ACT_RATE), weights=d.ACT_MIN):.4f} "
                     f"({np.average(d[c] - d.ACT_RATE, weights=d.ACT_MIN):+.4f})"
                     for lab, c in (("season", "S_RATE"), ("own only", "OWN"), ("hindsight", "HIND"), ("pre-tip", "PRE"))]
            print(f"  {name:24s} n={len(d):6,d} | " + " | ".join(parts))
        if SEASONS[i] == SEASONS[-1]:
            all_e = eligible[eligible.SEASON == SEASONS[-1]]
            preds = preds.merge(all_e[["D", "PLAYER"]].assign(LEARNED_RATE=m.predict(all_e[rate_pre].fillna(0)),
                                                              S_RATE=all_e.S_RATE.values),
                                on=["D", "PLAYER"], how="left")
    preds.to_parquet(PRED_OUT, index=False)


if __name__ == "__main__":
    main()
