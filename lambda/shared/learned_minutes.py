"""
Learned minutes model: player-game rows, features, training and prediction, in one place so training
(supervised-learning), serving (minutes-projection) and the backtests (scripts/) compute the same thing.

Rows: one per team-game x player on that team's roster, including players who did not play. A player is on a
team's roster for a game when his latest appearance before tip, in the same season, was for that team.
Pre-game averages use every earlier box-score row of the season, 0-minute rows included.

Tonight's out set: teammates listed Out or Doubtful on the injury report, with a pre-game season average of 10+
minutes, who missed no more than 10 team games. Doubtful players sat 98.8% of the time (2022-26).

Features:
  S_MIN, L7_MIN, PREV_MIN, GP, TEAM_GAMES_MISSED   the player's own season / last-7 / last-game minutes
  OUT_MIN_FRESH / OUT_MIN_ONGOING                  season minutes of out teammates who played the team's
                                                   previous game / did not
  WOWY_MIN, WOWY_N                                 his minutes change in earlier games this season when tonight's
                                                   out teammates sat vs played, summed over them, each shrunk by
                                                   n / (n + 3); n = games behind it
  IS_QUES, IS_PROB, IS_AVAI                        his own status on tonight's report (not listed = all 0)
  PS_MIN, PS_L10_MIN, PS_GP, HAS_PS, CHANGED_TEAM  last season (previous season's box scores only): average and
                                                   last-10 minutes, games, whether he had one, new team now
  TEAM_MARGIN                                      his team's expected margin from the betting spread (game_lines.py);
                                                   starters on 10+ point favorites: minutes bias +1.53 -> +0.40 (2025-26)
Position-based inputs were dropped (2026-10-18): minutes MAE 4.888 with them vs 4.900 without.

Two parts (2026-10-20): P_PLAY, the chance he plays (classifier), and MIN_IF_PLAY, his minutes if he plays
(regressor trained on players who played). Projection = P_PLAY x MIN_IF_PLAY. Walk-forward FP error vs the single
status-blind model: -0.094 (2024-25), -0.106 (2025-26); Questionable minutes bias +6.7 -> -0.6 (findings.md,
"Report status and the two-part model").

FP = projected minutes x blended FP per minute x the pace/defense multiplier (pace_defense.py). The blended rate
pulls this season's rate toward last season's: w = minutes this season / (minutes this season + k), k fit on the
training rows (2022-25 fit: k = 200). Early-season walk-forward (2026-10-25): FP error in the first 10 team games
-0.34 (2024-25), -0.30 (2025-26) with the last-season inputs; findings.md, "Early-season blend".
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from player_names import match_names

OUT_STATUSES = {"OUT", "DOUBTFUL"}
OUT_MIN_SEASON_AVG, OUT_MAX_MISSED, SHRINK_N = 10, 10, 3
MAX_MINUTES = 48
FORMULA_C_CAP = 37
OWN_STATUS_FEATURES = {"QUESTIONABLE": "IS_QUES", "PROBABLE": "IS_PROB", "AVAILABLE": "IS_AVAI"}
LAST_SEASON_FEATURES = ["PS_MIN", "PS_L10_MIN", "PS_GP", "HAS_PS", "CHANGED_TEAM"]
FEATURES = ["S_MIN", "L7_MIN", "PREV_MIN", "GP", "TEAM_GAMES_MISSED",
            "OUT_MIN_FRESH", "OUT_MIN_ONGOING", "WOWY_MIN", "WOWY_N", *OWN_STATUS_FEATURES.values(),
            *LAST_SEASON_FEATURES, "TEAM_MARGIN"]
FILL = -1                      # missing inputs (no last season) for the two-part model
RATE_K_GRID = [0, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1200]
RATE_PRIOR_MIN_MINUTES = 100   # last season's rate counts only with 100+ minutes behind it
GBR_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=0)


def formula_c(gp, s_min, l7_min, prev_min, missed):
    """Formula C without the DFF starter floor; the control model."""
    baseline = np.where(gp >= 4, s_min, 10.0)
    proj = np.minimum(0.5 * baseline + 0.3 * l7_min + 0.2 * prev_min, FORMULA_C_CAP)
    return np.where(missed >= 10, proj * 0.75, proj)


def player_games(box, tonight=None):
    """
    Roster rows with pre-game features. `box` needs SEASON, GAME_DATE, GAME_ID, TEAM_ABBREVIATION, MATCHUP, PLAYER,
    MIN, FP. `tonight` (optional): schedule rows for games not in the box scores yet, with GAME_DATE, GAME_ID, TEAM,
    and SEASON; their outcome columns are 0.
    """
    box = box.copy()
    # datetime64[ns] throughout: pandas 2.1 can't concatenate mixed datetime resolutions
    box["D"] = pd.to_datetime(box["GAME_DATE"]).dt.normalize().astype("datetime64[ns]")
    box["TEAM"] = box["TEAM_ABBREVIATION"]
    box["GAME_ID"] = box["GAME_ID"].astype(str)
    app = box.sort_values(["PLAYER", "D"]).copy()

    tg = (box[["SEASON", "TEAM", "D", "GAME_ID"]].drop_duplicates(["TEAM", "GAME_ID"]))
    if tonight is not None and len(tonight):
        t = tonight.assign(D=pd.to_datetime(tonight["GAME_DATE"]).dt.normalize().astype("datetime64[ns]"),
                           GAME_ID=tonight["GAME_ID"].astype(str))[["SEASON", "TEAM", "D", "GAME_ID"]]
        tg = pd.concat([tg, t[~t.set_index(["TEAM", "GAME_ID"]).index.isin(tg.set_index(["TEAM", "GAME_ID"]).index)]])
    tg = tg.sort_values(["TEAM", "D"]).reset_index(drop=True)
    tg["TEAM_GAME_NO"] = tg.groupby(["SEASON", "TEAM"]).cumcount()
    app = app.merge(tg[["TEAM", "GAME_ID", "TEAM_GAME_NO"]], on=["TEAM", "GAME_ID"], how="left")

    # Each appearance's post-game state is the pre-game state for the player's next game
    g = app.groupby(["PLAYER", "SEASON"])
    app["S_MIN"] = g["MIN"].transform(lambda s: s.expanding().mean())
    app["L7_MIN"] = g["MIN"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    app["S_FP"] = g["FP"].transform(lambda s: s.expanding().mean())
    app["GP"] = g.cumcount() + 1
    state = app.rename(columns={"MIN": "PREV_MIN", "D": "LAST_DATE", "TEAM": "LAST_TEAM", "SEASON": "LAST_SEASON",
                                "TEAM_GAME_NO": "LAST_TEAM_GAME_NO"})[
        ["PLAYER", "LAST_DATE", "LAST_TEAM", "LAST_SEASON", "LAST_TEAM_GAME_NO", "S_MIN", "L7_MIN", "PREV_MIN",
         "S_FP", "GP"]]

    roster = app[["SEASON", "TEAM", "PLAYER"]].drop_duplicates()
    cand = tg.merge(roster, on=["SEASON", "TEAM"])
    cand = pd.merge_asof(cand.sort_values("D"), state.sort_values("LAST_DATE"), left_on="D", right_on="LAST_DATE",
                         by="PLAYER", allow_exact_matches=False)
    cand = cand[(cand["LAST_TEAM"] == cand["TEAM"]) & (cand["LAST_SEASON"] == cand["SEASON"])].copy()
    cand["TEAM_GAMES_MISSED"] = cand["TEAM_GAME_NO"] - cand["LAST_TEAM_GAME_NO"] - 1

    acts = box.groupby(["D", "PLAYER"]).agg(ACT_MIN=("MIN", "sum"), ACT_FP=("FP", "sum")).reset_index()
    cand = cand.merge(acts, on=["D", "PLAYER"], how="left")
    cand[["ACT_MIN", "ACT_FP"]] = cand[["ACT_MIN", "ACT_FP"]].fillna(0)
    cand["PLAYED"] = cand["ACT_MIN"] > 0
    cand["FC_MIN"] = formula_c(cand["GP"], cand["S_MIN"], cand["L7_MIN"], cand["PREV_MIN"], cand["TEAM_GAMES_MISSED"])
    cand["S_RATE"] = np.where(cand["S_MIN"] > 0, cand["S_FP"] / cand["S_MIN"].where(cand["S_MIN"] > 0, 1), np.nan)
    cols = ["SEASON", "D", "GAME_ID", "TEAM", "TEAM_GAME_NO", "PLAYER", "GP", "S_MIN", "L7_MIN", "PREV_MIN", "S_FP",
            "S_RATE", "TEAM_GAMES_MISSED", "FC_MIN", "ACT_MIN", "ACT_FP", "PLAYED"]
    return cand[cols].sort_values(["D", "TEAM", "PLAYER"]).reset_index(drop=True)


def first_seen(box):
    """{player: date of his first box-score row}: the names the matcher may treat as real players before then."""
    return pd.to_datetime(box["GAME_DATE"]).dt.normalize().groupby(box["PLAYER"]).min().to_dict()


def _known_on(known_names, day):
    """A set is used as is; a first_seen dict gives the names seen before `day` (no later games leak in)."""
    if isinstance(known_names, dict):
        return {n for n, d in known_names.items() if d < day}
    return known_names


def attach_listed_out(rows, reports, known_names):
    """
    LISTED_OUT / HAS_REPORT from injury reports (D, PLAYER, STATUS; any capitalization), plus his own STATUS
    ('NOT LISTED' on a report day, None without a report) and the IS_* status flags. Report names are matched to
    box-score names against each day's rosters (player_names). known_names: a set, or a first_seen dict so each day
    only knows players seen before it. Since 2026-10-25 (OKC's Jalen vs Jaylin Williams): a name known from an
    earlier box score is never loosely matched to a different roster player. Returns (rows, unmatched Out names).
    """
    rows = rows.copy()
    rep = reports.assign(D=pd.to_datetime(reports["D"]).dt.normalize().astype("datetime64[ns]"),
                         STATUS=reports["STATUS"].astype(str).str.upper())
    rows["HAS_REPORT"] = rows["D"].isin(set(rep["D"]))
    out = rep[rep["STATUS"].isin(OUT_STATUSES)]
    listed, unmatched = set(), {}
    rosters = rows.groupby("D")["PLAYER"].apply(list)
    for day, g in out.groupby("D"):
        mapping, missing = match_names(g["PLAYER"], rosters.get(day, []), _known_on(known_names, day))
        listed |= {(day, m) for m in mapping.values() if m}
        if missing:
            unmatched[day] = missing
    rows["LISTED_OUT"] = [(d, p) in listed for d, p in zip(rows["D"], rows["PLAYER"])]
    # Own status: every listed name, matched separately so the out set above is unchanged
    status = {}
    for day, g in rep.groupby("D"):
        mapping, _ = match_names(g["PLAYER"], rosters.get(day, []), _known_on(known_names, day))
        status.update({(day, mapping[n]): st for n, st in zip(g["PLAYER"], g["STATUS"]) if mapping.get(n)})
    rows["STATUS"] = [status.get((d, p), "NOT LISTED") if has else None
                      for d, p, has in zip(rows["D"], rows["PLAYER"], rows["HAS_REPORT"])]
    for st, col in OWN_STATUS_FEATURES.items():
        rows[col] = (rows["STATUS"] == st).astype(float)
    return rows, unmatched


def _team_season_absences(ts):
    """Absence features for one team-season (rows = roster player x team game)."""
    games = np.sort(ts["TEAM_GAME_NO"].unique())
    players = ts["PLAYER"].unique()
    gi = {g: k for k, g in enumerate(games)}
    pi = {p: k for k, p in enumerate(players)}
    G, N = len(games), len(players)
    gidx, pidx = ts["TEAM_GAME_NO"].map(gi).values, ts["PLAYER"].map(pi).values

    def grid(values, fill=0.0, dtype=float):
        a = np.full((G, N), fill, dtype)
        a[gidx, pidx] = values
        return a
    on = grid(True, False, bool)
    played = grid(ts["PLAYED"].values, False, bool)
    mins = grid(ts["ACT_MIN"].values)
    absent = on & ~played

    # Earlier-game sums (tonight excluded) for every (player i, teammate x) pair
    def prior(a):
        c = np.cumsum(a, axis=0)
        return np.concatenate([np.zeros((1,) + a.shape[1:]), c[:-1]], axis=0)
    pi_ = played[:, :, None].astype(float)
    n_abs = prior(pi_ * absent[:, None, :])
    n_pre = prior(pi_ * played[:, None, :])
    min_abs = prior(mins[:, :, None] * pi_ * absent[:, None, :])
    min_pre = prior(mins[:, :, None] * pi_ * played[:, None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        d_min = np.where((n_abs > 0) & (n_pre > 0), min_abs / n_abs - min_pre / n_pre, 0.0)
    w = n_abs / (n_abs + SHRINK_N)

    smin = grid(ts["S_MIN"].fillna(0).values)
    missed = grid(ts["TEAM_GAMES_MISSED"].fillna(99).values, 99.0)
    tonight = grid(ts["LISTED_OUT"].values, False, bool)
    is_out = tonight & (smin >= OUT_MIN_SEASON_AVG) & (missed <= OUT_MAX_MISSED)
    fresh, ongoing = is_out & (missed == 0), is_out & (missed > 0)
    others = lambda a: a.sum(1, keepdims=True) - a   # teammates only
    feats = {
        "OUT_MIN_FRESH": others(fresh * smin),
        "OUT_MIN_ONGOING": others(ongoing * smin),
        "WOWY_MIN": np.einsum("gix,gx->gi", w * d_min, is_out.astype(float)),
        "WOWY_N": np.einsum("gix,gx->gi", n_abs, is_out.astype(float)),
    }
    return pd.DataFrame({k: v[gidx, pidx] for k, v in feats.items()}, index=ts.index)


def absence_features(rows):
    """Adds OUT_MIN_FRESH, OUT_MIN_ONGOING, WOWY_MIN, WOWY_N. Needs LISTED_OUT (attach_listed_out)."""
    parts = [_team_season_absences(ts) for _, ts in rows.groupby(["SEASON", "TEAM"])]
    return rows.join(pd.concat(parts))


def training_rows(rows):
    """Players the model projects: a season average exists, the day has an injury report, not listed out."""
    return rows[rows["S_MIN"].notna() & rows["HAS_REPORT"] & ~rows["LISTED_OUT"]]


def previous_season(season):
    """'2025-26' -> '2024-25'."""
    y = int(season[:4]) - 1
    return f"{y}-{str(y + 1)[2:]}"


def add_last_season(rows, box):
    """
    PS_* inputs from each row's previous season. `box` must hold that season's box scores (SEASON, GAME_DATE,
    PLAYER, TEAM_ABBREVIATION, MIN, FP). Also S_MINSUM, his minutes this season before tonight (the rate's sample).
    """
    b = box.assign(D=pd.to_datetime(box["GAME_DATE"]).dt.normalize()).sort_values(["PLAYER", "D"])
    g = b.groupby(["SEASON", "PLAYER"])
    prof = pd.DataFrame({"PS_MIN": g["MIN"].mean(), "PS_GP": g.size(), "PS_FP": g["FP"].sum(),
                         "PS_MINSUM": g["MIN"].sum(), "PS_L10_MIN": g["MIN"].apply(lambda x: x.tail(10).mean()),
                         "PS_TEAM": g["TEAM_ABBREVIATION"].last()}).reset_index()
    prof["PS_RATE"] = np.where(prof["PS_MINSUM"] > 0, prof["PS_FP"] / prof["PS_MINSUM"].where(prof["PS_MINSUM"] > 0, 1),
                               np.nan)
    prof = prof.rename(columns={"SEASON": "PREV_SEASON"}).drop(columns="PS_FP")
    rows = rows.drop(columns=[c for c in prof.columns if c in rows.columns and c != "PLAYER"] +
                     [c for c in ("HAS_PS", "CHANGED_TEAM", "S_MINSUM") if c in rows.columns])
    rows = rows.assign(PREV_SEASON=rows["SEASON"].astype(str).map(previous_season))
    rows = rows.merge(prof, on=["PREV_SEASON", "PLAYER"], how="left")
    rows["HAS_PS"] = rows["PS_GP"].notna().astype(float)
    rows["CHANGED_TEAM"] = (rows["PS_TEAM"].notna() & (rows["PS_TEAM"] != rows["TEAM"])).astype(float)
    rows["S_MINSUM"] = rows["S_MIN"] * rows["GP"]
    return rows


def blend_rate(rows, k):
    """This season's FP per minute pulled toward last season's: w = S_MINSUM / (S_MINSUM + k)."""
    w = rows["S_MINSUM"] / (rows["S_MINSUM"] + k) if k else 1.0
    use = rows["PS_RATE"].notna() & (rows["PS_MINSUM"] >= RATE_PRIOR_MIN_MINUTES)
    return pd.Series(np.where(use, w * rows["S_RATE"] + (1 - w) * rows["PS_RATE"], rows["S_RATE"]), index=rows.index)


def fit_rate_k(rows):
    """k with the lowest minute-weighted rate error on played rows. Returns (k, {k: error})."""
    p = rows[(rows["ACT_MIN"] > 0) & (rows["S_RATE"] > 0)]
    actual = p["ACT_FP"] / p["ACT_MIN"]
    err = {k: float((np.abs(blend_rate(p, k) - actual) * p["ACT_MIN"]).sum() / p["ACT_MIN"].sum()) for k in RATE_K_GRID}
    return min(err, key=err.get), err


def train(rows):
    """{'play': chance he plays, 'minutes': minutes if he plays, 'features': FEATURES}."""
    play = GradientBoostingClassifier(**GBR_PARAMS).fit(rows[FEATURES].fillna(FILL), rows["PLAYED"])
    played = rows[rows["PLAYED"]]
    minutes = GradientBoostingRegressor(**GBR_PARAMS).fit(played[FEATURES].fillna(FILL), played["ACT_MIN"])
    return {"play": play, "minutes": minutes, "features": list(FEATURES), "fill": FILL}


def predict_parts(model, rows):
    """(P_PLAY, MIN_IF_PLAY) arrays."""
    X = rows[model["features"]].fillna(model.get("fill", 0))
    return model["play"].predict_proba(X)[:, 1], np.clip(model["minutes"].predict(X), 0, MAX_MINUTES)


def predict(model, rows):
    """Projected minutes = P_PLAY x MIN_IF_PLAY. A single regressor (models before 2026-10-20) still works."""
    if not isinstance(model, dict):
        return np.clip(model.predict(rows[list(model.feature_names_in_)].fillna(0)), 0, MAX_MINUTES)
    p, m = predict_parts(model, rows)
    return p * m


def build(box, reports, tonight=None, known_names=None):
    """Rows + listed-out + absence features. Returns (rows, unmatched report names per day)."""
    rows = player_games(box, tonight)
    known = first_seen(box) if known_names is None else known_names
    rows, unmatched = attach_listed_out(rows, reports, known)
    return absence_features(rows), unmatched
