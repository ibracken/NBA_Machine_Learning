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
Position-based inputs were dropped (2026-10-18): minutes MAE 4.888 with them vs 4.900 without.

FP = projected minutes x season FP per minute (S_RATE).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from player_names import match_names

OUT_STATUSES = {"OUT", "DOUBTFUL"}
OUT_MIN_SEASON_AVG, OUT_MAX_MISSED, SHRINK_N = 10, 10, 3
MAX_MINUTES = 48
FORMULA_C_CAP = 37
FEATURES = ["S_MIN", "L7_MIN", "PREV_MIN", "GP", "TEAM_GAMES_MISSED",
            "OUT_MIN_FRESH", "OUT_MIN_ONGOING", "WOWY_MIN", "WOWY_N"]
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


def attach_listed_out(rows, reports, known_names):
    """
    LISTED_OUT / HAS_REPORT from injury reports (D, PLAYER, STATUS; any capitalization). Report names are matched
    to box-score names against each day's rosters (player_names). Returns (rows, unmatched names per day).
    """
    rows = rows.copy()
    rep = reports.assign(D=pd.to_datetime(reports["D"]).dt.normalize().astype("datetime64[ns]"),
                         STATUS=reports["STATUS"].astype(str).str.upper())
    rows["HAS_REPORT"] = rows["D"].isin(set(rep["D"]))
    out = rep[rep["STATUS"].isin(OUT_STATUSES)]
    listed, unmatched = set(), {}
    rosters = rows.groupby("D")["PLAYER"].apply(list)
    for day, g in out.groupby("D"):
        mapping, missing = match_names(g["PLAYER"], rosters.get(day, []), known_names)
        listed |= {(day, m) for m in mapping.values() if m}
        if missing:
            unmatched[day] = missing
    rows["LISTED_OUT"] = [(d, p) in listed for d, p in zip(rows["D"], rows["PLAYER"])]
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


def train(rows):
    model = GradientBoostingRegressor(**GBR_PARAMS)
    model.fit(rows[FEATURES].fillna(0), rows["ACT_MIN"])
    return model


def predict(model, rows):
    return np.clip(model.predict(rows[FEATURES].fillna(0)), 0, MAX_MINUTES)


def build(box, reports, tonight=None, known_names=None):
    """Rows + listed-out + absence features. Returns (rows, unmatched report names per day)."""
    rows = player_games(box, tonight)
    known = set(box["PLAYER"]) if known_names is None else known_names
    rows, unmatched = attach_listed_out(rows, reports, known)
    return absence_features(rows), unmatched
