"""
Pace / opponent-defense adjustment to FP per minute, shared by training (supervised-learning), serving
(minutes-projection) and the backtest (scripts/backtest_vs_dff.py). Results: documentation/findings.md,
"Pace and opponent defense (2026-10-20)".

Team-game: possessions = FGA + 0.44 FTA - OREB + TOV, averaged over both teams; PACE = possessions per 48 min
(overtime-adjusted); DEF_FP100 = DK FP allowed per 100 possessions.
Pre-game team rating = the team's mean over its earlier games this season, shrunk to the league mean of earlier
games: (n * team + K * league) / (n + K).
  EXP_PACE  team pace + opponent pace - league pace
  DEF_IDX   opponent's FP allowed per 100 / league
Player context, minute-weighted over his earlier games this season: P_PLAYER (realized game pace) and FACED_DEF
(the DEF_IDX of the opponents he faced). Inputs: PACE_RATIO = log(EXP_PACE / P_PLAYER), DEF_RATIO =
log(DEF_IDX / FACED_DEF); missing (no earlier game this season) = 0.

FP per minute = S_RATE x (1 + a PACE_RATIO + b DEF_RATIO); a, b fit by minute-weighted least squares on the training
rows (2022-25 fit: a 0.97, b 1.17).
"""

import numpy as np
import pandas as pd

SHRINK_K = 10
INPUTS = ["PACE_RATIO", "DEF_RATIO"]


def _dates(s):
    return pd.to_datetime(s).dt.normalize().astype("datetime64[ns]")


def team_games(box, tonight=None):
    """One row per team-game: SEASON, GAME_ID, TEAM, OPP, D, PACE, DEF_FP100. Tonight's games (GAME_DATE, GAME_ID,
    TEAM, SEASON; both teams) are appended with no stats."""
    b = box.assign(GAME_ID=box["GAME_ID"].astype(str), D=_dates(box["GAME_DATE"]))
    t = b.groupby(["SEASON", "GAME_ID", "TEAM_ABBREVIATION"]).agg(
        D=("D", "first"), MIN=("MIN", "sum"), FGA=("FGA", "sum"), FTA=("FTA", "sum"), OREB=("OREB", "sum"),
        TOV=("TOV", "sum"), FP=("FP", "sum")).reset_index().rename(columns={"TEAM_ABBREVIATION": "TEAM"})
    t["POSS_T"] = t["FGA"] + 0.44 * t["FTA"] - t["OREB"] + t["TOV"]
    opp = t[["GAME_ID", "TEAM", "POSS_T", "FP"]].rename(columns={"TEAM": "OPP", "POSS_T": "POSS_O", "FP": "FP_O"})
    t = t.merge(opp, on="GAME_ID")
    t = t[t["TEAM"] != t["OPP"]].copy()
    poss = (t["POSS_T"] + t["POSS_O"]) / 2
    t["PACE"] = poss * 240 / t["MIN"]
    t["DEF_FP100"] = t["FP_O"] / poss * 100
    t = t[["SEASON", "GAME_ID", "TEAM", "OPP", "D", "PACE", "DEF_FP100"]]
    if tonight is not None and len(tonight):
        n = tonight.assign(GAME_ID=tonight["GAME_ID"].astype(str), D=_dates(tonight["GAME_DATE"]))
        n = n[~n["GAME_ID"].isin(set(t["GAME_ID"]))][["SEASON", "GAME_ID", "TEAM", "D"]]
        n = n.merge(n[["GAME_ID", "TEAM"]].rename(columns={"TEAM": "OPP"}), on="GAME_ID")
        t = pd.concat([t, n[n["TEAM"] != n["OPP"]]], ignore_index=True)
    return t.sort_values(["TEAM", "D"]).reset_index(drop=True)


def team_ratings(t, k=SHRINK_K):
    """Adds pre-game EXP_PACE and DEF_IDX (games strictly before D, same season) to team_games output."""
    t = t.copy()
    stats = {"PACE": 100.0, "DEF_FP100": None}   # opening-night fallback; DEF_IDX is 1 whatever it is
    g = t.groupby(["SEASON", "TEAM"])
    t["N_PRE"] = g["PACE"].transform(lambda x: x.shift().notna().cumsum())
    daily = t.groupby(["SEASON", "D"])[list(stats)].agg(["sum", "count"])
    for s, fallback in stats.items():
        raw = g[s].transform(lambda x: x.shift().expanding().mean())
        cs = daily[(s, "sum")].groupby(level=0).cumsum().groupby(level=0).shift()
        cn = daily[(s, "count")].groupby(level=0).cumsum().groupby(level=0).shift()
        lg = (cs / cn).rename("LG").reset_index()
        league = t[["SEASON", "D"]].merge(lg, on=["SEASON", "D"], how="left")["LG"].values
        league = pd.Series(league, index=t.index).fillna(fallback if fallback is not None else t[s].mean())
        t["LG_" + s] = league
        t[s + "_PRE"] = (t["N_PRE"] * raw.fillna(league) + k * league) / (t["N_PRE"] + k)
    opp = t[["GAME_ID", "TEAM", "PACE_PRE", "DEF_FP100_PRE"]].rename(
        columns={"TEAM": "OPP", "PACE_PRE": "OPP_PACE_PRE", "DEF_FP100_PRE": "OPP_DEF_PRE"})
    t = t.merge(opp, on=["GAME_ID", "OPP"], how="left")
    t["EXP_PACE"] = t["PACE_PRE"] + t["OPP_PACE_PRE"] - t["LG_PACE"]
    t["DEF_IDX"] = t["OPP_DEF_PRE"] / t["LG_DEF_FP100"]
    return t


def add_inputs(rows, box, tonight=None):
    """learned_minutes rows + OPP, EXP_PACE, DEF_IDX, P_PLAYER, FACED_DEF, PACE_RATIO, DEF_RATIO."""
    t = team_ratings(team_games(box, tonight))
    rows = rows.drop(columns=[c for c in ["OPP", "EXP_PACE", "DEF_IDX", "P_PLAYER", "FACED_DEF", *INPUTS]
                              if c in rows.columns])
    rows = rows.merge(t[["GAME_ID", "TEAM", "OPP", "EXP_PACE", "DEF_IDX"]], on=["GAME_ID", "TEAM"], how="left")

    # Player context from his box-score games: the realized pace and the opponent's pre-game DEF_IDX of each
    b = box.assign(GAME_ID=box["GAME_ID"].astype(str), D=_dates(box["GAME_DATE"]))
    b = b.merge(t[["GAME_ID", "TEAM", "PACE", "DEF_IDX"]], left_on=["GAME_ID", "TEAM_ABBREVIATION"],
                right_on=["GAME_ID", "TEAM"]).sort_values(["PLAYER", "D"])
    keys = [b["PLAYER"], b["SEASON"]]
    w = b["MIN"].astype(float)
    st = pd.DataFrame({"PLAYER": b["PLAYER"], "LAST_SEASON": b["SEASON"], "LAST_D": b["D"],
                       "CW": w.groupby(keys).cumsum(), "CP": (w * b["PACE"]).groupby(keys).cumsum(),
                       "CD": (w * b["DEF_IDX"]).groupby(keys).cumsum()})
    order = rows.index
    rows = pd.merge_asof(rows.reset_index().sort_values("D"), st.sort_values("LAST_D"), left_on="D",
                         right_on="LAST_D", by="PLAYER", allow_exact_matches=False).set_index("index").loc[order]
    ok = (rows["LAST_SEASON"] == rows["SEASON"]) & (rows["CW"] > 0)
    rows["P_PLAYER"] = np.where(ok, rows["CP"] / rows["CW"].where(ok, 1), np.nan)
    rows["FACED_DEF"] = np.where(ok, rows["CD"] / rows["CW"].where(ok, 1), np.nan)
    rows["PACE_RATIO"] = np.log(rows["EXP_PACE"] / rows["P_PLAYER"])
    rows["DEF_RATIO"] = np.log(rows["DEF_IDX"] / rows["FACED_DEF"])
    return rows.drop(columns=["LAST_SEASON", "LAST_D", "CW", "CP", "CD"])


def fit(rows):
    """{'PACE_RATIO': a, 'DEF_RATIO': b}: minute-weighted least squares of (actual rate - S_RATE) on
    S_RATE x inputs, no intercept, over played rows with a season rate."""
    d = rows[(rows["ACT_MIN"] > 0) & (rows["S_RATE"] > 0) & rows[INPUTS].notna().all(axis=1)]
    w = np.sqrt(d["ACT_MIN"].values.astype(float))
    X = d[INPUTS].values * d["S_RATE"].values[:, None]
    y = d["ACT_FP"].values / d["ACT_MIN"].values - d["S_RATE"].values
    coef = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
    return {c: float(v) for c, v in zip(INPUTS, coef)}, len(d)


def multiplier(rows, coefs):
    """Tonight's FP-per-minute multiplier on the season rate; 1 where an input is missing."""
    return 1 + sum(coefs.get(c, 0.0) * rows[c].fillna(0) for c in INPUTS)
