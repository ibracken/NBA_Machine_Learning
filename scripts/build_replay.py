"""
Historical replay table: every rotation player for every team-game, 2022-23 .. 2025-26. Read-only on S3;
writes data/replay/replay.parquet locally (gitignored).

One row per (team-game, player on that team's roster). A player is on the roster for a game when his most
recent appearance before tip, in the same season, was for that team; he stays on it until he appears for
another team. Rows include players who did not play (ACT_MIN = 0), so DNPs, benchings and absences are
visible - box scores alone only show who played.

Pre-game columns use only games before tip, the way minutes-projection serves them (season and last-7 within
the season, career advanced from the box-score career columns):
  S_MIN, L7_MIN, PREV_MIN, GP          season avg / last-7 avg / last game minutes, games played this season
  S_FP, L7_FP, L3_FP, C_MIN, C_FP, CG  FP equivalents and career averages / career games
  S_USG                                season usage per minute: (FGA + 0.44*FTA + TOV) / MIN
  TEAM_GAMES_MISSED, DAYS_OFF          team games since his last appearance, days since it
  FC_MIN                               production Formula C (no DFF starter floor - no history)
  FREED_NEW_MIN, FREED_ALL_MIN         season-avg minutes of rotation teammates (S_MIN >= 15) absent tonight:
                                       NEW = played the team's previous game (fresh absence), ALL = missed <= 10
  POSITION, POS_SOURCE                 single position as DFF lists it (PG/SG/SF/PF/C; production's overlap logic
                                       expects single positions). 'dff' = the player's most common DFF listing;
                                       'nba_coarse' = NBA player index group (G, F, C, G-F, F-C), resolved to one
                                       position by a classifier trained on DFF-labeled players' per-minute stats.
                                       Re-check findings on 'dff' rows.
Outcome columns: ACT_MIN, ACT_FP (0 when he did not play), PLAYED.

Needs PROXY_URL (from .env) for the NBA player index.

Usage: python scripts/build_replay.py
"""

import io
from pathlib import Path

import importlib.util
import sys

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

BUCKET = "nba-prediction-ibracken"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
OUT = Path(__file__).resolve().parents[1] / "data" / "replay" / "replay.parquet"
MAX_MINUTES = 37           # minutes-projection config.MAX_MINUTES
ROTATION_MIN = 15          # season-avg minutes for a teammate's absence to count as freed minutes
# Positions each NBA coarse group can resolve to
COARSE_CHOICES = {'G': ['PG', 'SG'], 'F': ['SF', 'PF'], 'C': ['C'], 'G-F': ['SG', 'SF'], 'F-G': ['SG', 'SF'],
                  'F-C': ['PF', 'C'], 'C-F': ['PF', 'C']}
STYLE_STATS = ['AST', 'REB', 'BLK', 'FG3A', 'STL', 'OREB']


def load(key):
    s3 = boto3.client("s3")
    return pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()))


def formula_c(gp, s_min, l7_min, prev_min, missed):
    """Production Formula C without the DFF starter floor (projection_models.project_minutes_formula_c)."""
    baseline = np.where(gp >= 4, s_min, 10.0)
    proj = np.minimum(0.5 * baseline + 0.3 * l7_min + 0.2 * prev_min, MAX_MINUTES)
    return np.where(missed >= 10, proj * 0.75, proj)


def nba_player_positions():
    """PERSON_ID -> coarse position for every player in NBA history (stats.nba.com playerindex)."""
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    spec = importlib.util.spec_from_file_location("bss", root / "lambda" / "box-score-scraper" / "lambda_function.py")
    bss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bss)
    params = {'LeagueID': '00', 'Season': SEASONS[-1], 'Historical': '1', 'TeamID': '0', 'Active': '',
              'AllStar': '', 'College': '', 'Country': '', 'DraftPick': '', 'DraftRound': '', 'DraftYear': '',
              'Height': '', 'Weight': ''}
    data = bss.nba_api_get('https://stats.nba.com/stats/playerindex', params).json()['resultSets'][0]
    df = pd.DataFrame(data['rowSet'], columns=data['headers'])
    return dict(zip(df['PERSON_ID'], df['POSITION']))


def player_positions(box):
    """
    PLAYER -> (POSITION, POS_SOURCE). DFF's listing when we have one; otherwise the NBA coarse group, resolved to
    one position by a per-minute-stats classifier trained on DFF-labeled players (restricted to the group's
    choices, e.g. a 'G' is PG or SG).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    daily = load("data/daily_predictions/current.parquet").dropna(subset=["POSITION"])
    dff = daily.groupby("PLAYER")["POSITION"].agg(lambda s: s.value_counts().index[0])
    coarse = nba_player_positions()
    totals = box[box["MIN"] > 0].groupby("PLAYER")[STYLE_STATS + ["MIN"]].sum()
    style = totals[STYLE_STATS].div(totals["MIN"], axis=0) * 36
    ids = box.drop_duplicates("PLAYER").set_index("PLAYER")["PLAYER_ID"]

    labeled = style.loc[style.index.isin(dff.index)]
    clf = LogisticRegression(max_iter=2000)
    y = dff[labeled.index]
    # Honest accuracy: cross-validated, scored only among each player's own coarse-group choices
    proba = cross_val_predict(clf, labeled, y, cv=5, method="predict_proba")
    classes = sorted(y.unique())
    hits = total = 0
    for player, row in zip(labeled.index, proba):
        choices = COARSE_CHOICES.get(coarse.get(ids.get(player)))
        if choices and len(choices) > 1:
            pick = max(choices, key=lambda c: row[classes.index(c)])
            hits += pick == y[player]
            total += 1
    print(f"Position classifier: {hits}/{total} = {hits / max(total, 1):.0%} correct within the NBA coarse group "
          f"(cross-validated on DFF-labeled players)")
    clf.fit(labeled, y)

    out = {}
    for player, pid in ids.items():
        if player in dff.index:
            out[player] = (dff[player], "dff")
            continue
        choices = COARSE_CHOICES.get(coarse.get(pid))
        if not choices:
            out[player] = (None, None)
        elif len(choices) == 1 or player not in style.index:
            out[player] = (choices[0], "nba_coarse")
        else:
            row = clf.predict_proba(style.loc[[player]])[0]
            out[player] = (max(choices, key=lambda c: row[list(clf.classes_).index(c)]), "nba_coarse")
    return out


def main():
    box = pd.concat([load(f"data/box_scores/{s}.parquet").assign(SEASON=s) for s in SEASONS], ignore_index=True)
    box["D"] = pd.to_datetime(box["GAME_DATE"]).dt.normalize()
    box["TEAM"] = box["TEAM_ABBREVIATION"]
    # Every box-score row counts toward the averages, 0-minute rows included, as in minutes-projection's
    # serving features; PLAYED below still means minutes > 0
    app = box.sort_values(["PLAYER", "D"]).copy()

    # Team-game index within each team-season
    tg = (box[["SEASON", "TEAM", "D", "GAME_ID", "MATCHUP"]].drop_duplicates(["TEAM", "GAME_ID"])
          .sort_values(["TEAM", "D"]).reset_index(drop=True))
    tg["TEAM_GAME_NO"] = tg.groupby(["SEASON", "TEAM"]).cumcount()
    tg["IS_HOME"] = tg["MATCHUP"].str.contains(" vs. ").astype(int)
    tg["OPP"] = tg["MATCHUP"].str[-3:]
    app = app.merge(tg[["TEAM", "GAME_ID", "TEAM_GAME_NO"]], on=["TEAM", "GAME_ID"], how="left")

    # Each appearance's post-game state = the pre-game state for the player's next game
    g = app.groupby(["PLAYER", "SEASON"])
    app["S_MIN"] = g["MIN"].transform(lambda s: s.expanding().mean())
    app["L7_MIN"] = g["MIN"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    app["S_FP"] = g["FP"].transform(lambda s: s.expanding().mean())
    app["L7_FP"] = g["FP"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    app["L3_FP"] = g["FP"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    app["USG_EVENTS"] = app["FGA"] + 0.44 * app["FTA"] + app["TOV"]
    app["S_USG"] = (app.groupby(["PLAYER", "SEASON"])["USG_EVENTS"].cumsum()
                    / app.groupby(["PLAYER", "SEASON"])["MIN"].cumsum().replace(0, np.nan))
    app["GP"] = g.cumcount() + 1
    prior = app["Games_Played_Career"].fillna(0)
    app["C_MIN"] = (app["Career_MIN_Avg"].fillna(0) * prior + app["MIN"]) / (prior + 1)
    app["C_FP"] = (app["Career_FP_Avg"].fillna(0) * prior + app["FP"]) / (prior + 1)
    app["CG"] = prior + 1
    state = app.rename(columns={"MIN": "PREV_MIN", "D": "LAST_DATE", "TEAM": "LAST_TEAM", "SEASON": "LAST_SEASON",
                                "TEAM_GAME_NO": "LAST_TEAM_GAME_NO"})[
        ["PLAYER", "LAST_DATE", "LAST_TEAM", "LAST_SEASON", "LAST_TEAM_GAME_NO", "S_MIN", "L7_MIN", "PREV_MIN",
         "S_FP", "L7_FP", "L3_FP", "GP", "C_MIN", "C_FP", "CG", "S_USG"]]

    # Candidate rows: every team-game x every player who appeared for that team that season
    roster = app[["SEASON", "TEAM", "PLAYER"]].drop_duplicates()
    cand = tg.merge(roster, on=["SEASON", "TEAM"])
    cand = pd.merge_asof(cand.sort_values("D"), state.sort_values("LAST_DATE"), left_on="D", right_on="LAST_DATE",
                         by="PLAYER", allow_exact_matches=False)
    # Keep him only while his latest appearance this season was for this team
    cand = cand[(cand["LAST_TEAM"] == cand["TEAM"]) & (cand["LAST_SEASON"] == cand["SEASON"])].copy()
    cand["TEAM_GAMES_MISSED"] = cand["TEAM_GAME_NO"] - cand["LAST_TEAM_GAME_NO"] - 1
    cand["DAYS_OFF"] = (cand["D"] - cand["LAST_DATE"]).dt.days

    acts = box.groupby(["D", "PLAYER"]).agg(ACT_MIN=("MIN", "sum"), ACT_FP=("FP", "sum")).reset_index()
    cand = cand.merge(acts, on=["D", "PLAYER"], how="left")
    cand[["ACT_MIN", "ACT_FP"]] = cand[["ACT_MIN", "ACT_FP"]].fillna(0)
    cand["PLAYED"] = cand["ACT_MIN"] > 0

    cand["FC_MIN"] = formula_c(cand["GP"], cand["S_MIN"], cand["L7_MIN"], cand["PREV_MIN"], cand["TEAM_GAMES_MISSED"])

    # Minutes freed by absent rotation teammates (realized absences stand in for the injury report)
    rot_out = (~cand["PLAYED"]) & (cand["S_MIN"] >= ROTATION_MIN)
    new = rot_out & (cand["TEAM_GAMES_MISSED"] == 0)
    recent = rot_out & (cand["TEAM_GAMES_MISSED"] <= 10)
    team_key = ["TEAM", "GAME_ID"]
    cand["FREED_NEW_MIN"] = cand.assign(x=np.where(new, cand["S_MIN"], 0)).groupby(team_key)["x"].transform("sum")
    cand["FREED_ALL_MIN"] = cand.assign(x=np.where(recent, cand["S_MIN"], 0)).groupby(team_key)["x"].transform("sum")
    # A player's own absence is not freed minutes for himself
    cand["FREED_NEW_MIN"] -= np.where(new, cand["S_MIN"], 0)
    cand["FREED_ALL_MIN"] -= np.where(recent, cand["S_MIN"], 0)

    positions = player_positions(box)
    cand["POSITION"] = cand["PLAYER"].map(lambda p: positions.get(p, (None, None))[0])
    cand["POS_SOURCE"] = cand["PLAYER"].map(lambda p: positions.get(p, (None, None))[1])

    cols = ["SEASON", "D", "GAME_ID", "TEAM", "OPP", "IS_HOME", "TEAM_GAME_NO", "PLAYER", "POSITION", "POS_SOURCE",
            "GP", "S_MIN", "L7_MIN",
            "PREV_MIN", "S_FP", "L7_FP", "L3_FP", "C_MIN", "C_FP", "CG", "S_USG", "TEAM_GAMES_MISSED", "DAYS_OFF", "FC_MIN",
            "FREED_NEW_MIN", "FREED_ALL_MIN", "ACT_MIN", "ACT_FP", "PLAYED"]
    cand = cand[cols].sort_values(["D", "TEAM", "PLAYER"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(OUT, index=False)

    print(f"Wrote {OUT} - {len(cand):,} rows")
    print(cand.groupby("SEASON").agg(rows=("PLAYER", "size"), games=("GAME_ID", "nunique"),
                                     played=("PLAYED", "sum"), dnp_rate=("PLAYED", lambda s: 1 - s.mean())).to_string())


if __name__ == "__main__":
    main()
