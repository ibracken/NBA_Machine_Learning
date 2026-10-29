"""
Box scores from ESPN's free API (no key, no proxy): the fallback for days stats.nba.com does not answer (it stopped
answering scripts 2026-10-28). Rows are converted to the NBA game-log schema the whole pipeline is built on:
  - GAME_ID: the NBA id, from the season schedule (date, team, opponent); a game not on it is not added.
  - PLAYER_ID / PLAYER: NBA's, from an ESPN-to-NBA player map built by pairing identical stat lines in games both
    sources have (build_player_map). ESPN players not on the map: an exact name in NBA's box scores (when that name
    is one NBA player), else player_names.match_names among his team's NBA players; a player with no NBA box score
    keeps ESPN's name and the id -ESPN_ID.
  - only completed regular-season games, and only players who got into the game (NBA's log has no DNP rows).
Validated against NBA's own box scores for 2025-26 and 2026-27 (scripts/validate_espn_box.py).
Used by box-score-scraper and scripts/validate_espn_box.py.
"""

import numpy as np
import pandas as pd
from unidecode import unidecode

from game_lines import ESPN_TO_OURS, get
from player_names import key, match_names

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={}"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={}"
COUNTS = {"MIN": "MIN", "PTS": "PTS", "REB": "REB", "AST": "AST", "TO": "TOV", "STL": "STL", "BLK": "BLK",
          "OREB": "OREB", "DREB": "DREB", "PF": "PF", "+/-": "PLUS_MINUS"}
SPLITS = {"FG": ("FGM", "FGA"), "3PT": ("FG3M", "FG3A"), "FT": ("FTM", "FTA")}
STAT_COLS = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV",
             "PF", "PTS", "PLUS_MINUS"]
NBA_COLS = ["SEASON_ID", "PLAYER_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PLAYER TEAM", "GAME_ID", "GAME_DATE",
            "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB",
            "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS", "VIDEO_AVAILABLE", "PLAYER"]


def team(abbr):
    return ESPN_TO_OURS.get(abbr, abbr)


def events(day):
    """Regular-season games on `day` (a date, ET as ESPN's scoreboard uses): EVENT, HOME, AWAY, COMPLETED."""
    rows = []
    for e in get(SCOREBOARD.format(day.strftime("%Y%m%d"))).get("events", []):
        if e.get("season", {}).get("type") != 2:
            continue
        comp = {t["homeAway"]: t for t in e["competitions"][0]["competitors"]}
        rows.append({"EVENT": e["id"], "HOME": team(comp["home"]["team"]["abbreviation"]),
                     "AWAY": team(comp["away"]["team"]["abbreviation"]),
                     "COMPLETED": bool(e["status"]["type"].get("completed"))})
    return rows


def player_lines(summary):
    """
    One row per player who got into the game: ESPN_ID, ESPN_NAME, TEAM, OPP, IS_HOME, WL and the counting stats.
    Returns (rows, gaps). `gaps` lists data ESPN lacks for this game (a player who played but has no stats, or player
    points not adding up to the team score); the rows ESPN has are still correct. Malformed data raises.
    """
    comp = summary["header"]["competitions"][0]
    state = comp.get("status", {}).get("type", {}).get("name")
    if state != "STATUS_FINAL":
        raise ValueError(f"event {summary['header'].get('id')} is not final ({state})")
    sides = {team(c["team"]["abbreviation"]): c for c in comp["competitors"]}
    if len(sides) != 2:
        raise ValueError(f"event {summary['header'].get('id')}: {len(sides)} teams")
    rows, gaps = [], []
    for block in summary["boxscore"]["players"]:
        abbr = team(block["team"]["abbreviation"])
        opp = next(t for t in sides if t != abbr)
        start = len(rows)
        if len(block["statistics"]) != 1:
            raise ValueError(f"{abbr}: {len(block['statistics'])} stat blocks")
        st = block["statistics"][0]
        labels = st["labels"]
        missing = (set(COUNTS) | set(SPLITS)) - set(labels)
        if missing:
            raise ValueError(f"{abbr}: ESPN labels missing {sorted(missing)}")
        for a in st["athletes"]:
            if a.get("didNotPlay") or not a.get("stats"):
                continue
            who = a.get("athlete", {}).get("displayName") or a.get("athlete", {}).get("shortName")
            if len(a["stats"]) != len(labels):
                raise ValueError(f"{abbr} {who}: {len(a['stats'])} stats for {len(labels)} labels")
            v = dict(zip(labels, a["stats"]))
            # A player shown as playing with no ESPN athlete record has lost stats (Lachlan Olbrich in 22 games of
            # 2025-26: NBA had him playing in every one). Minutes "--" with every stat 0 is a player who never got
            # in (NBA has none of 12 such 2025-26 cases, and 2 at 0 minutes).
            if "id" not in a.get("athlete", {}):
                gaps.append(f"{abbr}: {who} played but ESPN has no record of his stats")
                continue
            if v["MIN"] == "--":
                if all(x in ("--", "0", "0-0", "+0", "-0") for x in a["stats"]):
                    continue
                raise ValueError(f"{abbr} {who}: stats with minutes '--'")
            row = {"ESPN_ID": int(a["athlete"]["id"]), "ESPN_NAME": a["athlete"]["displayName"], "TEAM": abbr,
                   "OPP": opp, "IS_HOME": sides[abbr]["homeAway"] == "home",
                   "WL": "W" if sides[abbr].get("winner") else "L", "STARTER": bool(a.get("starter"))}
            for label, col in COUNTS.items():
                row[col] = int(v[label].replace("+", ""))
            for label, (made, att) in SPLITS.items():
                m, n = v[label].split("-")
                row[made], row[att] = int(m), int(n)
            rows.append(row)
        scored = sum(r["PTS"] for r in rows[start:])
        if scored != int(sides[abbr]["score"]):
            gaps.append(f"{abbr}: player points sum to {scored}, team scored {sides[abbr]['score']}")
    return pd.DataFrame(rows), gaps


def check_lines(lines):
    """Internal consistency ESPN's lines must have; raises on any violation."""
    bad = lines[(lines.REB != lines.OREB + lines.DREB) | (lines.PTS != 2 * lines.FGM + lines.FG3M + lines.FTM)
                | (lines.FGM > lines.FGA) | (lines.FG3M > lines.FG3A) | (lines.FTM > lines.FTA)
                | (lines.FG3M > lines.FGM) | (lines.MIN < 0) | (lines.MIN > 70)]
    if len(bad):
        raise ValueError(f"inconsistent ESPN stat lines: {bad[['ESPN_NAME', 'TEAM']].values.tolist()}")
    if lines.duplicated(["EVENT", "ESPN_ID"]).any():
        raise ValueError("an ESPN player appears twice in one game")


def build_player_map(lines, nba):
    """
    ESPN_ID -> NBA PLAYER_ID and PLAYER, from games both have. `lines` need GAME_ID (mapped from the schedule);
    `nba` is NBA box-score rows. A pair counts only when the team-game has exactly one NBA row with the same full stat
    line. Returns (map, report): an ESPN id paired with two NBA ids (or the reverse) is left out and reported.
    """
    line = ["GAME_ID", "TEAM"] + STAT_COLS
    e = lines.assign(GAME_ID=lines.GAME_ID.astype(str))
    n = nba.rename(columns={"TEAM_ABBREVIATION": "TEAM"}).assign(GAME_ID=nba.GAME_ID.astype(str))
    n = n[~n.duplicated(line, keep=False)]
    e = e[~e.duplicated(line, keep=False)]
    pairs = e.merge(n[line + ["PLAYER_ID", "PLAYER"]], on=line)
    per = pairs.groupby("ESPN_ID").PLAYER_ID.nunique()
    rev = pairs.groupby("PLAYER_ID").ESPN_ID.nunique()
    clean = pairs[pairs.ESPN_ID.isin(per[per == 1].index) & pairs.PLAYER_ID.isin(rev[rev == 1].index)]
    out = (clean.groupby("ESPN_ID").agg(PLAYER_ID=("PLAYER_ID", "first"), PLAYER=("PLAYER", "last"),
                                         ESPN_NAME=("ESPN_NAME", "last"), GAMES=("GAME_ID", "nunique"))
           .reset_index())
    report = {"pairs": len(pairs), "espn_ids": int(len(per)), "conflicting_espn_ids": per[per > 1].index.tolist(),
              "conflicting_nba_ids": rev[rev > 1].index.tolist()}
    return out, report


def to_nba_rows(lines, player_map, nba_history, season):
    """
    NBA-schema rows from ESPN lines that already carry GAME_ID and GAME_DATE. `nba_history`: NBA box-score rows
    (for team ids/names and for matching unmapped players by name). Returns (rows, unmapped report).
    """
    teams = nba_history.drop_duplicates("TEAM_ABBREVIATION", keep="last").set_index("TEAM_ABBREVIATION")
    rows = lines.merge(player_map[["ESPN_ID", "PLAYER_ID", "PLAYER"]], on="ESPN_ID", how="left")
    report = []
    # Not on the map (debut on ESPN's side, e.g. back from injury or traded): first an exact name in any NBA
    # box score, if that name is one NBA player; then the looser rules, only among his team's NBA players
    ids_by_key = nba_history.groupby(nba_history.PLAYER.map(key)).PLAYER_ID.unique()
    for i in rows.index[rows.PLAYER_ID.isna()]:
        hit = ids_by_key.get(key(rows.at[i, "ESPN_NAME"]))
        if hit is not None and len(hit) == 1:
            nba_name = nba_history.loc[nba_history.PLAYER_ID == hit[0], "PLAYER"].iloc[-1]
            rows.loc[i, ["PLAYER_ID", "PLAYER"]] = [hit[0], nba_name]
            report.append((rows.at[i, "ESPN_NAME"], rows.at[i, "TEAM"], f"exact name: {nba_name} ({hit[0]})"))
    todo = rows[rows.PLAYER_ID.isna()]
    if len(todo):
        known = set(nba_history.PLAYER)
        for abbr, grp in todo.groupby("TEAM"):
            hist = nba_history[nba_history.TEAM_ABBREVIATION == abbr]
            mapping, _ = match_names(grp.ESPN_NAME.map(normalize).tolist(), hist.PLAYER.unique(), known)
            ids = hist.drop_duplicates("PLAYER", keep="last").set_index("PLAYER").PLAYER_ID
            for i, r in grp.iterrows():
                hit = mapping.get(normalize(r.ESPN_NAME))
                if hit is not None:
                    rows.loc[i, ["PLAYER_ID", "PLAYER"]] = [ids[hit], hit]
                    report.append((r.ESPN_NAME, abbr, f"name-matched to {hit}"))
                else:
                    rows.loc[i, ["PLAYER_ID", "PLAYER"]] = [-r.ESPN_ID, normalize(r.ESPN_NAME)]
                    report.append((r.ESPN_NAME, abbr, "no NBA history: ESPN name kept"))
    rows["PLAYER_ID"] = rows.PLAYER_ID.astype("int64")
    rows["SEASON_ID"] = "2" + season[:4]
    rows["TEAM_ID"] = rows.TEAM.map(teams.TEAM_ID)
    rows["PLAYER TEAM"] = rows.TEAM.map(teams["PLAYER TEAM"])
    if rows.TEAM_ID.isna().any():
        raise ValueError(f"teams with no NBA id: {sorted(rows[rows.TEAM_ID.isna()].TEAM.unique())}")
    rows["TEAM_ID"] = rows.TEAM_ID.astype("int64")
    rows["TEAM_ABBREVIATION"] = rows.TEAM
    rows["MATCHUP"] = rows.TEAM + rows.IS_HOME.map({True: " vs. ", False: " @ "}) + rows.OPP
    for pct, (m, a) in {"FG_PCT": ("FGM", "FGA"), "FG3_PCT": ("FG3M", "FG3A"), "FT_PCT": ("FTM", "FTA")}.items():
        # NBA rounds half up (9/16 = 0.563)
        rows[pct] = np.floor(rows[m] / rows[a].where(rows[a] > 0) * 1000 + 0.5 + 1e-9) / 1000
    rows["VIDEO_AVAILABLE"] = 0
    rows["GAME_DATE"] = pd.to_datetime(rows.GAME_DATE)
    dup = rows[rows.duplicated(["PLAYER", "GAME_ID"], keep=False)]
    if len(dup):
        raise ValueError(f"two ESPN players mapped to one NBA player in a game: "
                         f"{dup[['GAME_ID', 'TEAM', 'ESPN_ID', 'ESPN_NAME', 'PLAYER']].values.tolist()}")
    return rows[NBA_COLS], report


def normalize(name):
    """Same as box-score-scraper's normalize_name."""
    return unidecode(name.strip().lower())


def et_date(iso):
    return pd.Timestamp(iso).tz_convert("America/New_York").tz_localize(None).normalize()


def fill(existing, schedule, history, season, today, player_map):
    """
    NBA-schema rows (without FP) for this season's games before `today` (ET date) that `existing` lacks, from ESPN.
    `schedule`: NBA's (GAME_DATE, GAME_ID, TEAM, OPPONENT, IS_HOME), which also gives home/away (ESPN and NBA disagree
    on neutral-site games). Returns (rows, report): report['gaps'] = {GAME_ID: what ESPN lacks}, report['not_found'] =
    due NBA games with no ESPN game, report['not_final'] = ESPN shows them not final (postponed),
    report['players'] = how players not on the map were resolved.
    """
    sch = schedule.assign(D=pd.to_datetime(schedule.GAME_DATE).dt.normalize(), GAME_ID=schedule.GAME_ID.astype(str))
    have = set(existing.GAME_ID.astype(str)) if len(existing) else set()
    due = sch[(sch.D < pd.Timestamp(today)) & ~sch.GAME_ID.isin(have)]
    report = {"espn_games": 0, "gaps": {}, "not_found": [], "not_final": [], "players": []}
    if due.empty:
        return pd.DataFrame(columns=NBA_COLS), report
    parts, gaps_by_event, pending = [], {}, set()
    for day in sorted(due.D.dt.date.unique()):
        for e in events(day):
            if not e["COMPLETED"]:   # postponed (the saved NBA schedule goes stale while NBA is down)
                pending.add((pd.Timestamp(day), e["HOME"], e["AWAY"]))
                continue
            s = get(SUMMARY.format(e["EVENT"]))
            lines, gaps = player_lines(s)
            parts.append(lines.assign(EVENT=e["EVENT"], D=et_date(s["header"]["competitions"][0]["date"])))
            gaps_by_event[e["EVENT"]] = gaps
    lines = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["D", "TEAM", "OPP", "EVENT"])
    key_cols = due[["D", "TEAM", "OPPONENT", "GAME_ID", "IS_HOME"]].rename(columns={"OPPONENT": "OPP", "IS_HOME": "SCHED_HOME"})
    lines = lines.merge(key_cols, on=["D", "TEAM", "OPP"], how="inner")
    missing = due[(due.IS_HOME == 1) & ~due.GAME_ID.isin(set(lines.GAME_ID))]
    late = missing.apply(lambda g: (g.D, g.TEAM, g.OPPONENT) in pending, axis=1) if len(missing) else pd.Series(dtype=bool)
    report["not_final"] = sorted(missing[late].GAME_ID) if len(missing) else []
    report["not_found"] = sorted(missing[~late].GAME_ID) if len(missing) else []
    if lines.empty:
        return pd.DataFrame(columns=NBA_COLS), report
    if lines.groupby("GAME_ID").EVENT.nunique().max() > 1:
        raise ValueError("an NBA game matched two ESPN games")
    if lines.groupby("GAME_ID").TEAM.nunique().min() < 2:
        raise ValueError("an ESPN game matched only one team of an NBA game")
    report["espn_games"] = int(lines.GAME_ID.nunique())
    for ev, gid in lines.drop_duplicates("EVENT").set_index("EVENT").GAME_ID.items():
        if gaps_by_event.get(ev):
            report["gaps"][gid] = gaps_by_event[ev]
    lines["IS_HOME"] = lines.SCHED_HOME == 1
    check_lines(lines)
    rows, report["players"] = to_nba_rows(lines.assign(GAME_DATE=lines.D), player_map, history, season)
    return rows, report
