"""
Game betting lines from ESPN's free API (no key, no proxy): the spread and total for each game, from every pre-game
sportsbook ESPN lists ("Live Odds" in-game books skipped), median over books. Shared by minutes-projection (tonight's
lines, at run time) and scripts/fetch_game_lines.py (history). Backtest and training use the closing line (2023-26;
2022-23 has only one line); production uses the line at run time, about 17 min before the first tip.

Team-game rows: D, TEAM, OPP, TEAM_MARGIN (expected margin for TEAM = minus his spread), TOTAL, BOOKS.
"""

import json
import time
import urllib.request

import pandas as pd

ESPN_TO_OURS = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={}"
ODDS = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{0}/competitions/{0}/odds?limit=100"


def get(url, attempts=3, timeout=20):
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url.replace("http://", "https://"), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def tonight(day):
    """Team-game rows for `day` (a date) with the current pre-game line; games without a line are left out."""
    rows = []
    for e in get(SCOREBOARD.format(day.strftime("%Y%m%d"))).get("events", []):
        if e.get("season", {}).get("type") != 2:
            continue
        comp = {t["homeAway"]: t for t in e["competitions"][0]["competitors"]}
        home = ESPN_TO_OURS.get(comp["home"]["team"]["abbreviation"], comp["home"]["team"]["abbreviation"])
        away = ESPN_TO_OURS.get(comp["away"]["team"]["abbreviation"], comp["away"]["team"]["abbreviation"])
        books = [i for i in get(ODDS.format(e["id"])).get("items", [])
                 if "live" not in i.get("provider", {}).get("name", "").lower() and i.get("spread") is not None]
        if not books:
            continue
        spread = pd.Series([float(i["spread"]) for i in books]).median()
        totals = pd.Series([i.get("overUnder") for i in books], dtype=float).dropna()
        total = totals.median() if len(totals) else None
        for team, opp, margin in ((home, away, -spread), (away, home, spread)):
            rows.append({"D": pd.Timestamp(day), "TEAM": team, "OPP": opp, "TEAM_MARGIN": margin, "TOTAL": total,
                         "BOOKS": len(books)})
    return pd.DataFrame(rows, columns=["D", "TEAM", "OPP", "TEAM_MARGIN", "TOTAL", "BOOKS"])


def add_lines(rows, lines):
    """TEAM_MARGIN (and TOTAL) from team-game lines, joined on date and team."""
    lines = lines.assign(D=pd.to_datetime(lines["D"]).dt.normalize().astype("datetime64[ns]"))
    rows = rows.drop(columns=[c for c in ("TEAM_MARGIN", "TOTAL") if c in rows.columns])
    return rows.merge(lines[["D", "TEAM", "TEAM_MARGIN", "TOTAL"]].drop_duplicates(["D", "TEAM"]), on=["D", "TEAM"],
                      how="left")
