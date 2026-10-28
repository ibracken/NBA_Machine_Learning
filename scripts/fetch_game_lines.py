"""
Game betting lines (spread, total, moneyline) for every regular-season game in our box scores, from ESPN's free
odds API. Every pre-game sportsbook ESPN lists, with its opening, closing and latest ("current") line where present;
"Live Odds" books (in-game lines) are skipped. Raw values only. Resumable by date. Read-only on S3.
Writes data/replay/game_lines.parquet: one row per game x book. Spreads are from the home team's side (negative =
home favored).

Usage: python scripts/fetch_game_lines.py [minutes]   (stop and save after this long; rerun to continue)
"""

import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import boto3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "replay" / "game_lines.parquet"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
ESPN_TO_OURS = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
DELAY = 0.15


def get(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url.replace("http://", "https://"), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def num(d, *path):
    for p in path:
        d = d.get(p) if isinstance(d, dict) else None
    if d is None or d == "":
        return None
    text = str(d).strip()
    if text.upper() in ("PK", "PICK", "EVEN"):
        return 0.0 if text.upper() != "EVEN" else 100.0   # pick'em spread / even-money price
    try:
        return float(text.lstrip("ou+"))
    except ValueError:
        return None


def season_dates():
    s3 = boto3.client("s3")
    out = []
    for season in SEASONS:
        body = s3.get_object(Bucket="nba-prediction-ibracken", Key=f"data/box_scores/{season}.parquet")["Body"].read()
        dates = pd.to_datetime(pd.read_parquet(io.BytesIO(body), columns=["GAME_DATE"]).GAME_DATE).dt.strftime("%Y%m%d")
        out += [(season, d) for d in sorted(dates.unique())]
    return out


def lines_for(season, d):
    rows = []
    for e in get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}")["events"]:
        if e.get("season", {}).get("type") != 2:
            continue
        comp = {t["homeAway"]: t for t in e["competitions"][0]["competitors"]}
        base = {"SEASON": season, "GAME_DATE": pd.Timestamp(d), "EVENT_ID": e["id"], "TIP_UTC": e["date"],
                "HOME": ESPN_TO_OURS.get(comp["home"]["team"]["abbreviation"], comp["home"]["team"]["abbreviation"]),
                "AWAY": ESPN_TO_OURS.get(comp["away"]["team"]["abbreviation"], comp["away"]["team"]["abbreviation"]),
                "HOME_SCORE": num(comp["home"], "score"), "AWAY_SCORE": num(comp["away"], "score")}
        odds = get(f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{e['id']}/"
                   f"competitions/{e['id']}/odds?limit=100").get("items", [])
        time.sleep(DELAY)
        for it in odds:
            book = it.get("provider", {}).get("name", "")
            if "live" in book.lower():
                continue
            h, a = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
            rows.append({**base, "BOOK": book, "SPREAD_HOME": it.get("spread"), "TOTAL": it.get("overUnder"),
                         "OPEN_SPREAD_HOME": num(h, "open", "pointSpread", "american"),
                         "CLOSE_SPREAD_HOME": num(h, "close", "pointSpread", "american"),
                         "CURRENT_SPREAD_HOME": num(h, "current", "pointSpread", "american"),
                         "OPEN_TOTAL": num(it, "open", "total", "american"),
                         "CLOSE_TOTAL": num(it, "close", "total", "american"),
                         "CURRENT_TOTAL": num(it, "current", "total", "american"),
                         "ML_HOME": h.get("moneyLine"), "ML_AWAY": a.get("moneyLine")})
        if not odds:
            rows.append({**base, "BOOK": None})
    return rows


def main():
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    have = set(done.GAME_DATE.dt.strftime("%Y%m%d")) if not done.empty else set()
    todo = [(s, d) for s, d in season_dates() if d not in have]
    print(f"{len(have)} dates done, {len(todo)} to go", flush=True)
    deadline = time.time() + 60 * float(sys.argv[1]) if len(sys.argv) > 1 else float("inf")
    parts, failed = ([done] if not done.empty else []), []
    for i, (season, d) in enumerate(todo, 1):
        if time.time() > deadline:
            break
        try:
            parts.append(pd.DataFrame(lines_for(season, d)))
        except Exception as e:
            failed.append(d)
            print(f"  {d} failed: {type(e).__name__}", flush=True)
        if i % 25 == 0:
            pd.concat(parts, ignore_index=True).to_parquet(OUT, index=False)
            print(f"{i}/{len(todo)} dates saved", flush=True)
        time.sleep(DELAY)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"saved {out.GAME_DATE.nunique()} dates, {out.EVENT_ID.nunique()} games, {len(out)} game-book rows; failed {failed}")


if __name__ == "__main__":
    main()
