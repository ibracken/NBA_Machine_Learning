"""
Opening-night check (roadmap step 3): did every pipeline step run and write what it should for one game day?
Read-only on S3, CloudWatch Logs and alarms.

Usage: python scripts/opening_night_check.py [YYYY-MM-DD]   (default: today, US/Eastern)
Run it after the pipeline finishes (the lineup email arrives about 17 minutes before the first tip).
"""

import io
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import pandas as pd
import pytz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lambda" / "shared"))
from player_names import match_names  # noqa: E402

BUCKET = "nba-prediction-ibracken"
ET = pytz.timezone("US/Eastern")
s3 = boto3.client("s3")
logs = boto3.client("logs", region_name="us-east-1")
cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")
results = []


def report(ok, item, detail):
    status = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
    results.append(ok)
    print(f"[{status}] {item}: {detail}")


def log_messages(function, pattern, day):
    """Messages from one Lambda's logs between the day's start and 6am the next day (ET)."""
    start = ET.localize(datetime.combine(day, datetime.min.time()))
    end = start + timedelta(hours=30)
    out = []
    pages = logs.get_paginator("filter_log_events").paginate(
        logGroupName=f"/aws/lambda/{function}", startTime=int(start.timestamp() * 1000),
        endTime=int(end.timestamp() * 1000), filterPattern=pattern)
    for page in pages:
        out += [e["message"].strip() for e in page["events"]]
    return out


def modified_on(key, day):
    try:
        ts = s3.head_object(Bucket=BUCKET, Key=key)["LastModified"].astimezone(ET)
    except s3.exceptions.ClientError:
        return False, "missing"
    return ts.date() == day, f"last written {ts:%Y-%m-%d %H:%M} ET"


def load_parquet(key):
    return pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()))


def load_json(key):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None


def main():
    day = (datetime.strptime(sys.argv[1], "%Y-%m-%d").date() if len(sys.argv) > 1
           else datetime.now(ET).date())
    print(f"Opening-night check for {day}\n")

    # 1. game-scheduler scheduled the night
    created = log_messages("game-scheduler", '"Created rule"', day)
    skipped = log_messages("game-scheduler", '?"Preseason" ?"No regular-season games"', day)
    report(bool(created), "game-scheduler", f"{len(created)} rules created" if created else
           f"no rules created; {skipped[-1][-120:] if skipped else 'no skip message either'}")

    # 2. DFF returned players
    extracted = log_messages("daily-predictions", '"Extracted"', day)
    counts = [int(m) for msg in extracted for m in re.findall(r"Extracted (\d+) players", msg)]
    report(bool(counts) and max(counts) > 0, "DFF slate",
           f"Extracted {counts[-1]} players" if counts else "no 'Extracted' log line")

    # 3. schedule files refreshed
    for key in ("data/schedule/game_dates.json", "data/schedule/current.parquet"):
        ok, detail = modified_on(key, day)
        report(ok, key, detail)

    # 4. injury scraper
    found = log_messages("injury-scraper", '"Found latest NBA official report"', day)
    report(bool(found), "injury report found", found[-1][-110:] if found else
           "no report found; the season page may still list no links (see roadmap step 1)")
    for key in ("data/injuries/current.parquet", "data/injuries/report_statuses.parquet"):
        ok, detail = modified_on(key, day)
        report(ok, key, detail)
    try:
        statuses = load_parquet("data/injuries/report_statuses.parquet")
        if "REPORT_DATE" in statuses.columns:
            report(str(statuses["REPORT_DATE"].max()) == str(day), "report date", f"{statuses['REPORT_DATE'].max()}")
        known = set()
        for season_key in ("data/box_scores/current.parquet",):
            known |= set(load_parquet(season_key)["PLAYER"])
        mapping, unmatched = match_names(statuses["PLAYER"], known, known)
        odd = [n for n in unmatched if re.search(r"\b(i|ii|iid|l)\b", n)]
        report(not odd, "injury names", f"{len(statuses)} listed, {len(unmatched)} not in box scores "
               f"(rookies/two-ways expected){'; looks mangled: ' + str(odd[:5]) if odd else ''}")
        hyph = [n for n in statuses["PLAYER"] if "-" in n][:3]
        report(None, "hyphenated names parsed", hyph or "none listed tonight")
    except Exception as e:
        report(False, "report_statuses.parquet", f"could not read: {e}")

    # 5. lineup email: paused header + DFF lineup for tonight
    ran = log_messages("minutes-projection", '"START RequestId"', day)
    paused = log_messages("minutes-projection", '"In-house projections paused"', day)
    report(bool(ran), "minutes-projection", "did not run" if not ran else
           paused[-1][-110:] if paused else "ran, not paused (in-house lineups built)")
    dff = load_parquet("model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet")
    tonight = dff[pd.to_datetime(dff["DATE"]).dt.date == day]
    report(len(tonight) == 8, "DFF lineup saved", f"{len(tonight)} players for {day}")

    # 6. research
    research = load_json(f"llm/research/{day}.json")
    if research:
        games = research.get("games", [])
        cites = [c for g in games for c in g.get("citations", [])]
        x_links = sum("x.com" in (c.get("url") or "") or "twitter.com" in (c.get("url") or "") for c in cites)
        report(bool(games), "research", f"{len(games)} games, {len(cites)} cited snippets, {x_links} from X")
    else:
        report(False, "research", f"llm/research/{day}.json missing")

    # 7. AI cost
    usage = load_json(f"llm/usage/{day}.json")
    if usage:
        runs = usage.get("runs", [])
        cost = sum(r.get("cost_usd") or 0 for r in runs)
        searches = sum(r.get("web_search_requests", 0) for r in runs)
        report(None, "AI cost", f"${cost:.2f} in tokens over {len(runs)} calls, plus {searches} web searches "
               "(billed separately; check the Anthropic console for their cost)")
    else:
        report(False, "AI cost", f"llm/usage/{day}.json missing")

    # 8. alarms
    alarms = cloudwatch.describe_alarms(AlarmNamePrefix="nba-lambda-errors-")["MetricAlarms"]
    firing = [a["AlarmName"] for a in alarms if a["StateValue"] == "ALARM"]
    report(not firing, "error alarms", f"firing: {firing}" if firing else f"{len(alarms)} alarms, none firing")

    fails = sum(r is False for r in results)
    print(f"\n{fails} failed of {sum(r is not None for r in results)} checks")


if __name__ == "__main__":
    main()
