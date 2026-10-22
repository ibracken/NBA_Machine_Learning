"""
Who started each game, 2022-23 onward: NBA Stats BoxScoreTraditionalV3 (a starter has a position, the bench has
none). One request per game ID in our box scores, through PROXY_URL; resumable (games already fetched are
skipped). Read-only on S3. Writes data/replay/starters.parquet: GAME_ID, PLAYER_ID, TEAM, START_POSITION, STARTER.

Usage: python scripts/fetch_starters.py [minutes]   (needs nba_api: pip install nba_api)
  minutes: stop and save after this long (rerun to continue)
"""

import io
import os
import sys
import time
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv
from nba_api.stats.endpoints import boxscoretraditionalv3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "replay" / "starters.parquet"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
DELAY, RETRIES, SAVE_EVERY, TIMEOUT = 0.3, 3, 100, 10


def game_ids():
    s3 = boto3.client("s3")
    ids = []
    for season in SEASONS:
        body = s3.get_object(Bucket="nba-prediction-ibracken", Key=f"data/box_scores/{season}.parquet")["Body"].read()
        ids += sorted(pd.read_parquet(io.BytesIO(body), columns=["GAME_ID"]).GAME_ID.astype(str).unique())
    return ids


def fetch(gid, proxy):
    for attempt in range(RETRIES):
        try:
            df = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=gid, proxy=proxy, timeout=TIMEOUT).player_stats.get_data_frame()
            pos = df["position"].fillna("").astype(str).str.strip()
            return pd.DataFrame({"GAME_ID": gid, "PLAYER_ID": df["personId"].astype(int), "TEAM": df["teamTricode"],
                                 "START_POSITION": pos, "STARTER": pos != ""})
        except Exception as e:
            wait = 2 * (attempt + 1)   # failed games are retried on the next run
            print(f"  {gid} attempt {attempt + 1} failed ({type(e).__name__}); retry in {wait}s", flush=True)
            time.sleep(wait)
    return None


def main():
    load_dotenv(ROOT / "lambda" / ".env")
    load_dotenv(ROOT / ".env")
    proxy = os.getenv("PROXY_URL")
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    have = set(done.GAME_ID) if not done.empty else set()
    todo = [g for g in game_ids() if g not in have]
    print(f"{len(have)} games already fetched, {len(todo)} to go", flush=True)
    parts, failed = [done] if not done.empty else [], []
    deadline = time.time() + 60 * float(sys.argv[1]) if len(sys.argv) > 1 else float("inf")
    for i, gid in enumerate(todo, 1):
        if time.time() > deadline:
            pd.concat(parts, ignore_index=True).to_parquet(OUT, index=False)
            print(f"time limit: stopped after {i - 1} games this run; rerun to continue", flush=True)
            return
        got = fetch(gid, proxy)
        if got is None:
            failed.append(gid)
        else:
            parts.append(got)
        if i % SAVE_EVERY == 0 or i == len(todo):
            pd.concat(parts, ignore_index=True).to_parquet(OUT, index=False)
            print(f"{i}/{len(todo)} saved; failed so far {len(failed)}", flush=True)
        time.sleep(DELAY)
    out = pd.read_parquet(OUT)
    starters = out.groupby(["GAME_ID", "TEAM"]).STARTER.sum()
    print(f"done: {out.GAME_ID.nunique()} games; starters per team-game {starters.value_counts().to_dict()}; failed {failed}")


if __name__ == "__main__":
    main()
