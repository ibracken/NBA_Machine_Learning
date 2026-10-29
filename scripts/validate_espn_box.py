"""
Validate ESPN box scores (lambda/shared/espn_box.py, box-score-scraper's fallback) against NBA's own, and build the
ESPN-to-NBA player map the fallback uses.

  1. Download ESPN's game summaries for every date in NBA's saved box scores of the test seasons (cached in
     data/replay/espn_raw/, resumable).
  2. Parse them with the production code; map each ESPN game to its NBA game by (ET date, team, opponent).
  3. Out-of-sample check: build the player map from games before --cut only, convert later games with the
     production code (history = the 4 seasons before --cut, as the Lambda has), and compare with NBA's rows: rows
     on either side only, every column, NBA ids, DraftKings FP. Every missing NBA row must sit in a game the parser
     flagged as incomplete.
  4. Build the map from all games both have -> data/replay/espn_player_map.parquet; --publish-map also writes it to
     S3 (data/box_scores/espn_player_map.parquet).

Usage: python scripts/validate_espn_box.py [--cut 2026-01-01] [--publish-map]
"""
import argparse
import glob
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lambda" / "shared"))
import espn_box as eb  # noqa: E402

BUCKET = "nba-prediction-ibracken"
MAP_KEY = "data/box_scores/espn_player_map.parquet"
RAW = ROOT / "data" / "replay" / "espn_raw"
TEST_SEASONS = ["2025-26", "2026-27"]
HISTORY_SEASONS = ["2023-24", "2024-25"]
s3 = boto3.client("s3")


def load_box(season):
    b = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"data/box_scores/{season}.parquet")["Body"].read()))
    b["GAME_ID"] = b.GAME_ID.astype(str)
    b["D"] = pd.to_datetime(b.GAME_DATE).dt.normalize()
    return b


def download(dates):
    RAW.mkdir(parents=True, exist_ok=True)
    ev_path = RAW / "events.json"
    evs = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    for d in dates:
        if str(d) not in evs:
            evs[str(d)] = eb.events(d)
    ev_path.write_text(json.dumps(evs))
    todo = [e["EVENT"] for day in evs.values() for e in day if not (RAW / f"{e['EVENT']}.json").exists()]
    print(f"ESPN: {sum(len(v) for v in evs.values())} games on {len(dates)} dates; downloading {len(todo)}")

    def fetch(ev):
        s = eb.get(eb.SUMMARY.format(ev))
        (RAW / f"{ev}.json").write_text(json.dumps({"header": s["header"], "boxscore": s["boxscore"]}))

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(fetch, todo))
    return {e["EVENT"] for day in evs.values() for e in day}


def parse(events):
    lines, gap_games, skipped = [], {}, []
    for ev in sorted(events):
        s = json.loads((RAW / f"{ev}.json").read_text())
        comp = s["header"]["competitions"][0]
        state = comp.get("status", {}).get("type", {}).get("name")
        if state != "STATUS_FINAL":
            skipped.append((ev, state))
            continue
        l, gaps = eb.player_lines(s)   # malformed data raises: that fails the validation
        if gaps:
            gap_games[ev] = gaps
        lines.append(l.assign(EVENT=ev, D=eb.et_date(comp["date"])))
    lines = pd.concat(lines, ignore_index=True)
    eb.check_lines(lines)
    return lines, gap_games, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="2026-01-01", help="player map from games before this date; test on the rest")
    ap.add_argument("--publish-map", action="store_true", help=f"write the full map to s3://{BUCKET}/{MAP_KEY}")
    args = ap.parse_args()
    cut = pd.Timestamp(args.cut)
    pd.set_option("display.width", 250)

    nba = pd.concat([load_box(s) for s in TEST_SEASONS], ignore_index=True)
    history = pd.concat([load_box(s) for s in HISTORY_SEASONS] + [nba], ignore_index=True)
    events = download(sorted(set(nba.D.dt.date)))
    lines, gap_games, skipped = parse(events)
    print(f"Final ESPN games {lines.EVENT.nunique()}; not final (skipped): {skipped}")
    print(f"Games the parser flagged incomplete ({len(gap_games)}):")
    for ev, g in gap_games.items():
        print(f"  {ev}: {g}")

    # games: ESPN -> NBA by (ET date, team, opponent)
    tg = nba.drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION"])
    gmap = pd.DataFrame({"D": tg.D, "TEAM": tg.TEAM_ABBREVIATION, "OPP": tg.MATCHUP.str[-3:], "GAME_ID": tg.GAME_ID,
                         "NBA_HOME": tg.MATCHUP.str.contains("vs.")})
    lines = lines.merge(gmap, on=["D", "TEAM", "OPP"], how="left")
    per_game = lines.dropna(subset=["GAME_ID"]).groupby("GAME_ID").EVENT.nunique()
    stray = lines[lines.GAME_ID.isna()].drop_duplicates("EVENT")
    print(f"\nGAMES: NBA {nba.GAME_ID.nunique()}; with an ESPN game {per_game.size}; with 2+ {int((per_game > 1).sum())}; "
          f"ESPN games not in NBA's saved box scores {len(stray)} (dates {sorted(stray.D.dt.date.astype(str).unique())})")
    lines = lines[lines.GAME_ID.notna()].copy()
    lines["IS_HOME"] = lines.NBA_HOME   # production takes home/away from NBA's schedule

    # out-of-sample
    pm, rep = eb.build_player_map(lines[lines.D < cut], nba[nba.D < cut])
    print(f"\nPLAYER MAP from games before {cut.date()}: {len(pm)} ESPN ids; conflicts {rep['conflicting_espn_ids']} "
          f"{rep['conflicting_nba_ids']}; ESPN/NBA spellings differ for {int((pm.ESPN_NAME.map(eb.normalize).map(eb.key) != pm.PLAYER.map(eb.key)).sum())}")
    test = lines[lines.D >= cut]
    rows, report = eb.to_nba_rows(test.assign(GAME_DATE=test.D), pm, history[history.D < cut], "2025-26")
    rows["SEASON_ID"] = test.D.map(lambda d: "2" + str(d.year if d.month >= 10 else d.year - 1)).values
    how = pd.Series([r[2].split(":")[0] for r in report]).value_counts().to_dict()
    print(f"OUT-OF-SAMPLE (games from {cut.date()}): player-games not on the map {len(report)}, resolved by {how}")

    truth = nba[nba.D >= cut]
    m = truth.merge(rows, on=["GAME_ID", "PLAYER"], how="outer", suffixes=("_NBA", "_ESPN"), indicator=True)
    both = m[m._merge == "both"]
    nba_only, espn_only = m[m._merge == "left_only"], m[m._merge == "right_only"]
    gap_ids = set(lines[lines.EVENT.isin(gap_games)].GAME_ID)
    print(f"\nCOMPARISON ({truth.GAME_ID.nunique()} games): NBA rows {len(truth):,}; ESPN rows {len(rows):,}; "
          f"same game and player {len(both):,}")
    print(f"  ESPN rows with no NBA row (wrong or invented player): {len(espn_only)} {espn_only[['GAME_ID', 'PLAYER']].values.tolist()}")
    print(f"  NBA rows with no ESPN row: {len(nba_only)}; in games flagged incomplete {int(nba_only.GAME_ID.isin(gap_ids).sum())}; "
          f"others: {nba_only[~nba_only.GAME_ID.isin(gap_ids)][['GAME_ID', 'PLAYER', 'MIN_NBA', 'PTS_NBA']].values.tolist()}")
    ids = both[both.PLAYER_ID_NBA != both.PLAYER_ID_ESPN]
    print(f"  NBA id differs on {len(ids)} rows: placeholder ids (player with no NBA box score before the cut) "
          f"{int((ids.PLAYER_ID_ESPN < 0).sum())}, a different real id {int((ids.PLAYER_ID_ESPN > 0).sum())}")
    cols = eb.STAT_COLS + ["TEAM_ABBREVIATION", "TEAM_ID", "PLAYER TEAM", "MATCHUP", "WL", "FG_PCT", "FG3_PCT", "FT_PCT", "SEASON_ID"]
    mis = {}
    for c in cols:
        a, b = both[f"{c}_NBA"], both[f"{c}_ESPN"]
        n = int((~((a == b) | (a.isna() & b.isna()))).sum())
        if n:
            mis[c] = n
    print(f"  columns differing on matched rows: {mis}")
    for c in [c for c in eb.STAT_COLS if c in mis and c != "PLUS_MINUS"]:
        d = both[both[f"{c}_NBA"] != both[f"{c}_ESPN"]]
        print(f"    {c}: ESPN minus NBA {(d[f'{c}_ESPN'] - d[f'{c}_NBA']).value_counts().to_dict()}")

    def dk(d, s):
        n10 = sum((d[f"{c}{s}"] >= 10).astype(int) for c in ["PTS", "REB", "AST", "STL", "BLK"])
        return (d[f"PTS{s}"] + .5 * d[f"FG3M{s}"] + 1.25 * d[f"REB{s}"] + 1.5 * d[f"AST{s}"] + 2 * d[f"STL{s}"]
                + 2 * d[f"BLK{s}"] - .5 * d[f"TOV{s}"] + 1.5 * (n10 >= 2) + 1.5 * (n10 >= 3))
    fpd = dk(both, "_ESPN") - both.FP
    print(f"  DraftKings FP identical on {(fpd.abs() < 1e-9).mean():.3%} of matched rows; differences {fpd[fpd.abs() > 1e-9].round(2).tolist()}")

    full, rep = eb.build_player_map(lines, nba)
    print(f"\nFULL MAP: {len(full)} ESPN ids; conflicts {rep['conflicting_espn_ids']} {rep['conflicting_nba_ids']}; "
          f"ESPN ids that played but got no entry (resolved by name in production): "
          f"{lines[~lines.ESPN_ID.isin(full.ESPN_ID)].drop_duplicates('ESPN_ID')[['ESPN_NAME', 'TEAM']].values.tolist()}")
    out = ROOT / "data" / "replay" / "espn_player_map.parquet"
    full.to_parquet(out, index=False)
    print(f"Saved {out}")
    if args.publish_map:
        buf = io.BytesIO()
        full.to_parquet(buf, index=False)
        s3.put_object(Bucket=BUCKET, Key=MAP_KEY, Body=buf.getvalue())
        print(f"Published s3://{BUCKET}/{MAP_KEY}")


if __name__ == "__main__":
    main()
