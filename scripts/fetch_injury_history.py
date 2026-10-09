"""
Historical pre-tip injury reports, 2022-23 .. 2025-26, from the official NBA injury report PDFs (the source
injury-scraper reads live). Writes data/replay/injury_reports.parquet (gitignored); PDFs cached in
data/replay/injury_pdfs/ so reruns only fetch what is missing.

For each game date: the latest report published at or before (earliest tip - LEAD_MINUTES), which is what the
pipeline could have seen before lock. Report files are named Injury-Report_YYYY-MM-DD_HHPM.pdf (hourly) and, from
some point in 2025-26, Injury-Report_YYYY-MM-DD_HH_MMPM.pdf (15-minute slots); both are tried.

Keeps every listed player with his status (Out / Doubtful / Questionable / Probable / Available), unlike
production, which keeps Out only. Names go through production's normalize_name. Hyphenated names are parsed
(production's parser drops them).

Names are then matched to box-score names (data/replay/replay.parquet) on letters only, ignoring suffixes, which
covers "o.g." vs "og", "jimmy butler" vs "jimmy butler iii" and PDF text that loses hyphens or spaces.

Output: D, FIRST_TIP, REPORT, PLAYER, STATUS, GLEAGUE
Needs PROXY_URL (from .env) for the NBA schedule API.
Usage: python scripts/fetch_injury_history.py [--reparse]   (--reparse: re-read cached PDFs, no downloads)
"""

import importlib.util
import logging
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pdfplumber
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "replay" / "injury_reports.parquet"
PDF_DIR = ROOT / "data" / "replay" / "injury_pdfs"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
LEAD_MINUTES = 30
EARLIEST_REPORT_HOUR = 9
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{}.pdf"
NAME_STATUS = re.compile(r"([A-Z][\w'\.\-]*(?:\s?(?:Jr\.|Sr\.|II|III|IV))?,\s?[A-Z][\w'\.\-]*)\s+"
                         r"(Out|Doubtful|Questionable|Probable|Available)\b(.*)$")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def first_tips(bss, season):
    """Earliest scheduled tip (ET, naive) per game date, regular season and play-in/playoffs."""
    sched = bss.nba_api_get("https://stats.nba.com/stats/scheduleleaguev2", {"LeagueID": "00", "Season": season}).json()
    tips = {}
    for gd in sched["leagueSchedule"]["gameDates"]:
        for g in gd["games"]:
            if g["gameId"].startswith(bss.EXCLUDED_GAME_ID_PREFIXES):
                continue
            t = pd.Timestamp(g["gameDateTimeEst"]).tz_localize(None)
            d = t.normalize()
            tips[d] = min(tips.get(d, t), t)
    return tips


def slot_names(cutoff):
    """Candidate report file stems at or before cutoff, newest first."""
    t = cutoff.floor("15min")
    while t.hour >= EARLIEST_REPORT_HOUR:
        h12 = t.strftime("%I")
        ampm = t.strftime("%p")
        day = t.strftime("%Y-%m-%d")
        if t.minute == 0:
            yield f"{day}_{h12}{ampm}"
            yield f"{day}_{h12}_00{ampm}"
        else:
            yield f"{day}_{h12}_{t.minute:02d}{ampm}"
        t -= timedelta(minutes=15)


def fetch_report(session, cutoff):
    for stem in slot_names(cutoff):
        path = PDF_DIR / f"{stem}.pdf"
        if path.exists():
            return stem, path
        r = session.get(URL.format(stem), timeout=30)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            path.write_bytes(r.content)
            return stem, path
        time.sleep(0.2)
    return None, None


def parse(path, normalize_name):
    rows = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for line in (page.extract_text() or "").split("\n"):
                m = NAME_STATUS.search(line)
                if m:
                    rows.append({"PLAYER": normalize_name(m.group(1)), "STATUS": m.group(2),
                                 "GLEAGUE": "GLeague" in m.group(3) or "G League" in m.group(3)})
    return rows


def name_key(name):
    return re.sub(r"[^a-z]", "", re.sub(r"(jr|sr|ii|iii|iv)\.?$", "", name.strip()))


def match_box_names(df):
    replay = ROOT / "data" / "replay" / "replay.parquet"
    if not replay.exists():
        return df
    box = pd.read_parquet(replay, columns=["PLAYER"]).PLAYER.unique()
    keys = pd.Series(box, index=[name_key(n) for n in box])
    keys = keys[~keys.index.duplicated(keep=False)]
    mapped = df.PLAYER.map(lambda n: n if n in set(box) else keys.get(name_key(n), n))
    print(f"name matching: {(mapped != df.PLAYER).sum():,} rows renamed to box-score spelling")
    return df.assign(PLAYER=mapped)


def reparse():
    load_dotenv(ROOT / ".env")
    logging.disable(logging.INFO)
    inj = load_module("inj", "lambda/injury-scraper/lambda_function.py")
    keys = pd.read_parquet(OUT).drop_duplicates("D")[["D", "FIRST_TIP", "REPORT"]]
    out = [{"D": k.D, "FIRST_TIP": k.FIRST_TIP, "REPORT": k.REPORT, **row}
           for k in keys.itertuples() for row in parse(PDF_DIR / f"{k.REPORT}.pdf", inj.normalize_name)]
    return pd.DataFrame(out), []


def main():
    if "--reparse" in sys.argv:
        df, missing = reparse()
        save(df, missing)
        return
    load_dotenv(ROOT / ".env")
    logging.disable(logging.INFO)
    bss = load_module("bss", "lambda/box-score-scraper/lambda_function.py")
    inj = load_module("inj", "lambda/injury-scraper/lambda_function.py")
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"

    out, missing = [], []
    for season in SEASONS:
        tips = first_tips(bss, season)
        print(f"{season}: {len(tips)} game dates", flush=True)
        for d, tip in sorted(tips.items()):
            stem, path = fetch_report(session, tip - timedelta(minutes=LEAD_MINUTES))
            if path is None:
                missing.append(d)
                continue
            for row in parse(path, inj.normalize_name):
                out.append({"D": d, "FIRST_TIP": tip, "REPORT": stem, **row})
    save(pd.DataFrame(out), missing)


def save(df, missing):
    df = match_box_names(df).drop_duplicates(["D", "PLAYER"], keep="first")
    df.to_parquet(OUT, index=False)
    print(f"{len(df):,} rows, {df.D.nunique()} dates with a report; no report found for {len(missing)} dates: "
          f"{[str(d.date()) for d in missing[:20]]}")
    print(df.STATUS.value_counts().to_dict())


if __name__ == "__main__":
    main()
