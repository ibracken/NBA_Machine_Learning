# injury-scraper

Parses the NBA's official injury report PDF and writes `OUT` players (Doubtful counted as Out) to S3 for
`minutes-projection`, plus every listed status for models that use them.

## Source

`https://official.nba.com/nba-injury-report-{season}-season/` — the season slug is derived from today's date
(July onward = new season) and falls back to the prior season's page on 404, since the new page appears late
in the offseason. The most recent `Injury-Report_*.pdf` link on the page is downloaded and parsed with
`pdfplumber`. Reports are not published in the offseason, so the Lambda fails (and alarms) until preseason.

Requests use a 30s timeout with 3 retries. If `PROXY_URL` is set in the Lambda environment, requests are
routed through it — the NBA's CDN has throttled AWS egress IPs before (10s timeouts every run, Jan–Jun 2026).

## Output

`s3://nba-prediction-ibracken/data/injuries/current.parquet` (minutes-projection treats every row as injured)

| Column | Meaning |
|---|---|
| `PLAYER` | normalized name (`first last`, lowercase, unidecode; suffixes spaced: `nance jr.`; hyphens kept: `shai gilgeous-alexander`) |
| `TEAM` | 3-letter abbreviation inferred from the game matchup |
| `STATUS` | always `OUT`: Out and Doubtful (Doubtful players sat 98.8% of the time, 2022-26). Other statuses and G League assignments are dropped |
| `RETURN_DATE`, `RETURN_DATE_DT` | `None` — the PDF has no return dates |
| `ESTIMATED_INJURY_DATE` | day after the player's last box-score game (current season, then previous season); players with no NBA games are dropped |

`s3://nba-prediction-ibracken/data/injuries/report_statuses.parquet`: every listed player (G League excluded) with
`STATUS` in `OUT` / `DOUBTFUL` / `QUESTIONABLE` / `PROBABLE` / `AVAILABLE`, plus `PLAYER`, `TEAM` and the report columns.

Before 2026-10-09 hyphenated names were never parsed, and the suffix rule split names containing "ii" or "iv"
(`joel emb iid`, `trey murphy i ii`), so those players were never matched as injured.

Historical reports for backtests: `scripts/fetch_injury_history.py` (the PDFs stay online by URL; the season
page lists none in the offseason).

## Running

- In the pipeline: scheduled by `game-scheduler` 11 minutes into the daily chain (no standalone rule).
- Locally: `python lambda/injury-scraper/lambda_function.py` (writes to S3).
- Deploy: `python lambda/deploy.py injury-scraper`.

## Monitoring

CloudWatch alarm `nba-lambda-errors-injury-scraper` emails the `lineup-optimizer-notifications` SNS topic
whenever an invocation raises. Logs: `/aws/lambda/injury-scraper`. Sanity check: the `LastModified` of
`data/injuries/current.parquet` should be today on any game day.
