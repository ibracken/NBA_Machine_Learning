# injury-scraper

Parses the NBA's official injury report PDF and writes `OUT` players to S3 for `minutes-projection`.

## Source

`https://official.nba.com/nba-injury-report-{season}-season/` — the season slug is derived from today's date
(July onward = new season) and falls back to the prior season's page on 404, since the new page appears late
in the offseason. The most recent `Injury-Report_*.pdf` link on the page is downloaded and parsed with
`pdfplumber`. Reports are not published in the offseason, so the Lambda fails (and alarms) until preseason.

Requests use a 30s timeout with 3 retries. If `PROXY_URL` is set in the Lambda environment, requests are
routed through it — the NBA's CDN has throttled AWS egress IPs before (10s timeouts every run, Jan–Jun 2026).

## Output

`s3://nba-prediction-ibracken/data/injuries/current.parquet`

| Column | Meaning |
|---|---|
| `PLAYER` | normalized name (`first last`, lowercase, unidecode; suffixes spaced: `nance jr.`) |
| `TEAM` | 3-letter abbreviation inferred from the game matchup |
| `STATUS` | always `OUT` (Questionable/Probable/Available and G League assignments are dropped) |
| `RETURN_DATE`, `RETURN_DATE_DT` | `None` — the PDF has no return dates |
| `ESTIMATED_INJURY_DATE` | day after the player's last box-score game (current season, then previous season); players with no NBA games are dropped |

## Running

- In the pipeline: scheduled by `game-scheduler` 11 minutes into the daily chain (no standalone rule).
- Locally: `python lambda/injury-scraper/lambda_function.py` (writes to S3).
- Deploy: `python lambda/deploy.py injury-scraper`.

## Monitoring

CloudWatch alarm `nba-lambda-errors-injury-scraper` emails the `lineup-optimizer-notifications` SNS topic
whenever an invocation raises. Logs: `/aws/lambda/injury-scraper`. Sanity check: the `LastModified` of
`data/injuries/current.parquet` should be today on any game day.
