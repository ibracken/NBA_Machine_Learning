# Roadmap

**Status as of 2026-09-28.** The 2026-27 season opens **2026-10-20** (per the NBA schedule API). In-house
projections stay **paused** until every team has played 4 games (about 10 days; see "Early-season pause"). Measurement tools: `scripts/evaluate_projections.py` scores minutes and lineups by data-freshness
regime. `scripts/evaluate_llm.py` scores L2 and L3 and reports LLM spend. 148 dates of 2025-26 projections
with actuals sit in `model_comparison/` for backtesting.

Standing rule: **a change that cannot beat its own absence gets switched off.** That applies to LLM
components exactly as it applies to features.

**Nothing on this list has to wait for the season to be *built*.** Only some things need season data to be
*judged*:
- the L2 go/no-go needs 30 live slates;
- L3 has to run on live slates, because web search on past dates returns the results;
- L4 needs a week of data before its first report;
- the O1 re-fit needs a clean run of fresh injury data.

Everything else can be done and backtested now.

---

## To do before opening night (Ian)

The 2026-09-22 and 2026-09-28 work is committed (2026-09-28) but **not deployed**. In order:

1. ~~Review and commit~~ Done. Full lists under "Done 2026-09-22" and "Done 2026-09-28".
2. ~~Fix the opening-night blocker~~ **Done 2026-09-28** as the early-season pause (next section).
3. **Create the `llm-analyst` Lambda.** It's a one-time step; `deploy.py` only updates existing functions.
   - Settings: PackageType Image, role `lambda-execution-role`, timeout **900 s**, memory **1024 MB**.
   - Environment variable **`CLAUDE_API_KEY`** (same value as in your local `.env`). `deploy.py` does not set
     Lambda environment variables, so add it in the console/CLI like `PROXY_URL`. No live call has been
     made yet.
   - The role must allow S3 `ListBucket` on the bucket for `scripts/evaluate_llm.py`'s usage report. The
     Lambda itself never lists.
4. **Deploy:**
   `python lambda/deploy.py box-score-scraper minutes-projection supervised-learning injury-scraper game-scheduler llm-analyst`
   - O3 is worth doing before this rebuild. `llm-analyst` is already fully pinned; the others are not.
5. **EventBridge permission:** run `scripts/fix_eventbridge_lambda_permissions.ps1 -Apply`.
   `llm-analyst` has been added to its function list.
6. **Error alarm:** create a CloudWatch alarm `nba-lambda-errors-llm-analyst` on the same SNS topic as the
   others.
7. **Smoke-test with real API calls** on the first live slate, or locally:
   `python lambda/llm-analyst/lambda_function.py research` (local runs read `CLAUDE_API_KEY` from `.env`).
   - The local `.venv` does **not** have `anthropic` installed; install `lambda/llm-analyst/requirements.txt`
     into a separate env first rather than disturbing `.venv`.
   - A local run writes to the real S3 bucket, like the Lambda would.
   - Check `llm/usage/{date}.json` for actual cost.
   - Offline tests pass (59 checks). The offline test script is not in the repo; ask Claude to re-create it
     if wanted.

---

## Early-season pause (BUILT 2026-09-28; replaces the opening-night fix)

**The problem it avoids.** `current.parquet` holds the latest season *with games*. Once 2026-27 has games,
players without one are dropped as "free agents" (109 of 117 on a synthetic night 2), players with 0–3 games
get a flat 10 MPG (20/24 if DFF lists them as a starter), players with no current-season row get no FP
features, and offseason movers sit on their old team. The four patches from 2026-09-22 were rejected as
bolt-ons; skipping the first ~10 days avoids all of it.

**Rule** (`serving_features.projection_gate`, mirrored in `llm-analyst/slate.py`; constants in both
`config.py` files - keep in sync): paused until **all 30 teams have 4+ games of today's season** before
today. The season is derived from the date, so opening night (file still holds 2025-26) is paused too.
- Replayed on 2025-26: paused Oct 21–29, **opens Oct 30** (day 10). Expect about Oct 29–30, 2026.
- Mid-season (2026-01-15) it is open.

**While paused:**

| Step | Behavior |
|---|---|
| minutes-projection | Actuals update, then **DFF lineup only**, emailed with a "projections paused (reason)" header. No complex / Formula C projections or lineups. |
| `llm_lineups` | Skipped. |
| research + L3 minutes | **Run normally** (they don't use our projections), so L3 collects data from night 1. |
| L2 `adjustments` | Skipped (nothing to adjust). |
| L1 `preflight` | Input checks only (stale box scores / injuries, missing L3). Saved to `llm/preflight/{date}.json`; **emails only if critical**. No LLM review. |
| supervised-learning, scrapers | Unchanged. |

**What the pause does not fix: the <4-game 10 MPG fallback is a season-long problem.** Share of rotation
players (15+ min that night) who had fewer than 4 games, 2025-26:

| Week | Share |
|---|---|
| Oct 27–Nov 2 (gate opens Oct 30) | 12.8% |
| Nov 3–9 | 6.6% |
| Nov 10–Dec 7 | 1.7–3.3% every week |

Those are returners from injury and call-ups, projected at 10 MPG unless DFF lists them as starters. See M7.

**Tested (read-only, writes and email intercepted):** minutes-projection on 2025-10-25 builds only the DFF
lineup and sends the "paused" email; `llm_lineups` is skipped. llm-analyst offline suite now 59/59, including
L2 skipped, L1 input-only (no email unless critical), and research + L3 still running while paused.

---

## Done 2026-09-28

- **Early-season pause** built (section above) in minutes-projection and llm-analyst. The DFF lineup code
  was moved into `build_dff_lineup()` so it runs whether or not projections are paused; behavior unchanged.
- **Fable removed.** Everything runs on Opus 5.5: L4 at effort high; L2 has one judge (Opus 5.5, high).
  `ADJUSTMENT_JUDGES` still accepts more models if an A/B is ever wanted. Note: `fallbacks: "default"` means
  a *refused* request may be retried by Anthropic on another model; that is logged as `served_model`.
- Re-checked: Opus 5.5 pricing in `config.py` ($4 / $20, cache read $0.20) matches the API docs; the L4
  tool loop only appends to history, which Opus 5.5 requires.
- Measured the season-long <4-game fallback (M7).
- The API key is read from **`CLAUDE_API_KEY`** (not the SDK default `ANTHROPIC_API_KEY`); local runs load
  it from `.env`.
- Email subject now counts only lineups actually built (it counted every entry, including failed ones).

## Done 2026-09-22

Each item was verified by running the real Lambda code read-only on S3 data unless noted.

**Serving bugs (all projections).**
- **Served features were one game stale.** Rolling features use `shift(1)`, but serving read each player's
  latest row, so the most recent game was left out of every average.
  - New `minutes-projection/serving_features.py` advances the features. Checked 100% exact on 582 players.
  - Stored production projections matched the stale rebuild on 93.5% of rows.
  - Gain on games the models never trained on: Formula C MAE 4.952 → 4.907. FP models −0.011 to −0.018;
    only barebones is significant.
- **FP_PER_MIN cutoff mismatch.** Training used the career rate for a player's first 2 games; serving used
  it for the first 5 games already played. Serving now matches training
  (`config.CAREER_RATE_MAX_PRIOR_GAMES = 1`). About 2,224 player-games per season were affected.
- **`IS_HOME` was always 0 at serve time.** `box-score-scraper` now saves the season schedule
  (`stats.nba.com/stats/scheduleleaguev2` via the proxy) to `data/schedule/current.parquet`, and
  minutes-projection maps `IS_HOME` by team.
  - Agrees with box-score `MATCHUP` on 2,455 of 2,460 team-games; all 5 misses are neutral-site games.
  - Effect size in our data: home is worth +0.29 FP per game (CI +0.18 to +0.41), all from efficiency.
    The models learned +0.15 to +0.18.
- **Rest days in training.** Every player's final training row was overwritten with "today minus last
  game" (575 of 582 wrong). Fixed.
  - Measured: on back-to-backs, players who suit up log **+0.88 minutes** with no change in FP per minute.
    That's a minutes-model signal, not an FP one.
- **Crash on an empty injury file.** An empty file meant no `STATUS` column, and the handler crashed.
  Fixed.
- **Local training scripts.**
  - `scripts/supervised_learning.py` deleted. It used a random split and wrote pickles to production from a
    local environment.
  - The training Lambda's local `__main__` now never publishes (`run_supervised_learning(publish=False)`).
- Withdrawn after checking: a feature-list/model mismatch race in training. Training takes about 3.5
  minutes, well inside the 7-minute gap.

**LLM components: all four built** (new `lambda/llm-analyst`, details under each L item below).
- Supporting changes:
  - game-scheduler runs the new steps.
  - minutes-projection gains an `llm_lineups` action.
  - `actuals_updater` and `evaluate_projections.py` include `llm_head_to_head`.
  - injury-scraper now saves `REPORT_URL`, `REPORT_DATE` and `REPORT_TIME`.
- `llm-analyst/requirements.txt` pins every dependency, transitive ones included. Resolved against Linux
  x86_64 / Python 3.11 wheels.
- **Tests:** 53 offline checks pass.
  - The request format is checked through the real SDK: fallbacks, beta header, effort, schemas, tools.
  - All four actions run on the real Jan 15, 2026 slate with a scripted fake Claude that includes bad
    outputs.
  - **Not yet tested:** a live API call; the `minutes-projection` `llm_lineups` action (it compiles but was
    never run); the new scheduler rules against AWS.

**Findings from the historical data.**
- **Silent FP-model failure in production.** All three FP models returned 0 on **9 slates, Jan 7–16, 2026**,
  so in-house lineups those nights were built from zeros. L1's `fp_models_zero` rule catches this. This is
  the "failed model load fails silently" bug. We agreed the email was enough, but it had in fact happened.
- DNP rate for players projected over 10 minutes: 5.5%. Only 37 of 556 are name mismatches, so the 5% in
  M3 is real.
- The Oct 24, 2025 DFF slate has no positions, the only one of 191, so no lineups can be built for that date.

---

## LLM integration

The pipeline sees only what it scrapes. Every night there is public information that exists only as text:
minutes restrictions, load management, game-time decisions, rotation changes announced before the game. No
amount of box-score history recovers it.

Do **not** ask an LLM to emit numeric projections for production. L3 does exactly that, but only as a
measured experiment. The production value is text → structure.

**Models:**
- **Opus 5.5** (`claude-opus-5-5`, $4 / $20 per million tokens) for everything: research, L1, L2, L3, L4.
- L4 also uses Opus 5.5 (effort high). Fable was dropped 2026-09-28.
- Effort is always set explicitly; Opus 5.5 defaults to `medium`.
- Every call opts into server-side refusal fallbacks (`fallbacks: "default"`) and raises on a refusal or a
  cut-off response.
- A **$25/day budget guard** stops further calls once reached.
- Cost per call is logged to `llm/usage/{date}.json`. Web searches are billed separately and only counted.
- **Real nightly cost is unknown until the first live run**; check `scripts/evaluate_llm.py`.

**Nightly schedule** (minutes after pipeline start, which is T−30 before the main slate):
- +12 `research`
- +13 minutes-projection (unchanged)
- +16 `adjustments`
- +18 `preflight`
- +22 `llm_lineups` (shadow only, never emailed; its lateness doesn't matter)
- Mondays: `postmortem`

The lineup you'd actually enter comes from the minutes-projection email (+13, about T−17), as last season;
the preflight verdict follows at +18 (about T−12). Steps that depend on an upstream output poll for it for
up to 5 minutes. Durations of the existing Lambdas
weren't measured; check the first night's CloudWatch logs.

### Shared: nightly research — BUILT
One Opus 5.5 web-search call per game (up to 8 searches) writes a per-team briefing on availability,
restrictions, starting lineups and rotation news. It keeps the **exact snippet** behind each statement.
- Saved to `llm/research/{date}.json`; feeds both L2 and L3.
- One game failing doesn't discard the others; the action saves what succeeded and then raises.

### L1. Pre-flight sanity checker — BUILT
**Coded rules first**, with thresholds measured on 2025-26 fresh-input slates:

| Rule | Severity | Fires when |
|---|---|---|
| `slate_coverage` | critical | fewer than 90% of DFF slate players have a projection (catches the opening-week bug) |
| `fp_models_zero` | critical | most players have 0 FP from all three models (the Jan 7–16 failure) |
| `no_lineups` | critical | no in-house lineup was built today |
| `zeroed_regular` | critical | projected 1 minute or less, averaged 20+ over the last 3 games, and not on the injury report |
| team totals outside 150–290 | critical | the 1st/99th percentile of 2025-26 slate totals |
| team totals outside 190–274 | warning | the 5th/95th percentile |
| `above_role` | warning | projection is 10+ minutes over the season average (historically about 6.7 minutes too high) |
| stale box scores | critical | scheduled games since the last box score are missing |
| stale injuries | critical | the report date or the file's write time is over 24 hours old |
| status conflicts | warning | L2 found a reported status contradicting our projection |

**Then an Opus 5.5 review** of the full slate table (salary, DFF FP, our FP and minutes, Formula C and LLM
minutes, last 5 games, injury status).
- It's asked only for what the rules missed, and must cite numbers from the table.
- Its findings are stored separately from rule findings.

**Output:** an email titled "CRITICAL - check before using lineups" or "ok", plus `llm/preflight/{date}.json`.

**Success:** at least one real defect a month that **no rule** caught. Compare `llm_findings` with
`rule_findings`.

### L2. Beat-writer synthesis → bounded minutes adjustments — BUILT, shadow mode
One judge, Opus 5.5 at effort high (the Fable A/B arm was dropped 2026-09-28). Skipped while the
early-season pause is on. Guardrails, enforced in code:
- Each adjustment cites one snippet, with a verbatim quote of at least 20 characters. The quote is checked
  mechanically against the snippet; paraphrases are rejected.
- The source URL is required.
- The size is clamped to **±25%**.
- A player projected at 0 can't be adjusted.
- Unknown players are rejected.
- Pre- and post-adjustment minutes are logged to `llm/adjustments/log.parquet`. **Nothing is applied.**
- Reported status conflicts ("we project him, the report says out") go to L1, not into adjustments.

**Go/no-go** (`scripts/evaluate_llm.py`): ship only after **30+ slates** with the 95% bootstrap CI of the
adjusted-minus-unadjusted absolute error **entirely below 0**.

**Precision** (share of adjustments a human would endorse): run `--export-review review.csv`, fill in
ENDORSED y/n, then run `--labels review.csv`.

**Earliest possible decision:** about early-to-mid December, since collection starts when the pause lifts
(~Oct 30).

### L3. LLM vs. the whole pipeline — BUILT, prospective only
Opus 5.5 projects minutes for every slate player from the research briefing plus public slate facts only:
names, teams, positions, salaries. It never sees our projections or features.
- Saved as a minutes model: `model_comparison/llm_head_to_head/`.
- `minutes-projection`'s `llm_lineups` action runs our FP models and optimizer on those minutes, so it's
  also scored on realized lineup FP. Those lineups are never emailed.
- `scripts/evaluate_llm.py` reports MAE and bias against complex and Formula C, **split by players with and
  without news** that night, plus realized lineup FP against our lineups and DFF.
- **The planned 20-slate historical backtest was dropped.** Web search on past dates returns the box
  scores, and the model's training data may include 2025-26, so any historical result would be invalid.
  L3 runs only on live slates.
- Prior stated up front: the LLM loses on aggregate MAE and wins on players with news. If it wins outright,
  the statistical pipeline needs rethinking.

### L4. Weekly post-mortem analyst — BUILT
On Mondays, Opus 5.5 (effort high) analyzes projections vs actuals: the season, with the last 7 days
flagged. It can only see the data through a `run_query` tool that the Lambda executes.
- Unsafe filter expressions are refused.
- Each query is logged with its row count.
- Every finding must cite query ids that actually ran, and its row count is checked against the log.
  Findings citing no executed query are rejected.
- Output: an email plus `llm/postmortem/{date}.json` with the full query log, so every finding can be re-run.

---

## Model quality (all can be done now on historical data)

### M1. Predict FP per minute, not FP — next priority
`MIN` is about 70% of the `current` model's importance. The model trains on **actual** minutes but serves
on **projected** minutes, which carry about 5.0 MAE. The `np.random.uniform(-8, 8)` noise in training is a
crude patch, and it's unseeded, so runs differ by about 0.01 R².
- Target `FP_PER_MIN`, drop `MIN` from the features, and serve `projected_FP = predicted_FP_PER_MIN ×
  projected_MIN`.
- Don't retrain on stored historical projections; that teaches the model to correct for bugs that have
  since been fixed.
- Local training experiments are now safe (local runs can't publish).
- barebones is currently best: R² 0.661 on 13 features.

### M2. Shrink projections before optimizing
Our lineups project about 262 FP and realize about 242 (+20). DFF projects 272 and realizes 265 (+7). Most
of the gap is projection bias amplified by picking the maximum.
- Shrink each projection toward the positional mean in proportion to its uncertainty; optionally optimize
  `proj_FP - λ·σ`.
- Backtestable now on the 148 stored dates.

### M3. Model DNP risk explicitly
5.5% of players projected over 10 minutes log zero; this was verified as real, not name mismatches.
Probably better served by L1/L2 than by a classifier, since a scratch is usually announced in text first.
Revisit after L2 has data.

### M4. Cut dead features
- The nine cluster dummies are about 0.4% of importance combined.
- `IS_HOME` is real but small (+0.29 FP per game).
- `REST_DAYS` matters for minutes (+0.88 on back-to-backs), not for FP per minute.
- If the clusters stay dead after M1, drop `cluster-scraper` → `nba-clustering`: two Lambdas and an ECR repo.

### M5. Opponent and pace adjustment
Correction to the earlier plan: the `PACE` and `DEF_RATING` that `cluster-scraper` pulls are **player-level
and season-to-date**, overwritten on every run. They aren't opponent-team stats and have no history of past
values to train on.
- The opponent join key now exists (the schedule's `OPPONENT` column).
- What's still needed is point-in-time team pace and defense: team stats as of each game date, e.g. a
  date-bounded `leaguedashteamstats` query (not yet verified).

### M6. Team-minute constraint
Per-player bias tracks the team's projected total: 180 → −2.22, 230 → +0.10, 292 → +5.62. Naively
rescaling to 240 makes MAE worse, because the slate covers only about 10 of about 15 rotation players.
- Needs a per-slate target based on how many players are covered.
- L1 now flags extreme team totals in the meantime.

### M7. Blended early-season baseline (replaces the hard 4-game cutoff)
Players with 0–3 games this season get a flat 10 MPG (starter floor 20/24) in both `project_minutes_*` and
`injury_system`. That hits 2–3% of rotation players every week all season (returners, call-ups), not just
opening week (measurements under "Early-season pause").
- Replace the cutoff with a shrinkage blend: `w·this_season + (1−w)·prior`, `w = n/(n+k)`, where the prior
  is last season's average (same team) and `k` is fit by replaying 2025-26.
- Could also let the pause lift earlier. Do it in-season; it touches the core minutes path.

---

## Operations

### O1. Retune `INJURY_ADJUSTMENT_WEIGHT` on fresh data
Currently 0.35, fit on the 2025-26 fresh-injury window. Re-fit after about 30 slates of 2026-27 with a
working injury feed. That is the only item here that genuinely needs season data first.

### O2. Alarm on data staleness, not just Lambda errors — mostly covered by L1
L1 checks the dates *inside* the data:
- box scores missing scheduled games;
- the injury report's own date (`REPORT_DATE`, new);
- the injury file's age.

Those findings arrive in the preflight email but **don't set off a CloudWatch alarm**. If L1 itself fails,
its Lambda error alarm fires. Remaining option: a standalone check that raises, independent of the LLM
Lambda.

### O3. Pin transitive dependencies — do before the next deploy
Two of nine rebuilds failed on 2026-09-18/19 because unpinned transitive dependencies resolved to versions
with no Lambda-compatible wheel (Pillow, scipy).
- Remaining drift: pandas 2.1.3/2.1.4, pyarrow 14.0.1/14.0.2/20.0.0, and boto3 unpinned in three scrapers.
- `llm-analyst` shows the approach: resolve with
  `pip download --platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all:` and pin
  everything. Apply the same to each function.
