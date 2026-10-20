# Roadmap

**Status, 2026-10-11.** Season opens **Oct 20**. In-house projections stay paused until every team has played 4
games (about Oct 30); until then only the DFF lineup is emailed.

**Where we stand.** On last season's 135 slates (backtest):
- the new learned minutes model beats our production lineups by **+9 to +12 FP per slate**;
- it still trails DFF by **about 11 FP per slate** (95% CI −17.7..−4.2);
- the remaining gap is mostly per-minute scoring, not minutes.

Evidence for every number here is in [`findings.md`](findings.md).

**Standing rules**
- A change that can't beat its own absence gets switched off. That covers models, AI pieces and features.
- Decisions are scored on realized lineup FP. One season (about 135 slates) can only detect about ±5–7 FP.
- DFF is the benchmark, never an input.

**Decisions in force**
- **The learned minutes model replaces the complex model as our #1 projection everywhere:** the lineup email,
  the AI steps, preflight and comparisons. (Ian, 2026-10-11)
- **Removed (Ian, 2026-10-11):**
  - the old complex model and its hand-tuned injury rules;
  - the `current` and `fp_per_min` FP models;
  - the cluster pipeline (cluster-scraper, nba-clustering);
  - the team-minutes check;
  - the X/Grok question.

  Formula C stays as the cheap control. The removals happen in one change during the pause (step 4), so the
  pipeline changes once. **Barebones and the cluster pipeline are commented out, not deleted** (Ian), so they can
  come back.
- **Weekly post-mortem:** paused 2026-10-11 (Ian), re-enabled 2026-10-18 on learned-model data.
- **FP = learned minutes × the player's season FP per minute.** That beat every FP-model combination in the
  backtests.
- **Injury designations:** Out and Doubtful count as out. Questionable, Probable and Available are **not**
  model inputs; whether the AI research can tell real Questionables from fake ones is an experiment. (Ian,
  2026-10-11)
- **Preflight email at about T−12 is fine.** (Ian, 2026-10-09)

---

## The plan

### Before opening night (Oct 20)

1. **Injury scraper: recommend no change before opening night (pending Ian's OK).** The season page listing no reports looks
   like the offseason, not a break: the 2026-27 page doesn't exist yet (404), the NBA publishes no preseason
   reports, and the scraper read links from the page all last season. How soon the new page appears is unverified
   (no logs survive from last October). If the opening-night check finds no report, build the report address
   directly (`Injury-Report_YYYY-MM-DD_HH_MMPM.pdf`, as `scripts/fetch_injury_history.py` does). The injury
   feed hasn't been written since Jan 13, so Oct 20 is also the first live run of the September fixes.
2. **daily-predictions error message:** skipped (Ian, 2026-10-11). The crash already raises an alarm.
3. **Opening-night check (Oct 20): `python scripts/opening_night_check.py`** after the lineup email. It checks:
   - game-scheduler skipped preseason and scheduled the 20th;
   - DFF returned players ("Extracted N players");
   - the schedule files refreshed;
   - the injury scraper wrote `current.parquet` and `report_statuses.parquet`, with hyphenated and "III" names
     correct;
   - the lineup email says "paused" and contains the DFF lineup;
   - the research call ran (`llm/research/2026-10-20.json`);
   - `llm/usage/2026-10-20.json` shows the real nightly AI cost;
   - no `nba-lambda-errors-*` alarm.

### Built early (2026-10-18): the learned model is #1

4. **Learned model in production, and what it replaces removed. Built, tested and deployed 2026-10-18.**
   - **Shared code:** `lambda/shared/learned_minutes.py` holds the rows, features, training and prediction, and is
     used by supervised-learning (nightly training), minutes-projection (serving) and the backtests.
     - It reproduced the backtest table and features exactly (163,879 rows).
     - The serving path was identical on four test nights.
     - The nightly Lambda picked the same 8 players as the backtest on every night tested.
     - Head-to-head with DFF on this code: −12.0 per slate (CI −19.0..−5.1), +7.9 over Formula C.
   - **Position inputs dropped:** minutes MAE 4.888 → 4.900, and the lineup difference was within noise. No
     position table is needed.
   - **Injury history:** `data/injuries/history.parquet` (archived 2022-26 pre-tip reports), appended nightly by
     injury-scraper.
   - **Names:** `lambda/shared/player_names.py` in minutes-projection (DFF slate and actuals) and llm-analyst (slate
     teams).
   - **Removed:** the complex model, and the `current` / `fp_per_min` FP models.
   - **Switched off but kept:** barebones (`TRAIN_BAREBONES`), the cluster schedule entries and the cluster join.
   - **Also:** the post-mortem is re-enabled on learned-model data, and the viewer shows the learned lineup.
   - **Left to do:**
     - Run supervised-learning once to publish the first model, then `check_model` (load + predict in the
       Lambda), per the AGENTS.md release checklist.
     - About Oct 25, once a few 2026-27 games exist: confirm a nightly training run succeeds on them.
5. **AI steps pointed at the learned model. Built and tested offline 2026-10-18 (all checks pass).**
   - **The adjustments judge** gets the model's description, tonight's out list, and each player's absence
     adjustment, and is told not to adjust for anything the model already counts.
   - **The AI lineup replaces the per-player head-to-head:**
     - one call per slate;
     - code checks the slate players, the slots and the $50k cap, with one retry;
     - it runs during the pause too.
   - **Research prompt:** states the current time, and covers rotation news for non-slate players.
   - **Preflight** reads the learned model. Its team-total thresholds were re-measured on learned backtest
     projections (warning outside 162–243 slate minutes, critical outside 132–254), and the "above role" rule was
     dropped: learned jumps of 10+ minutes realized within 1.3 min.

### Live season

6. **Pause lifts (about Oct 30):** the learned model's lineup goes in the email; the AI judge and AI lineup start
   collecting.
7. **Live comparison at 30 slates (about late November):** learned vs Formula C vs DFF on realized lineup FP.
   If the learned model trails Formula C, stop and investigate. That would contradict every backtest.
8. **AI go/no-go (about mid-December, 30+ slates):** each AI piece is kept or switched off by its own rule below.

### Later: one at a time, once the learned model is live (the remaining ~11 FP to DFF)

9. **Opponent pace and defense, built from our own box scores.** Team possessions and points allowed are
    derivable game by game. First test whether they explain the per-minute misses on the players DFF picked
    and we didn't, before building anything.
10. **Historical betting lines (game total, spread).** Find a source with past seasons; likely the biggest
    per-minute input DFF has and we don't.
11. **Early-season blend:** players with 0–3 games currently get a flat 10 minutes (2–3% of rotation players
    every week). Replace it with a blend of this season and last season, weighted by games played. This might
    also let the pause lift earlier.
12. **Faster backtests:** switch the backtest models to `HistGradientBoostingRegressor` (multi-core, same Lambda
    version) after confirming equal accuracy. Runs drop from about 45 minutes to about 5.

### Housekeeping (whenever convenient)
- Pin transitive dependencies in every Lambda the way llm-analyst does. Unpinned versions broke 2 of 9 rebuilds
  in September.
- box-score-scraper re-downloads three finished seasons every night; load them from S3 instead.
- Staleness checks live in the preflight email, not in CloudWatch alarms. Add an alarm if the email proves too
  easy to miss.

---

## AI pieces (Opus 5.5, all in the `llm-analyst` Lambda)

None of these changes the emailed lineup today. Each runs in shadow mode or as a report until it earns its place.
Cost: one preflight review measured $0.11; the real nightly total is unknown until Oct 20. A $25/day guard
stops all calls.

**Research (nightly, before the judge and the AI lineup).**
- **Goal:** collect tonight's availability and playing-time news per game, with the exact source snippets.
- **Opening night (2026-10-20):**
  - **0 citations on all 3 games.** The tool's default "dynamic filtering" runs searches inside code execution.
    Switched to direct search (`allowed_callers: ["direct"]`), capped at 5 searches. Retested on OKC@SAS: 25
    citations, $0.40 against $0.56 before.
  - **Cost:** about $0.45 per game ($0.40 tokens plus 5 searches at $10 per 1,000), so about $4.50 on a 10-game
    night for research alone.
  - **Option to cut it (Ian's call):** research only games with open news. Last season 44% of games had a rotation
    player Questionable or Doubtful, and 68% counting rotation players returning from 3+ missed games.
- **Status:** built. Prompt fixes done 2026-10-18:
  - state the current time, so the AI can tell a report from this morning from one 20 minutes old;
  - ask about off-slate rotation players too, since the learned model uses every absent teammate.

**Adjustments judge (shadow).**
- **Goal:** move a player's minutes only for news the model can't see: a stated minutes restriction, planned
  rest, a lineup change, or a Questionable player confirmed in or out.
- **Change (Ian, 2026-10-11; built 2026-10-18):** the judge must know what the model already did, so it doesn't stack a second
  overreaction on top, which is the complex model's old failure. Per game it gets:
  - a description of the model: what it accounts for, which is the official injury report's Out/Doubtful list,
    each teammate's with/without history, recent minutes and games missed;
  - the teammates the model is treating as out tonight;
  - per player: projected minutes, season average, last-7 average, and **how much the model moved him for
    tonight's absences** (projection minus the same model with no absences);
  - the instruction: don't adjust for anything already in that list.

  The verbatim-quote check and ±25% cap stay. Status contradictions are still flagged rather than applied.
- **Kill rule:** after 30+ slates, adjusted minutes must have lower error than unadjusted (bootstrap CI below 0),
  measured separately on Questionable players. Otherwise switch it off.

**AI lineup (shadow; replaces the per-player AI projections, Ian 2026-10-11; built 2026-10-18).**
- **Goal:** can an AI pick a better lineup from tonight's news and its own knowledge than our pipeline does?
- **Design:**
  - One call per slate: the slate (name, team, opponent, position, salary) plus the research briefings.
  - It returns 8 players. Code checks the salary cap, positions and that every player is on the slate, with one
    retry on an invalid lineup. Saved to `model_comparison/ai_lineup/daily_lineups.parquet`; runs during the pause.
  - It never sees our projections or DFF's.
- **Kill rule:** after 30+ slates, scored on realized FP against our lineup and DFF's. Drop it if it doesn't beat
  ours.

**Preflight check (T−12 email).**
- **Goal:** catch broken inputs or absurd projections before you enter a lineup.
- **Design:**
  - Coded rules: slate coverage, a regular projected at 0, team totals, stale box scores or injury file, no
    lineup built.
  - Then an AI review for anything the rules missed, citing numbers from the table.
- **Kill rule for the AI review:** it must catch at least one real problem a month that no rule caught.
  Otherwise keep only the rules.

**Weekly post-mortem (Mondays).**
- **Goal:** find systematic misses in the logged projections.
- **Status:** re-enabled 2026-10-18 on learned-model data. Until about Oct 30 it has nothing to grade.


---

## Reference

- **Backtest scripts** (all read-only on S3; outputs in `data/replay/`, gitignored):
  - `build_replay.py`: the 4-season player-game table;
  - `fetch_injury_history.py`: archived pre-tip injury reports;
  - `injury_response.py`: learned models and leak checks; `--predict-only` refits only the final models;
  - `backtest_learned_lineups.py`: learned model vs production;
  - `backtest_vs_dff.py`: head-to-head with DFF.
- **Name matching:** `lambda/shared/player_names.py`. `deploy.py` passes `lambda/shared` as a build context; a
  Dockerfile uses it with `COPY --from=shared player_names.py ${LAMBDA_TASK_ROOT}`.
- **Redeploying:** `python lambda/deploy.py <function>` with `.venv/Scripts` first on PATH (PowerShell:
  `$env:PATH = "$PWD\.venv\Scripts;$env:PATH"`). Environment variables (`PROXY_URL`, `CLAUDE_API_KEY`) are set by
  hand.
- **Nightly schedule** (minutes after the pipeline starts at T−30 before the first tip):
  - +0 cluster-scraper, +2 nba-clustering (both being removed), +4 box-score-scraper, +6 supervised-learning;
  - +9 daily-predictions (DFF), +11 injury-scraper, +12 research;
  - +13 minutes-projection (the lineup email, about T−17);
  - +16 adjustments judge, +18 preflight email (about T−12), +22 AI lineup;
  - Mondays: post-mortem.
