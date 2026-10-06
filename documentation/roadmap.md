# Roadmap

**Status as of 2026-10-05.** Everything through 2026-09-29 is committed and deployed. The 2026-10-05
preseason fixes are deployed but not yet committed. The 2026-27 season
opens **2026-10-20** (per the NBA schedule API). In-house projections stay **paused** until every team has
played 4 games, about 10 days (see "Early-season pause"). Only the DFF lineup is emailed until then.

Standing rule: **a change that cannot beat its own absence gets switched off.** That applies to LLM components
and features alike. Every item below says how it gets measured, and what result would kill it.

**Measurement tools**
- `scripts/evaluate_projections.py` scores minutes and lineups by data-freshness regime.
- `scripts/evaluate_llm.py` scores L2 and L3 and reports LLM spend.
- `scripts/backtest_fp_rate.py` replays 2025-26 slates against FP models trained on earlier seasons.
- **`scripts/build_replay.py` builds the historical replay table** (`data/replay/replay.parquet`, local,
  gitignored): **163,879 rows**, one per rotation player per team-game, 2022-23..2025-26, **including players
  who didn't play**. Each row has pre-game features (season / last-7 / previous minutes, games played, FP
  averages, career averages, team games missed, days off), production Formula C, minutes freed by absent
  teammates, and actual minutes and FP.
  - Validated 2026-10-06: the pre-game averages match the box-score files' own columns on 100% of rows, and
    `FC_MIN` matches production `project_minutes_formula_c` on 400 of 400 sampled rows (including the
    return and <4-game rules).
  - Use it to fit on 2022-25 and test on 2025-26 for every projection fix. Salaries exist only for 2025-26
    (from 10/24), so the lineup harness remains the final check.
  - Absences are realized (who didn't play), standing in for the injury report. That's right for measuring
    how many minutes teammates actually gain. Positions aren't in box scores.
- Stored 2025-26 history: minutes projections for **101 slates (Nov 29 – Apr 12)**, plus lineups for each
  model.

**Only these need live season data to be judged:** the L2 go/no-go (30 live slates), L3 (live slates only),
L4 (a week of data), and the O1 re-fit (fresh injury data). Everything else can be backtested now.

---

## Next up (in order)

1. **Fix A: DONE and deployed 2026-10-06.** The Lambda trains on all data (231 s, unchanged), with per-game
   deterministic noise (identical models locally and in AWS), and publishes feature names together with the
   model. New models were published and verified to load and predict through minutes-projection's serving
   code.
   - **Fix B** (retire `current` / `fp_per_min`) is still awaiting Ian.
2. **M2: fix lineup selection. TOP PRIORITY** (Ian, 2026-10-05). The repeat-pick / optimizer's-curse
   problem is the biggest measured lineup loss, about 40 FP per lineup.
3. **M1b: fit the shared minutes baseline to data.** This improves complex minutes. Complex is the target,
   and Formula C is only the control.
4. **Opening night (10/20):** run through the watch list below.
5. **About Oct 30:** the pause lifts, and L2 starts collecting.
6. **About mid-December:** the L2 go/no-go, and the X/Grok check (under L2).
7. **In-season:** M7 (blended baseline) and O1 (injury weight re-fit).

**Opening-night alarm still expected (fix proposed 2026-10-05, not approved yet):** cluster-scraper runs
before the first tip on 10/20, when 2026-27 stats are empty, and fails the same way it did in preseason.
- Proposed fix: when the current season returns zero rows (as opposed to a failed request), warn and keep
  last season's clusters.
- Also proposed: daily-predictions should say "DFF returned no projections" instead of `KeyError: 'Player'`.

**Open questions for Ian**
- How many minutes before lock do you need the preflight email? It lands around T−12. It can come earlier
  if it stops waiting for L2.
- Approve fix B (retire two FP models)?

---

## Opening night (10/20) watch list

These are the first live runs of the new steps:
- game-scheduler skipped every preseason day (it logs "Preseason: ...") and schedules on 10/20.
- DFF actually has projections: daily-predictions logs "Extracted N players", not 0. The table layout
  still matched in preseason, but it only had a placeholder row.
- `data/schedule/game_dates.json` and `current.parquet` are refreshed by box-score-scraper.
- `llm/research/2026-10-20.json` and L3 rows in `model_comparison/llm_head_to_head/` are written. This is
  the first live web-search call.
  - Check whether its citations include any x.com links (see the X/Grok note under L2).
- The lineup email says "In-house projections paused" and still contains the DFF lineup.
- `llm/usage/2026-10-20.json` shows the real nightly cost.
- Any `nba-lambda-errors-*` alarm email.

---

## Model quality

### M1. Predict FP per minute — FAILED its backtest (2026-10-02); not shipping
`scripts/backtest_fp_rate.py` trains on 2022-23..2024-25 and replays the 100 stored 2025-26 slates. It uses our
real projected minutes, minutes-projection's serving features, and the production optimizer.

**Per-minute rate × projected minutes is worse than barebones:**

| Minutes input | FP MAE vs barebones | 95% CI |
|---|---|---|
| Complex | +0.20 | +0.13 to +0.26 |
| Formula C | +0.12 | +0.05 to +0.19 |

- Lineups: −4 realized FP per slate (CI −11 to +3).
- The pass rule was "beat barebones with the CI below 0", so it fails.
- **With actual minutes it ties barebones** (6.24 vs 6.23 MAE), with almost no bias (−0.09 vs −1.35). The loss
  is in how it handles minutes **error**: rate × minutes passes every minute of projection error straight
  through.
- **The ±8 noise in training does real work.** It teaches the old models to lean on history when the minutes
  input looks off. That is a crude but useful shrinkage. See M1b for a principled replacement.

**Training-pipeline problems confirmed while specifying M1** (fix A):
- **The deployed models never train on the newest 20% of games.** The 80/20 time split feeds the R² log,
  and the model trained on the 80% is the one published.
  - On opening night 2026 the cutoff is 2025-12-31, so 60% of 2025-26 is never trained on.
  - Training on all data improves barebones by **−0.064 FP MAE (CI −0.083 to −0.045)**, and the rate model
    by −0.23.
  - Fix: evaluate on 80/20, then refit on everything and publish that.
- **The noise is unseeded**, and seed-to-seed swings are as large as the effects we measure.
  - Projected-minutes MAE across three seeds: current 8.19–8.25, fp_per_min 8.18–8.31, barebones 8.14–8.18.
  - Fix: seed it.
- **The logged R² is measured on noisy actual minutes**, so it ranks `current` first. On projected minutes,
  which is what serving sees, barebones is best. Log the projected-minutes metric instead (it needs M1b's
  historical minutes).

**barebones is best or tied everywhere** (fix B, pending Ian's call):
- Projected-minutes MAE across all seeds.
- Lineup median (complex / Formula C): barebones 240.6 / 239.2, current 240.4 / 237.9,
  fp_per_min 234.2 / 236.1.
- Retiring `current` and `fp_per_min` cuts in-house lineups from 6 to 2, with no measured loss.

### M1b. Fix the minutes model at the root (supersedes "train on served minutes")
**Finding (2026-10-05): our projected minutes are overconfident.** Over 2025-26:
- actual ≈ 5.4 + 0.79 × complex projection, and 4.7 + 0.83 × Formula C (a calibrated model would be
  0 + 1.0×);
- players projected about 36 minutes played 33.7; players projected about 13 played 15.0.

That is why rate × minutes lost: it multiplies by the overconfident number. Calibrating minutes first improved
a simple rate × minutes from 8.30 to 8.24 FP MAE, about a third of the gap to barebones.

**Calibrating after the fact would be a patch.** The root cause is that the minutes models are hand-built, not
fit to outcomes. Formula C's 0.5 / 0.3 / 0.2 weights are constants in the code, and complex adds a
redistribution known to overshoot about 3×. A model fit to actual minutes is calibrated by construction.
Baseball's [Marcel](https://library.fangraphs.com/the-projection-rundown-the-basics-on-marcels-zips-cairo-oliver-and-the-rest/)
builds regression to the mean *into* the projection, and is hard to beat. Post-hoc recalibration is what ML
reaches for when retraining isn't possible
([overview](https://www.emergentmind.com/topics/post-hoc-calibration-methods)).
- **Complex is the target.** Complex uses this same formula (`formula_c_projection`) as every player's
  no-injury baseline, then adds injury redistribution. Formula C is the one version that can be recomputed
  for past seasons (complex needs injury reports, which only exist from 2025-26). It is therefore the
  training proxy, and results are reported on complex.
- **Approved idea (Ian):** fit the baseline's weights, plus a pull toward the player's baseline, by least
  squares on 2022-23..2024-25 (78,307 games; plenty for about 4–6 numbers). Score minutes MAE and the
  calibration slope on the held-out 2025-26 (about 26k games), for both the base formula and complex
  minutes. Then rerun the M1 harness with those minutes.
- **The streak test shows why fitting beats guessing** (2025-26, 21,167 games of players with 10+ prior
  games):

  | Recent streak (last-7 avg vs season avg) | Error (formula − actual) |
  |---|---|
  | Hot, more than +4 | −2.2 |
  | Steady | −0.1 |
  | Cold, more than −4 | +0.4 |

  - The formula trusts recent minutes *too little*: role jumps mostly stick.
  - The overshoot on projections of 33+ minutes (about +0.8) is separate, and appears whatever the streak.
  - (An earlier claim in this session that it "trusts hot streaks too much" was wrong.)
- **The earlier ideas below are kept for reference.**

#### Earlier M1b idea: train on the minutes we actually serve
**How others do it** (researched 2026-10-02):
- DFS projection sites build projections as **projected minutes × FP per minute**
  ([RotoGrinders](https://rotogrinders.com/fantasy/lessons/accurately-predicting-minutes-nba-dfs),
  [DraftKings Network](https://dknetwork.draftkings.com/nba/2020/5/31/21309300/nba-all-star-lesson-02-fantasy-points-per-minute)).
  That only works when the minutes are good, and they adjust minutes by hand from news. Ours are raw model
  output, which is why M1's version lost.
- [SaberSim](https://support.sabersim.com/en/articles/12078831-how-projections-work) simulates each game
  thousands of times, and a projection is the mean over simulated outcomes. That's too heavy for us.
- The standard ML answer to "an input is itself a prediction at serve time" is to **train on out-of-sample
  predictions of that input**, so training inputs look like serve-time inputs
  ([input-space expansion](https://arxiv.org/pdf/1211.6581),
  [out-of-fold stacking](https://machinelearningmastery.com/out-of-fold-predictions-in-machine-learning/)).
  The ±8 noise is a crude stand-in for this.
- The related statistics fix is **regression calibration**: replace the noisy input with its expected true
  value given what's observed. Here that means shrinking projected minutes toward history before using them.

**Experiment** (same harness as M1, no production change):
- **Variant a:** keep the barebones features, but set `MIN` to Formula C's projection recomputed for every
  historical game from pre-game box-score features, with no noise. This is leak-free, because those features
  are already `shift(1)`.
  - Formula C is `0.5·baseline + 0.3·last7 + 0.2·prev_game`, capped at 37, with the <4-game rule and the
    return-from-absence reduction. All of that is recomputable.
  - It omits the DFF starter floor and the injury OUT flags, which have no history before 2025-26.
- **Variant b:** rate × calibrated minutes, where the calibration (actual minutes regressed on Formula C
  projection plus history) is fit on the training seasons.
- **Caveat:** complex minutes include injury bumps that Formula C training rows never show. Report complex and
  Formula C minutes separately.
- **Pass:** beat barebones-with-noise on projected-minutes FP MAE, with the 95% CI below 0. Otherwise keep the
  noise (seeded, per fix A).

### M2. Shrink projections before optimizing
**The repeat-pick pattern (checked 2026-10-05).** Our lineups reuse the same players far more than DFF's do:
- barebones lineups: Kel'el Ware 30% of slates, Mikal Bridges 26%; DFF's most-picked player is at 15%.
- (Davion Mitchell specifically was in only 2–10% of slates in every lineup set, archive included.)
- The repeat picks are players we project 1.5–5 FP above DFF: Ware +4.5, Keldon Johnson +3.9, Derik Queen
  +3.9, Schröder +5.0.
- They then fall short of our projection by about as much as everyone else (+3.4 vs +3.3 FP per slot). DFF's
  repeat picks *beat* DFF's projection (−2.5).
- This is the [optimizer's curse](https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/The_Optimizers_Curse.pdf):
  picking the maximum of noisy estimates selects the most over-estimated ones. A player whose features keep
  producing the same optimistic error gets picked night after night. The standard fix is shrinking estimates
  toward a prior, which is what M2 is.

Our lineups project about 262 FP and realize about 242 (+20). DFF projects 272 and realizes 265 (+7). Most of
the gap is projection bias amplified by picking the maximum.
- The M1 backtest showed the same thing. Barebones lineups projected 265 vs 238 realized, and the rate model
  287 vs 234.
- Shrink each projection toward the positional mean in proportion to its uncertainty; optionally optimize
  `proj_FP - λ·σ`.
- Backtestable now on the 100 stored slates with the M1 harness. Pass if realized lineup FP improves.

**Diagnostic (2026-10-06): where the lineup over-projection comes from.** Complex minutes + barebones FP,
54 slates, DFF-slate players only. Projected-minus-actual FP is split exactly into a **minutes** part
(projected minutes too high) and a **per-minute** part (projected FP per minute too high):

| | Players | Proj − actual | Minutes part | Per-minute part | DNP |
|---|---|---|---|---|---|
| Whole slate pool | 5,566 | −0.40 | +0.34 | −0.75 | 6.8% |
| Picked for lineups | 432 | **+3.35** | +2.33 | +1.02 | 4.2% |
| **Selection effect** | | **+3.76 per slot (~30 per lineup)** | **+1.98** | **+1.77** | |

- **The projections themselves are fine on average** (pool −0.4). The optimizer then picks the players whose
  errors happen to be positive, and it picks them on **both** parts, about half each. With Formula C minutes:
  +1.49 minutes, +1.95 per minute.
- **Cheap picks (<$4k): mostly minutes.** +4.6 FP from minutes, and a 21% DNP rate (15% with Formula C).
- **$5.5k+ picks: mostly per-minute** (+2.2 to +2.4).
- **Picks we projected 2+ minutes above their season average** overshoot more (+3.9 to +4.6, few players).
  Hypothesis, not yet tested: many are injury-redistribution beneficiaries.
- **Implication:** no single model fix removes it. Selection always finds whatever error remains, so the fix
  must make the projections less confident where they are least reliable. Priors come from **our own data**
  (no DFF; salary only if Ian OKs it):
  - **(2a) minutes:** shrink toward the player's season average, more strongly for big projected jumps,
    cheap players and few games;
  - **(2b) per minute:** shrink the season rate toward the career rate, by games played and volatility;
  - shrink strength fit on 2022-25, the standard empirical-Bayes setup.
- **Pass rule:** cut the selection effect clearly in the harness, and don't lower realized lineup FP.

**Root causes (2026-10-06).** All 432 picks (complex minutes + barebones FP, 54 slates) over-shoot the pool by
1,622 FP in total (+3.76 per slot). Share of that excess carried by picks with each condition (conditions
overlap):

| Condition | Picks | Avg error | Share of excess |
|---|---|---|---|
| **Injury-redistribution bump ≥0.5 min** (complex − Formula C) | 113 | +5.3 | **40%** |
| No DFF starter flag (bench) | 133 | +4.3 | 39% |
| Missed 1+ team games since last appearance | 42 | +5.4 | 15% |
| Thin sample (<10 games this season) | 21 | +8.5 | 12% |
| Returning after 8+ days | 16 | +10.4 | 11% |
| Hot season rate (season FP/min ≥ career +0.10) | 82 | +1.8 | 11% |
| **No condition at all** (pure selection noise) | 156 | +1.8 | 21% |

1. **Injury redistribution still overshoots, and the optimizer feeds on it.** Even at
   `INJURY_ADJUSTMENT_WEIGHT` 0.35, bumped players are over-projected across the whole pool: +1.2 min for
   0.5–3 min bumps, +3.0 for 3+. A bump raises the projection while the salary stays put, which is exactly
   what the optimizer looks for. Bumped picks: +3.9 / +7.9 min. This is the complex model's own
   redistribution.
2. **Frozen projections for players falling out of the rotation.** A DNP has no box-score row, so a benched or
   unlisted-injured player keeps his old averages. Pool players who missed 2+ team games are over-projected
   by about 3 min, with a 25% DNP rate. Example: Julian Reese was picked 4 times at 29.8 projected minutes
   (from 5 games), and didn't play any of them.
3. **Thin samples and returns from absence** (12% / 11%): small-sample averages and minutes restrictions.
4. **Per minute: season rates don't regress.** Picks' actual FP/min ran 0.034 below their season rate (pool:
   +0.009). The FP model tracks the season rate faithfully (+0.008); the season rate itself is the inflated
   input, picked when hot.
- Checked and ruled out: stars being under-projected. Pool error is −0.3 to +0.3 in every salary band.
- **Fix order (each tested in the harness on the selection-effect metric):** (1) the redistribution size
  for bumped players; (2) count missed team games in the minutes features; (3) M1b fitted baseline,
  including a games-played term; (4) regress the season FP/min rate toward career (Marcel-style, fitted).

**Fix #1 investigation on the replay table (2026-10-06): the root cause is DNP-blind baselines, not the boost.**
- **Setup:** 4 seasons; 5,853 team-games with a fresh rotation absence (S_MIN ≥ 15, played the previous team
  game); 65,743 eligible teammates. Complex's fresh-injury redistribution was emulated exactly (position
  overlap, 2× exact position, caps, first-absence-only, ×0.35 damping).
- **Raw comparison:** boosted teammates were predicted +1.54 min over Formula C and gained +0.90. A single
  weight scan (fit 2022-25, test 2025-26) prefers 0.20 over 0.35 (test MAE 5.707 vs 5.756; bias −0.05 vs
  +0.62).
- **But against normal nights (no absence), the boost is not too big.** Teammates truly gained +2.09 vs +1.54
  predicted: same position +2.65 (pred 2.04), adjacent +1.75 (1.24), no overlap +0.34 (0). The raw
  "overshoot" comes from the baseline the boost sits on.
- **The baseline over-projects bench players, and 94% of that is DNP nights.** On normal nights:

  | Formula C projected | DNP rate | Error, all nights | Error, nights played |
  |---|---|---|---|
  | 0–8 min | 64% | +3.64 | +0.04 |
  | 8–14 min | 35% | +3.96 | +0.16 |
  | 14–20 min | 4% | +0.20 | −0.42 |

  Formula C averages only the games a player appeared in, so it ignores how often he doesn't play.
  Players who appeared in ≤50% of team games: projected 7.2, actual 2.9, 61% DNP.
- **This one cause links three diagnostic flags:** bench picks (39%), frozen projections after missed
  games (15%), and much of the "redistribution" excess (40%), because boosted players are often bench
  players.
- **Revised fix order:**
  1. **DNP-aware baseline:** projected minutes = P(plays) × minutes-if-plays. P(plays) is fit on 2022-25 from
     appearance rate, team games missed and recent DNPs; minutes-if-plays is the current formula, which is
     accurate. Test on 2025-26 by FC band.
  2. **Re-tune the redistribution weight on top of that baseline.** The 0.20-vs-0.35 answer above is
     contaminated by the baseline bias.
  3. M1b weight fit (mid/high-minute shape: 20–26 under by 0.85, 26+ over by 0.5–0.8).
  4. Per-minute rate regression.
- The redistribution emulator is `scratchpad/redistribution.py`; it will move into `scripts/` with the fix.

**DNP-aware baseline: tested 2026-10-06; NOT shipping. Production's pool is already DNP-filtered.**
- **Method:** walk-forward on the replay table (fit 2022-23 → test 2023-24; 22-24 → 24-25; 22-25 → 25-26),
  not a random split. Random splits leak through the rolling features, and production always predicts
  forward. "Chance he plays" excludes likely-injury absences (a rotation player in a 2+ game absence
  streak), since the injury report zeroes those in production.
- **All non-injured roster players, every test season:** P(plays) × fitted minutes wins. MAE 4.96 / 5.19 /
  5.31 vs Formula C 6.04 / 6.08 / 6.17; bias about 0 vs +1.7.
- **Production pool (2025-26 players complex projected >0 and on the DFF slate, n=10,513): no gain.**
  MAE 5.66 vs 5.61, bias −0.82 vs +0.34.
  - **Why:** daily-predictions drops every player DFF projects at 0.0 (`data-ppg_proj != "0.0"`), so chronic
    DNPs never reach us. Pool DNP rate is 5.9%, vs 69% for the replay's lowest band.
  - **So the "DNP-blind baseline explains 39% + 15% + 40%" conclusion holds for the full roster, not for
    our pool.**
  - Even players who missed 2–3 team games: Formula C +2.95, P×fitted −1.26 (equal MAE). It over-corrects.
- **What survives:** the fitted minutes-if-plays weights are stable across all three folds: season
  0.23–0.28, last-7 0.38–0.45, previous game 0.24, plus a +2.2–2.6 return term (the ×0.75 return cut points
  the wrong way). On the production pool: MAE 5.573 vs 5.612, but bias +0.90 vs +0.34. Needs the lineup
  test.
- **Lesson:** decisions must be scored on the **production pool**, which exists only for 2025-26. Use the
  replay walk-forward for stability checks and fitting, and the lineup harness on 2025-26 to decide.

**Selection-aware shrinkage toward the player's own season average: tested 2026-10-06; no lineup gain.**
`scripts/backtest_shrinkage.py`, 2025-26 production pool, 101 slates. Barebones FP is regenerated (trained on
2022-25), because production only stored FP projections from January. The fractions are cross-fitted by
time halves.

| Variant | What picking adds | Player MAE | Realized lineup FP |
|---|---|---|---|
| No shrink | +3.53 / slot | 8.51 | 238.8 |
| One fraction (k ≈ 0.55–0.65) | **+2.73** | **8.44** | 238.8 (Δ 0.0, CI −5.1..+4.8) |
| Fraction per risk group | +3.34 | 8.49 | 238.9 (Δ +0.1) |

- **Shrinking makes projections more honest** (smaller overshoot, better MAE) **but the lineups don't score
  more.** It changes the lineup on 82% of slates and simply picks a different set of over-projected
  players. The per-group fractions are unstable between halves.
- **Realized lineup FP depends on ranking, not calibration.** The lineup gains points only if we're better at
  telling which players will beat their price. Shrinking toward a player's own average adds no new
  information about that.
- DFF's lineups realize about 265 vs our 239. That gap is ranking quality.
- Detectable effect size: across 101 slates the 95% CI on a lineup-FP difference is about ±5 FP, so smaller
  real gains can't be confirmed from one season.
- **Next:** candidates that add information or accuracy rather than calibration. Re-test the minutes ×
  FP-per-minute split with the fitted minutes; the fitted minutes weights; the injury boost weight.
  Separately, study the slates where our lineup and DFF's diverge: who was right, and what we got wrong.

### M3. Model DNP risk explicitly
5.5% of players projected over 10 minutes log zero; this was verified as real, not name mismatches.
- Last season's lineups started a player who didn't play 0.3–0.4 times per slate (DFF: 0.1–0.2). Most of ours
  came from the stale-injury months.
- Probably better served by L1/L2 than by a classifier, since a scratch is usually announced in text first.
  Revisit after L2 has data.

### M4. Cut dead features
- The nine cluster dummies are about 0.4% of importance combined.
- `IS_HOME` is real but small (+0.29 FP per game).
- `REST_DAYS` matters for minutes (+0.88 on back-to-backs), not for FP per minute.
- If fix B lands, barebones' features are `FP_PER_MIN`, `MIN`, `REST_DAYS` and `CLUSTER`. If dropping
  `CLUSTER` doesn't cost MAE in the harness, retire `cluster-scraper` → `nba-clustering` (two Lambdas and an
  ECR repo).

### M5. Opponent and pace adjustment
The `PACE` and `DEF_RATING` that `cluster-scraper` pulls are **player-level and season-to-date**, overwritten
on every run. They aren't opponent-team stats and have no history of past values to train on.
- The opponent join key exists: the schedule's `OPPONENT` column.
- Still needed: point-in-time team pace and defense, i.e. team stats as of each game date. A date-bounded
  `leaguedashteamstats` query might do it (not yet verified).

### M6. Team-minute constraint
Per-player bias tracks the team's projected total: 180 → −2.22, 230 → +0.10, 292 → +5.62. Naively rescaling to
240 makes MAE worse, because the slate covers only about 10 of about 15 rotation players.
- Needs a per-slate target based on how many players are covered.
- L1 flags extreme team totals in the meantime.

### M7. Blended early-season baseline (replaces the hard 4-game cutoff)
Players with 0–3 games this season get a flat 10 MPG (starter floor 20/24) in both `project_minutes_*` and
`injury_system`. That hits 2–3% of rotation players every week all season (returners, call-ups), not just
opening week. Measurements are under "Early-season pause".
- Replace the cutoff with a shrinkage blend: `w·this_season + (1−w)·prior`, with `w = n/(n+k)`. The prior is
  last season's average (same team), and `k` is fit by replaying 2025-26.
- This could also let the pause lift earlier. Do it in-season; it touches the core minutes path.

Return games after a long absence are the main place minutes restrictions show up. For rotation players
(20+ MPG), the first game back after missing 5+ games has 5.4–6.0 MAE vs 4.3–4.5 normally, and is
over-projected by 8+ minutes 16–21% of the time. But that is only 44 games and about 2% of total minutes
error (fresh window, Oct–Jan).

---

## LLM integration

The pipeline sees only what it scrapes. Every night some public information exists only as text: minutes
restrictions, load management, game-time decisions, rotation changes announced before the game. No amount of
box-score history recovers it.

Do **not** ask an LLM to emit numeric projections for production. L3 does exactly that, but only as a measured
experiment. The production value is text → structure.

**Models and cost**
- **Opus 5.5** (`claude-opus-5-5`, $4 / $20 per million tokens) for everything: research, L1, L2, L3 and L4.
  Fable was dropped 2026-09-28.
- Effort is always set explicitly; Opus 5.5 defaults to `medium`.
- Every call opts into server-side refusal fallbacks (`fallbacks: "default"`). A refused request may be retried
  on another model, which is logged as `served_model`. Calls raise on a refusal or a cut-off response.
- A **$25/day budget guard** stops further calls once reached. Cost per call is logged to
  `llm/usage/{date}.json`; web searches are billed separately and only counted.
- Measured so far: one preflight review costs $0.11. Real nightly cost is unknown until opening night.
- The key comes from `CLAUDE_API_KEY` (Lambda environment variable, or `.env` locally).

**Nightly schedule** (minutes after pipeline start, which is T−30 before the main slate):
- +12 `research`
- +13 minutes-projection: the lineup email you'd actually enter, at about T−17
- +16 `adjustments`
- +18 `preflight`: the verdict, at about T−12
- +22 `llm_lineups`: shadow only, never emailed, so its lateness doesn't matter
- Mondays: `postmortem`

Steps that depend on an upstream output poll for it for up to 5 minutes. The durations of the existing Lambdas
haven't been measured; check the first night's CloudWatch logs.

### Shared: nightly research — BUILT
One Opus 5.5 web-search call per game (up to 8 searches) writes a per-team briefing on availability,
restrictions, starting lineups and rotation news. It keeps the **exact snippet** behind each statement.
- Saved to `llm/research/{date}.json`; feeds both L2 and L3.
- One game failing doesn't discard the others. The action saves what succeeded, then raises.

### L1. Pre-flight sanity checker — BUILT
**Coded rules first**, with thresholds measured on 2025-26 fresh-input slates:

| Rule | Severity | Fires when |
|---|---|---|
| `slate_coverage` | critical | fewer than 90% of DFF slate players have a projection |
| `fp_models_zero` | critical | most players have 0 FP from all three models (the Jan 7–16 failure) |
| `no_lineups` | critical | no in-house lineup was built today |
| `zeroed_regular` | critical | projected 1 minute or less, averaged 20+ over the last 3 games, and not on the injury report |
| team totals outside 150–290 | critical | the 1st/99th percentile of 2025-26 slate totals |
| team totals outside 190–274 | warning | the 5th/95th percentile |
| `above_role` | warning | projection 10+ minutes over the season average (historically about 6.7 minutes too high) |
| stale box scores | critical | scheduled games since the last box score are missing |
| stale injuries | critical | the report date, or the file's write time, is over 24 hours old |
| `no_schedule` | warning | the schedule file is missing |
| status conflicts | warning | L2 found a reported status contradicting our projection |

**Then an Opus 5.5 review** of the full slate table: salary, DFF FP, our FP and minutes, Formula C and LLM
minutes, last 5 games, injury status.
- It's asked only for what the rules missed, and must cite numbers from the table.
- Its findings are stored separately from the rule findings.

**Output:** an email titled "CRITICAL - check before using lineups" or "ok", plus `llm/preflight/{date}.json`.
While paused, it runs input checks only and emails only if something is critical.

**Success:** at least one real defect a month that **no rule** caught. Compare `llm_findings` with
`rule_findings`.

### L2. Beat-writer synthesis → bounded minutes adjustments — BUILT, shadow mode
One judge: Opus 5.5 at effort high. Skipped while the early-season pause is on. Guardrails, enforced in code:
- Each adjustment cites one snippet, with a verbatim quote of at least 20 characters. The quote is checked
  mechanically against the snippet; paraphrases are rejected.
- The source URL is required.
- The size is clamped to **±25%**.
- A player projected at 0 can't be adjusted, and unknown players are rejected.
- Pre- and post-adjustment minutes are logged to `llm/adjustments/log.parquet`. **Nothing is applied.**
- Reported status conflicts ("we project him, the report says out") go to L1, not into adjustments.

**Go/no-go** (`scripts/evaluate_llm.py`): ship only after **30+ slates**, with the 95% bootstrap CI of the
adjusted-minus-unadjusted absolute error **entirely below 0**. The earliest decision is about early-to-mid
December, since collection starts when the pause lifts (about Oct 30).

**Precision** (the share of adjustments a human would endorse): run `--export-review review.csv`, fill in
ENDORSED y/n, then run `--labels review.csv`.

**X / Grok check, at the same time** (researched 2026-10-02):
- **Why consider it:** X is where beat writers first post restriction and availability news.
- **Claude's web search mostly can't see it live.** In a live test (about $0.40), search restricted to x.com
  returned 29 X posts, but only 1 was from that week. Unrestricted search returned 0 X posts, and got the same
  news secondhand via RotoWire and ESPN.
- **Grok's X Search reads X directly.** It costs $5 per 1,000 posts fetched plus tokens; date filtering is by
  whole day only; up to 20 handles can be allowed
  ([docs](https://docs.x.ai/developers/tools/x-search)).
- **The official X API** has full-archive search with exact timestamps, at $0.005 per post read.
- **Decision rule:**
  - If L2 adjustments don't beat no adjustment, drop X: a better source won't rescue them.
  - If they do, take our biggest over-projections (8+ minutes) and use the X API archive to check whether a
    trusted account posted the news before tip while our research missed it.
  - Add Grok only if that's a meaningful share, as a second evidence source for the same judge, scored with the
    same metric.
- **Sentiment analysis was rejected.** The evidence is weak and old ([RIT](https://www.rit.edu/news/tweets-predict-nba-player-performance-says-expert)).

### L3. LLM vs. the whole pipeline — BUILT, prospective only
Opus 5.5 projects minutes for every slate player from the research briefing plus public slate facts only:
names, teams, positions and salaries. It never sees our projections or features.
- Saved as a minutes model in `model_comparison/llm_head_to_head/`.
- `minutes-projection`'s `llm_lineups` action runs our FP models and optimizer on those minutes, so it's also
  scored on realized lineup FP. Those lineups are never emailed, and are skipped while paused.
- `scripts/evaluate_llm.py` reports MAE and bias against complex and Formula C, **split by players with and
  without news** that night, plus realized lineup FP against our lineups and DFF.
- No historical backtest: web search on past dates returns the box scores, and the model's training data may
  include 2025-26.
- Prior stated up front: the LLM loses on aggregate MAE and wins on players with news. If it wins outright,
  the statistical pipeline needs rethinking.

### L4. Weekly post-mortem analyst — BUILT
On Mondays, Opus 5.5 (effort high) analyzes projections vs actuals over the season, with the last 7 days
flagged. It can see the data only through a `run_query` tool that the Lambda executes.
- Unsafe filter expressions are refused, and each query is logged with its row count.
- Every finding must cite query ids that actually ran, and its row count is checked against the log. Findings
  that cite no executed query are rejected.
- Output: an email plus `llm/postmortem/{date}.json` with the full query log, so every finding can be re-run.

---

## Operations

### O1. Retune `INJURY_ADJUSTMENT_WEIGHT` on fresh data
Currently 0.35, fit on the 2025-26 fresh-injury window. Re-fit after about 30 slates of 2026-27 with a working
injury feed.

### O2. Alarm on data staleness, not just Lambda errors — mostly covered by L1
L1 checks the dates *inside* the data: box scores missing scheduled games, the injury report's own date
(`REPORT_DATE`), the injury file's age, and a missing schedule. Those findings arrive in the preflight email
but **don't set off a CloudWatch alarm**. If L1 itself fails, its Lambda error alarm fires. The remaining
option is a standalone check that raises, independent of the LLM Lambda.

### O3. Pin transitive dependencies
Two of nine rebuilds failed on 2026-09-18/19 because unpinned transitive dependencies resolved to versions
with no Lambda-compatible wheel (Pillow, scipy). The 2026-09-29 rebuild of five functions succeeded, but the
risk is unchanged.
- Remaining drift: pandas 2.1.3/2.1.4, pyarrow 14.0.1/14.0.2/20.0.0, and boto3 unpinned in three scrapers.
- `llm-analyst` shows the approach: resolve with
  `pip download --platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all:`, then pin
  everything. Do this before the next mid-season rebuild.

### Preseason / no-game days (fixed 2026-10-05)
DFF posts preseason slates, so from Oct 3 the whole pipeline ran nightly. cluster-scraper (no stats yet),
daily-predictions (no DFF projections), injury-scraper (no reports yet) and box-score-scraper (see below)
all alarmed. No harm was done: with no DFF players, research, L2, preflight and the lineups had nothing to
work on, and there was no LLM spend.
- **Fix:** box-score-scraper saves the season's game dates to `data/schedule/game_dates.json`, and
  game-scheduler skips any day before the first regular-season game, or with no games (the All-Star break).
  - A missing file, a file from last season, or a date after the last scheduled game falls back to the old
    behavior. Next preseason therefore runs one night, saves the new season's dates, and skips from then on.
  - Tested on 9 date cases. Bootstrapped from the schedule saved 2026-10-05 (156 dates, 10/20 to 4/11).
- **box-score-scraper failures were the proxy, not NBA blocking.** In a local test of 12 fetches through the
  proxy, 10 returned HTTP 200 with full data and 2 dropped (a timeout and a TLS EOF), with no 403/429. Every
  NBA API call now retries 3 times (backoff 5 s, 15 s).
  - It still re-downloads three finished seasons every night. Loading those from S3 would cut the exposure
    further; not done.

### Redeploying
`python lambda/deploy.py <function>` needs `.venv/Scripts` first on PATH (PowerShell:
`$env:PATH = "$PWD\.venv\Scripts;$env:PATH"`). Otherwise the `aws.cmd` it finds runs the system Python, which
has no `awscli`, and the ECR login fails. A new function needs its ECR repo created first; deploy.py creates
neither repos nor functions. Lambda environment variables (`PROXY_URL`, `CLAUDE_API_KEY`) are set by hand.

---

## Reference: early-season pause (built 2026-09-28)

**The problem it avoids.** `current.parquet` holds the latest season *with games*. Once 2026-27 has games:
- players without one are dropped as "free agents" (109 of 117 on a synthetic night 2);
- players with 0–3 games get a flat 10 MPG (20/24 if DFF lists them as a starter);
- players with no current-season row get no FP features;
- offseason movers sit on their old team.

Four targeted patches were rejected as bolt-ons; skipping the first ~10 days avoids all of it.

**Rule** (`serving_features.projection_gate`, mirrored in `llm-analyst/slate.py`; the constants live in both
`config.py` files and must be kept in sync): paused until **all 30 teams have 4+ games of today's season**
before today. The season is derived from the date, so opening night (when the file still holds 2025-26) is
paused too.
- Replayed on 2025-26: paused Oct 21–29, **opens Oct 30** (day 10). Expect about Oct 29–30, 2026.

**While paused:**

| Step | Behavior |
|---|---|
| minutes-projection | Actuals update, then the **DFF lineup only**, emailed with a "projections paused (reason)" header |
| `llm_lineups` | Skipped |
| research + L3 minutes | **Run normally**, so L3 collects data from night 1 |
| L2 `adjustments` | Skipped |
| L1 `preflight` | Input checks only; **emails only if critical**; no LLM review |
| supervised-learning, scrapers | Unchanged |

**Not fixed by the pause: the <4-game 10 MPG fallback is a season-long problem.** Share of rotation players
(15+ min that night) with fewer than 4 games, 2025-26:

| Week | Share |
|---|---|
| Oct 27–Nov 2 (gate opens Oct 30) | 12.8% |
| Nov 3–9 | 6.6% |
| Nov 10–Dec 7 | 1.7–3.3% every week |

See M7.

---

## Done log

### 2026-10-05
- Investigated the nightly alarms (preseason runs, plus proxy drops); fixed both (see Operations), deployed
  game-scheduler and box-score-scraper.
- The Monday postmortem (L4) ran for $0.58 and worked as designed. It had nothing to grade, so it used last
  season's final week. Two new points to check:
  - Its DNP flag mixes real DNPs with ungraded rows, so about 750 real DNPs are missing from its minutes
    errors (fix in `postmortem.build_datasets`).
  - 312 player-games where DFF listed a starter, we projected about 0 minutes, and he played about 28.
    Probably stale OUT flags; check whether the 2026-09-20 stale-OUT fix covers them.

### 2026-10-02
- Researched X/Grok; rejected adding it now. The decision rule is under L2.
- M1 backtest failed. It confirmed two training-pipeline problems (fix A), and showed barebones is best or tied
  (fix B). Researched how others handle predicted inputs (M1b).
- Cleaned up this roadmap.

### 2026-09-28/29
- **Early-season pause** built in minutes-projection and llm-analyst. The DFF lineup code moved into
  `build_dff_lineup()` so it runs whether or not projections are paused; behavior unchanged.
- **Fable removed**; everything runs on Opus 5.5. `ADJUSTMENT_JUDGES` still accepts more models.
- Verified that Opus 5.5 pricing in `config.py` matches the API docs, and that the L4 tool loop only appends to
  history.
- The API key is read from `CLAUDE_API_KEY`; local runs load it from `.env`.
- The email subject now counts only lineups actually built.
- **Deployed** box-score-scraper, minutes-projection, supervised-learning, injury-scraper, game-scheduler, and
  the new `llm-analyst` (image, 900 s, 1024 MB). Created its ECR repo and the
  `nba-lambda-errors-llm-analyst` alarm, and applied EventBridge invoke permissions to all 8 pipeline
  functions.
- **Live smoke test passed:** preflight on 2026-01-15 for $0.11. Its first run caught a real bug
  (`slate.tonight()` crashed when the schedule file was missing), which was fixed and redeployed. The offline
  suite is now 60/60.
- `scripts/fix_eventbridge_lambda_permissions.ps1` fixed for a single matching statement under strict mode, and
  for functions with no policy yet.

### 2026-09-22
Each item was verified by running the real Lambda code read-only on S3 data unless noted.

**Serving bugs (all projections)**
- **Served features were one game stale.** Rolling features use `shift(1)`, but serving read each player's
  latest row, so the most recent game was left out of every average.
  - New `minutes-projection/serving_features.py` advances the features. Checked 100% exact on 582 players.
  - Gain: Formula C MAE 4.952 → 4.907; FP models −0.011 to −0.018 (only barebones significant).
- **FP_PER_MIN cutoff mismatch.** Training used the career rate for a player's first 2 games; serving used it
  for the first 5. Serving now matches training (`config.CAREER_RATE_MAX_PRIOR_GAMES = 1`). About 2,224
  player-games per season were affected.
- **`IS_HOME` was always 0 at serve time.** box-score-scraper now saves the season schedule
  (`scheduleleaguev2` via the proxy) to `data/schedule/current.parquet`, and minutes-projection maps it by team.
  - It agrees with box-score `MATCHUP` on 2,455 of 2,460 team-games; all 5 misses are neutral-site games.
  - Home is worth +0.29 FP per game (CI +0.18 to +0.41), all from efficiency. The models learned +0.15 to +0.18.
- **Rest days in training.** Every player's final training row was overwritten with "today minus last game"
  (575 of 582 wrong). Fixed. On back-to-backs, players who suit up log +0.88 minutes, with no change in FP per
  minute.
- **Crash on an empty injury file** (no `STATUS` column). Fixed.
- **Local training scripts:** `scripts/supervised_learning.py` was deleted (random split, and it published
  pickles from a local environment). The training Lambda's local `__main__` never publishes.

**LLM components: all four built** in a new `lambda/llm-analyst`.
- game-scheduler runs the new steps, and minutes-projection gains `llm_lineups`.
- `actuals_updater` and `evaluate_projections.py` include `llm_head_to_head`.
- injury-scraper saves `REPORT_URL`, `REPORT_DATE` and `REPORT_TIME`.
- `llm-analyst/requirements.txt` pins every dependency, transitive ones included.

**Findings from the historical data**
- **Silent FP-model failure in production:** all three FP models returned 0 on **9 slates, Jan 7–16, 2026**.
  L1's `fp_models_zero` rule now catches this.
- DNP rate for players projected over 10 minutes: 5.5%; only 37 of 556 are name mismatches.
- The Oct 24, 2025 DFF slate has no positions, the only one of 191.
