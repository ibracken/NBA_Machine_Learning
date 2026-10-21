# Minutes Projection System

## Overview

The minutes-projection Lambda builds three DraftKings lineups each night and emails them:

1. **Learned minutes (our #1):** a gradient-boosted minutes model (`lambda/shared/learned_minutes.py`), trained
   nightly by supervised-learning.
2. **Formula C (control):** `0.5·season + 0.3·last-7 + 0.2·previous game`, capped at 37; a flat 10 with fewer than
   4 games; ×0.75 after missing 10+ team games.
3. **DailyFantasyFuel:** DFF's own projections.

**FP = projected minutes × FP per minute.** For the learned lineups, FP per minute is the season rate × tonight's
pace / opponent-defense multiplier (below). Formula C keeps the plain season rate, so the control is unchanged.
Lineups use the production optimizer (`lineup_optimizer.optimize_lineup`) with DFF's salaries and positions.

## Pace and opponent defense

`lambda/shared/pace_defense.py`, added 2026-10-20 (`findings.md`, "Pace and opponent defense").
- **Team ratings from the current season's box scores before tonight:**
  - pace (possessions per 48);
  - FP allowed per 100 possessions;
  - each shrunk to the league mean with 10 games' weight.
- **Multiplier** = 1 + a·log(expected pace / pace of his games so far) + b·log(opponent FP allowed / that of the
  opponents he faced).
  - The weights a and b come from `models/pace_defense.json`, fit nightly by supervised-learning on the same
    training rows as the minutes model.
  - A player with no game yet this season gets ×1.
- **If the weights file is missing:** learned FP uses ×1 (the plain season rate), the lineups are still built and
  emailed, and the run raises so the error alarm fires.

Removed 2026-10-18: the complex injury-redistribution model and the `current` / `fp_per_min` FP models. The
barebones FP model is kept but not served (`process_model.py`, `lineup_optimizer.predict_fantasy_points`), and the
cluster pipeline that fed it is switched off. Backtest results are in `findings.md`.

## The learned minutes model

One shared module is used by training (supervised-learning), serving (this Lambda) and the backtests (`scripts/`),
so production computes what was tested. On 2025-26 it reproduced the backtest table and features exactly, and this
Lambda picked the same 8 players as the backtest on every night tested.

- **Rows:** every player on the roster of a team playing tonight, including players who don't play. A player is on
  a team's roster while his latest appearance this season was for that team.
- **Inputs:**
  - the player's season, last-7 and previous-game minutes, games played, team games missed;
  - season minutes of teammates listed Out or Doubtful on tonight's injury report, split into "played the team's
    last game" and "already out";
  - his with/without history: how his minutes changed in earlier games this season when each of tonight's out
    teammates sat, shrunk by n / (n + 3);
  - his own status on tonight's report: Questionable, Probable, Available, or not listed (since 2026-10-20).
- **Two parts (2026-10-20):** `P_PLAY`, the chance he plays, and `MIN_IF_PLAY`, his minutes if he plays (trained
  on players who played). Projected minutes = `P_PLAY` × `MIN_IF_PLAY`. The pickle is a dict
  `{'play', 'minutes', 'features'}`; `lm.predict` still accepts an older single model.
- **Out set:** teammates listed Out or Doubtful, with a season average of 10+ minutes, who missed no more than 10
  team games.
- **Training:** the current season and the three before it, every roster row not listed out on a day with an
  injury report (`data/injuries/history.parquet`, archived 2022-26 pre-tip reports plus each night's report).
- **Absence adjustment:** for each player, the projection minus the same model with no absences
  (`ABSENCE_ADJ_MIN`). The AI adjustments judge reads it.

If the injury report isn't from today, nobody is treated as out (logged). Players with no game yet this season
have no projection.

## Early-season pause

`serving_features.projection_gate` pauses in-house projections until all 30 teams have played 4 games of the
current season (about 10 days; Oct 30 in 2025-26). While paused, the default action updates actuals, then builds
and emails only the DFF lineup. See `findings.md`, "Reference: early-season pause".

## Actions

- `project` (default): update actuals, then build and email the three lineups.
- `adjusted_lineup` (+19, after the AI judge): waits for the judge. Its quote-verified status calls change the out
  list and the model is re-run; its minutes tweaks (at most ±25%) are applied on top. Builds the AI-adjusted
  lineup (`model_comparison/learned_ai_adjusted/`) and emails it with what changed and why, plus the AI's own lineup
  (`model_comparison/ai_lineup/`, from llm-analyst research). While paused, the email carries only the AI's lineup.
- `update_actuals`: fill ACTUAL_MIN / ACTUAL_FP from the box scores (names matched per date with
  `lambda/shared/player_names.py`; a projected player with no box-score row on a graded date gets 0).
- `check_model`: load the published model and predict the last graded game day. No writes, no email. Use it after
  publishing a model (AGENTS.md release checklist).

## Inputs (S3)

- `data/box_scores/current.parquet`: current season
- `data/schedule/current.parquet`: which teams play tonight
- `data/injuries/report_statuses.parquet`: tonight's report, every status (`REPORT_DATE` must be today). Out and
  Doubtful make the out set; Questionable / Probable / Available are model inputs
- `data/daily_predictions/current.parquet`: DFF slate, salaries, positions, projections
- `models/learned_minutes.pkl`, with `models/learned_minutes_feature_names.json` and
  `models/learned_minutes_summary.json`

## Outputs (S3)

- `model_comparison/learned_minutes/minutes_projections.parquet`: every roster player tonight: `DATE`, `PLAYER`
  (box-score spelling), `DFF_NAME`, `TEAM`, `PROJECTED_MIN`, `PROJECTED_FP`, `S_RATE`, `S_MIN`, `L7_MIN`, `GP`,
  `TEAM_GAMES_MISSED`, `OUT_TONIGHT`, `STATUS`, `P_PLAY`, `MIN_IF_PLAY`, `LEARNED_MIN_NO_ABSENCES`,
  `ABSENCE_ADJ_MIN`, absence features, `OPP`,
  `EXP_PACE`, `DEF_IDX`, `PACE_RATIO`, `DEF_RATIO`, `RATE_MULT`, `FP_RATE` (= `S_RATE` × `RATE_MULT`),
  `ACTUAL_MIN`, `ACTUAL_FP`
- `model_comparison/learned_minutes/daily_lineups.parquet` (with `RATE_MULT`; the AI-adjusted files carry the
  same pace columns)
- `model_comparison/formula_c_baseline/minutes_projections.parquet` and `daily_lineups.parquet`. The lineup file
  also holds pre-January-2026 lineups from before the FP-model split.
- `model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet`
- `data/daily_predictions/current.parquet`: today's rows get the learned `PROJECTED_MIN` and `TEAM_ABBREVIATION`

Lineup columns: `DATE`, `SLOT`, `PLAYER` (DFF spelling), `TEAM`, `POSITION`, `SALARY`, `PROJECTED_MIN`,
`PROJECTED_FP`, `ACTUAL_FP`.

Files from retired models (`complex_position_overlap/`, `direct_position_only/`, `*/fp_*/`, `llm_head_to_head/`,
`injury_context/`) are complete 2025-26 history and no longer written.

## Dependency constraints (Lambda)

Pinned: numpy==1.24.3, scipy==1.11.4, pandas==2.1.3, scikit-learn==1.5.2 (Python 3.11), plus unidecode for the
shared name matcher. Models are trained only in the supervised-learning Lambda, never published from a local run.
