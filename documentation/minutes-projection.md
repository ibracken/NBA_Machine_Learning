# Minutes Projection System

## Overview

The minutes-projection Lambda generates NBA minutes projections and then builds DraftKings lineups using multiple fantasy-point (FP) models. It produces:

Minutes models (3):
1. complex_position_overlap
2. direct_position_only
3. formula_c_baseline

FP models (3) applied to each minutes model for lineup optimization:
- current
- fp_per_min
- barebones

Additional lineup-only baseline:
- daily_fantasy_fuel_baseline (DFF)

Total lineups per run: 9 (minutes x FP) + 1 (DFF) = 10.

## Inputs (S3)

- Daily predictions (salary/position/opponent context):
  - data/daily_predictions/current.parquet
- Injury data:
  - data/injuries/current.parquet
- Box scores (current season):
  - data/box_scores/current.parquet
- Box scores (previous season, for season-long returns):
  - data/box_scores/2024-25.parquet
- Supervised-learning FP models (pickled sklearn models):
  - models/current.pkl
  - models/fp_per_min.pkl
  - models/barebones.pkl

## Minutes Modeling

### complex_position_overlap
- Allows adjacent position overlap (PG/SG, SG/SF, SF/PF, PF/C).
- Exact position match gets a multiplier in injury redistribution.

### direct_position_only
- Exact position match only.

### formula_c_baseline
- No injury redistribution. Uses the statistical baseline formula only.

## Injury-aware logic (high level)

- Injured players with return_date < today are removed from the injury set.
- Players returning today remain in the pool but receive a 25% minutes reduction.
- Injured players marked OUT redistribute minutes to eligible teammates based on position overlap and baseline weights.
- Beneficiaries use a frozen baseline to avoid compounding inflation.
- Ex-beneficiaries exclude inflated periods from baseline calculation.
- Players returning from season-long absences get a conservative baseline for 4 games.

## Lineup generation

For each minutes model, the system:
1. Updates projected minutes in daily_predictions for the current slate date.
2. Predicts FP with each FP model.
3. Optimizes a DraftKings lineup for each FP model.

DFF lineups are built from scraped DFF projections (no minutes model).

## Outputs (S3)

### Minutes projections
- model_comparison/{model_name}/minutes_projections.parquet

model_name is one of:
- complex_position_overlap
- direct_position_only
- formula_c_baseline

### Lineups
- model_comparison/{model_name}/fp_{fp_model}/daily_lineups.parquet

Where:
- model_name is one of the three minutes models above
- fp_model is one of: current, fp_per_min, barebones

DFF lineups:
- model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet

### Injury context
- injury_context/{model_name}.parquet

Saved for complex_position_overlap and direct_position_only.

## Columns (key outputs)

Minutes projections (minutes_projections.parquet):
- DATE
- PLAYER
- TEAM
- POSITION
- PROJECTED_MIN
- ACTUAL_MIN
- PROJECTED_FP
- ACTUAL_FP
- CONFIDENCE

Daily lineups (daily_lineups.parquet):
- DATE
- SLOT
- PLAYER
- TEAM
- POSITION
- SALARY
- PROJECTED_MIN
- PROJECTED_FP
- ACTUAL_FP

## Dependency constraints (Lambda)

Lambda must use NumPy 1.24.x-compatible wheels; higher NumPy can break sklearn pickles.
Recommended pins for Lambda:
- numpy==1.24.3
- scipy==1.11.4
- pandas==2.1.3
- scikit-learn==1.5.2

If FP models are re-trained locally, re-pickle them in a Lambda-compatible environment
before deploying to avoid unpickle errors.
