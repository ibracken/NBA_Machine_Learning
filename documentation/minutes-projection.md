# Minutes Projection System

## Overview

The minutes-projection lambda generates NBA player minutes predictions using injury-aware modeling. It produces 3 minutes models and, for each, 3 fantasy point (FP) variants from the supervised-learning models:

Minutes models:
1. **Complex Position Overlap** - Adjacent positions can substitute (PG/SG, SG/SF, SF/PF, PF/C), 2x multiplier for exact position match
2. **Direct Position Exchange** - Exact position match only, no multiplier
3. **Formula C Baseline** - No injury handling, pure statistical formula

FP variants (applied to each minutes model for lineup optimization):
- **current**
- **fp_per_min**
- **barebones**

Additional lineup-only baseline:
4. **DailyFantasyFuel Baseline** - DFF fantasy projections (no minutes projections)

## Data Sources

### Injury Data (from injury-scraper)
- **Source**: ESPN injury page (https://www.espn.com/nba/injuries)
- **Output**: `s3://nba-prediction-ibracken/data/injuries/current.parquet`
- **Key Columns**:
  - `PLAYER` - Normalized name (lowercase, unidecode)
  - `TEAM` - NBA team abbreviation
  - `STATUS` - OUT, QUESTIONABLE, DOUBTFUL, PROBABLE, DAY-TO-DAY
  - `RETURN_DATE_DT` - Parsed return date (players with return_date < today are filtered out; return_date = today are kept for 25% reduction)
  - `ESTIMATED_INJURY_DATE` - Last game date + 1 day (for baseline calculations)
- **Filtering**: Stale injuries (return_date < today) are removed. Players returning today (return_date = today) are kept and receive 25% minutes reduction.

### Box Scores
- Current season: `data/box_scores/current.parquet`
- Previous season: `data/box_scores/2024-25.parquet` (for season-long injuries)

## Core Projection Scenarios

### 1. Season-Long Returns
**Trigger**: Player has 0 games this season (GAMES_PLAYED=0), FROM_PREV_SEASON=True, STATUS ≠ 'OUT', CAREER_AVG_MIN ≥ 20

**Example**: Kawhi Leonard averaged 30 MPG last season, missed all season, now healthy

**Action**:
- **Returning player** gets conservative **10 MPG for first 4 games**
- **Affected teammates** (same/adjacent position) also get **10 MPG for first 4 games**
- Affected teammates marked as **EX_BENEFICIARY** to exclude inflated games from baseline calculation
- After 4 games, system has enough data for normal projections

**Why**: Prevents overestimation when rotation changes dramatically. Teammates' season averages are inflated by playing extra minutes all season.

### 2. Active Injuries (Redistribution)
**Trigger**: Teammate has STATUS='OUT' and RETURN_DATE_DT > today

**Action**:
- Get injured player's TRUE baseline (pre-injury minutes)
- Find teammates at same/overlapping positions (based on model's position overlap function)
- Calculate TRUE baselines for each teammate using injury context
- Distribute lost minutes weighted by baseline + BENCH_OPPORTUNITY_CONSTANT (0.1)
- Apply 2x multiplier for exact position match (complex model only)
- Handle overflow from 37-minute cap by redistributing to non-capped players
- Track beneficiaries in injury context (if boost > 3 MPG)

**Special Cases**:
- **Season-long injuries (0 games this season)**: Skip redistribution entirely. Teammates' SEASON_AVG_MIN already accounts for their absence all season.
- **Players returning today**: Excluded via `RETURN_DATE_DT <= today` filter - don't redistribute their minutes since they're playing.

### 3. Players Returning Today (25% Reduction)
**Trigger**: Player has STATUS (OUT/QUESTIONABLE/etc.) and RETURN_DATE_DT = today

**Example**: LeBron James listed as OUT yesterday, expected to return today (still on injury report when scraped)

**Action**:
- Player is NOT given 0 minutes (they're playing)
- Player gets normal projection (redistribution or Formula C)
- **25% reduction applied** to final projection (accounts for possible minutes restrictions)
- Player's minutes are NOT redistributed to teammates (they're playing)

**Rationale**: Players returning from injury often face minutes restrictions in their first game back. ESPN's injury report shows return_date = today, meaning they're expected to play but may not be at full capacity. The 25% reduction accounts for coach caution without needing explicit minutes restriction data.

**Example Flow**:
- LeBron's normal projection: 32 MPG
- Apply 25% reduction: 32 × 0.75 = 24 MPG
- Realistic expectation for first game back

### 4. Injury Context Tracking

The system maintains injury context state to handle complex scenarios:

#### BENEFICIARY
- Currently filling in for injured player(s)
- Uses frozen **TRUE baseline** from before they started filling in
- Prevents compounding inflation (baseline stays fixed during fill-in period)
- **Can have multiple rows per player** - allows tracking benefits from multiple simultaneous injuries
- **Duplicate prevention** - checks before adding to prevent duplicate relationships
- Example: Player C filling in for both injured Player A (+5 MPG) and injured Player B (+8 MPG) has 2 BENEFICIARY records

#### EX_BENEFICIARY
- Previously filled in for injured player who has now returned
- **Excludes inflation periods** from baseline calculation
- Can have multiple records per player (multiple injury periods)
- Uses **2.0x recency weight** for games in last 14 days for faster convergence
- Example: Norman Powell filled in for Kawhi (Nov 1 - Jan 15). His baseline calculation excludes those games to get his "normal" minutes.

#### RETURNING
- Player returning from season-long injury (tracked for 4 games)
- Conservative 10 MPG until system has real data
- **Preserves BENEFICIARY and EX_BENEFICIARY records** - only replaces old RETURNING rows
- Allows a returning player to simultaneously benefit from other injuries
- Removed from tracking after 4 games played

### 5. Normal Play (No Injury Impact)
**Formula C**: `0.5 × baseline + 0.3 × last_7 + 0.2 × prev_game`

**Baseline Priority**:
1. SEASON_AVG_MIN (if GAMES_PLAYED ≥ 4) - Reliable current season data
2. Previous season AVG (if FROM_PREV_SEASON=True and same team) - Season-long injury, same team
3. Conservative 10 MPG - Insufficient data, traded players, unreliable scenarios

## Key Algorithms

### TRUE Baseline Calculation
Prevents compounding minutes problem when players fill in for injuries:

```python
# For BENEFICIARY: Use frozen baseline from before injury
# For EX_BENEFICIARY: Exclude games between INJURY_DATE and RETURN_DATE
# Apply 2.0x weight to games in last 14 days (faster convergence)
# Fallback to conservative 10 MPG if insufficient valid games
```

### Redistribution Weighting
```python
base_weight = baseline + BENCH_OPPORTUNITY_CONSTANT (0.1)
if use_multiplier and player_position == injured_position:
    weighted = base_weight * 2.0  # Complex model only
```

### Position Overlap
- **Complex**: PG/SG, SG/SF, SF/PF, PF/C adjacencies
- **Direct**: Exact position match only

## Constants

- `MAX_MINUTES = 37` - Individual player cap
- `BENCH_OPPORTUNITY_CONSTANT = 0.1` - Tiny boost to prevent zero-weight for bench players
- `EXACT_POSITION_MULTIPLIER = 2.0` - Direct backups get 2x weight (complex model)
- `CONFIDENCE_CLEARANCE_GAMES = 3` - Games needed to clear LOW confidence
- `CAREER_AVG_MIN >= 20` - Threshold for season-long return detection (filters bench players)

## Confidence Levels

- **LOW**: Player involved in active injury scenario (< 3 games since injury)
- **MEDIUM**: 1-2 games since injury/return
- **HIGH**: 3+ games since injury or no injury impact

*Note: Confidence is metadata only, does not affect projection calculations.*

## State Transitions

```
New injury detected:
  Injured Player → (tracked in games_log)
  Teammates → BENEFICIARY (frozen baseline, creates new row if doesn't exist)

Multiple injuries affect same player:
  Player C benefits from injury A → BENEFICIARY (of A)
  Player C benefits from injury B → BENEFICIARY (of B) [2nd row, not replacement]

Injured player returns (partial):
  Specific BENEFICIARY row → EX_BENEFICIARY (for that specific injury)
  Other BENEFICIARY rows remain active (player still filling in for other injuries)

Injured player returns (all):
  Last BENEFICIARY → EX_BENEFICIARY (exclude inflation period)
  Injured Player → Normal play (remove from injury context)

Season-long return detected:
  Returning Player → RETURNING (10 MPG for 4 games)
  Affected Teammates → EX_BENEFICIARY (exclude inflation period)
  If returning player is also BENEFICIARY → keeps BENEFICIARY rows active

After 4 games:
  RETURNING → Normal play (remove RETURNING status only)
  BENEFICIARY/EX_BENEFICIARY rows preserved
```

## Edge Cases

1. **Traded players during injury**: Use conservative 10 MPG (previous season data unreliable)
2. **Players returning today**: Excluded from redistribution via RETURN_DATE_DT filter
3. **Deep bench false positives**: Season-long return requires CAREER_AVG_MIN ≥ 20
4. **Players with no NBA games**: Excluded by injury-scraper (G-League/rookies)
5. **37-minute cap overflow**: Redistributed to non-capped teammates using same weighting

### Recent Edge Case Fixes (2025-11-26)

#### Multiple Simultaneous Beneficiary Tracking
**Problem**: Player could only be tracked as BENEFICIARY of one injury at a time. When a second teammate got injured, the first BENEFICIARY record was deleted.

**Solution**: `add_to_injury_context()` now allows multiple BENEFICIARY rows per player (similar to EX_BENEFICIARY). When Player C benefits from injuries to both Player A and Player B, two separate BENEFICIARY records are maintained.

**Impact**: More accurate minute projections when multiple injuries affect the same backup player.

#### RETURNING Status Preservation
**Problem**: When a returning player (status=RETURNING) was also benefiting from other injuries, marking them as RETURNING would delete all their BENEFICIARY records.

**Solution**: RETURNING status now only removes old RETURNING rows, preserving BENEFICIARY and EX_BENEFICIARY records. This allows a player to simultaneously be returning from injury while filling in for other injured teammates.

**Impact**: Correctly handles complex scenarios where a returning player is also a beneficiary.

#### Duplicate BENEFICIARY Prevention
**Problem**: Running the lambda multiple times while the same player was injured could create duplicate BENEFICIARY records for the same relationship.

**Solution**: Before adding a BENEFICIARY record, the system now checks if the same relationship already exists (same player, same injured teammate). Only creates the record if it doesn't exist.

**Impact**: Cleaner injury context data without duplicate tracking rows.

#### Partial Beneficiary Transitions
**Problem**: When one injured player returned but another remained out, the system would transition ALL BENEFICIARY records to EX_BENEFICIARY.

**Solution**: Beneficiary transition logic now only transitions the specific BENEFICIARY record for the returned player, leaving other active BENEFICIARY relationships intact.

**Impact**: Correct minute adjustments when multiple injuries resolve at different times.

## Timezone Handling

All date operations use **Eastern Time (US/Eastern)** via `pytz.timezone('US/Eastern')` to match NBA game schedule timing. RETURN_DATE_DT is a date object (not datetime) for clean comparison.

## Output

### Minutes Projections
Models 1-3 save to: `s3://nba-prediction-ibracken/model_comparison/{model_name}/minutes_projections.parquet`

**Columns**: DATE, PLAYER, TEAM, POSITION, PROJECTED_MIN, ACTUAL_MIN, PROJECTED_FP, ACTUAL_FP, CONFIDENCE

**Model Names**:
- `complex_position_overlap`
- `direct_position_only`
- `formula_c_baseline`

### Daily Lineups
For minutes models, each FP variant saves its own lineup to:

`s3://nba-prediction-ibracken/model_comparison/{model_name}/fp_{fp_model}/daily_lineups.parquet`

Where:
- `{model_name}` is one of: `complex_position_overlap`, `direct_position_only`, `formula_c_baseline`
- `{fp_model}` is one of: `current`, `fp_per_min`, `barebones`

DailyFantasyFuel remains:

`s3://nba-prediction-ibracken/model_comparison/daily_fantasy_fuel_baseline/daily_lineups.parquet`

**Columns**: DATE, SLOT, PLAYER, TEAM, POSITION, SALARY, PROJECTED_MIN, PROJECTED_FP, ACTUAL_FP

**Additional Model**:
- `daily_fantasy_fuel_baseline` (lineup-only, no minutes projections file)

### Injury Context
Saved to `s3://nba-prediction-ibracken/injury_context/{model_name}.parquet` (complex and direct models only)

**Columns**: PLAYER, TEAM, STATUS, INJURY_DATE, RETURN_DATE, TRUE_BASELINE, BENEFICIARY_OF, UPDATED_DATE
