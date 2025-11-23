# Injury Context Tracking System

**Purpose:** Prevent compounding inflation when redistributing minutes for long-term injuries.

---

## The Problem

**Without tracking:**
- Week 1: Reed (15 min baseline) + Embiid injury boost (+12) = 27 min ✓
- Week 3: Reed (22 min season_avg) + Embiid injury boost (+12) = 34 min ❌
- Week 6: Reed (25 min season_avg) + Embiid injury boost (+12) = 37 min ❌❌

Season average climbs BECAUSE of injury, but we keep adding boost on top = **massive inflation**.

**With tracking:**
- Week 1-6: Reed (15 min TRUE baseline) + Embiid boost (+12) = 27 min ✓✓✓

---

## Core Data Structure

### Injury Context Table
**S3 Path:** `data/injury_context/current.parquet`

```python
{
    'PLAYER': str,              # Player name
    'TEAM': str,                # Team abbreviation
    'STATUS': str,              # 'INJURED' or 'BENEFICIARY'
    'INJURY_DATE': date,        # When injury started
    'TRUE_BASELINE': float,     # Pre-injury minutes (frozen)
    'BENEFICIARY_OF': str,      # Which injured player they're filling in for
    'UPDATED_DATE': date        # Last update
}
```

### Example Data:
```
PLAYER       | TEAM | STATUS      | INJURY_DATE | TRUE_BASELINE | BENEFICIARY_OF
-------------|------|-------------|-------------|---------------|---------------
Joel Embiid  | PHI  | INJURED     | 2025-01-15  | 35.0          | NULL
Paul Reed    | PHI  | BENEFICIARY | 2025-01-15  | 15.0          | Joel Embiid
Tyrese Maxey | PHI  | BENEFICIARY | 2025-01-15  | 36.0          | Joel Embiid
```

---

## Implementation Algorithm

### 1. Get Player Baseline (with Fallbacks)

```python
def get_player_baseline(player, injury_context):
    """Get player's TRUE baseline minutes"""

    # Check if player is currently benefiting from injury
    beneficiary_record = injury_context[
        (injury_context['PLAYER'] == player.name) &
        (injury_context['STATUS'] == 'BENEFICIARY')
    ]

    if not beneficiary_record.empty:
        # Return frozen baseline from before they started filling in
        return beneficiary_record['TRUE_BASELINE'].iloc[0]

    # Not benefiting - get current baseline with fallbacks
    if player.games_played >= 5:
        return player.season_avg_min
    elif player.prev_season_avg_min is not None:
        return player.prev_season_avg_min  # Handles Tatum case
    elif player.career_avg_min is not None:
        return player.career_avg_min
    else:
        return 10  # Conservative fallback
```

### 2. Apply Injury Redistribution

```python
def redistribute_injury_minutes(injured_player, team, injury_context):
    """Redistribute minutes using TRUE baselines"""

    # Constants for distribution weighting
    BENCH_OPPORTUNITY_CONSTANT = 3  # Small boost for players who rarely play
    EXACT_POSITION_MULTIPLIER = 2.0  # Direct backups get 2x weight (prioritize exact position match)

    # Get injured player's baseline (with fallback for 0 games)
    if injured_player.games_played >= 5:
        lost_minutes = injured_player.season_avg_min
    elif injured_player.prev_season_avg_min is not None and injured_player.prev_season_avg_min > 0:
        lost_minutes = injured_player.prev_season_avg_min
    elif injured_player.career_avg_min is not None and injured_player.career_avg_min > 0:
        lost_minutes = injured_player.career_avg_min
    else:
        lost_minutes = 5  # Very low - player hasn't played in forever

    # Find teammates at same position
    teammates = [p for p in team
                 if position_overlap(p.position, injured_player.position)
                 and p.status != 'OUT']

    # Validation: If no teammates available, return empty (can't redistribute)
    if not teammates:
        logger.warning(f"No teammates available to absorb {injured_player.name}'s minutes")
        return {}

    # Calculate weights using TRUE baselines + constant (not inflated values)
    baselines = {p: get_player_baseline(p, injury_context) for p in teammates}

    # Apply position priority weighting (exact position gets 2x, adjacent positions get 1x)
    weighted_baselines = {}
    for player in teammates:
        base_weight = baselines[player] + BENCH_OPPORTUNITY_CONSTANT

        # Apply 2x multiplier for exact position match (prioritizes direct backups)
        if player.position == injured_player.position:
            weighted_baselines[player] = base_weight * EXACT_POSITION_MULTIPLIER
        else:
            weighted_baselines[player] = base_weight

    total_weighted = sum(weighted_baselines.values())

    # Initial distribution
    projections = {}
    for player in teammates:
        weight = weighted_baselines[player] / total_weighted
        boost = lost_minutes * weight
        projected = baselines[player] + boost
        projections[player] = projected

    # Handle overflow from 36-min cap (redistribute to non-capped players)
    MAX_MINUTES = 36
    max_iterations = 5  # Prevent infinite loop

    for iteration in range(max_iterations):
        overflow_minutes = 0
        capped_players = set()

        # Collect overflow from capped players
        for player, projected in projections.items():
            if projected > MAX_MINUTES:
                overflow_minutes += (projected - MAX_MINUTES)
                projections[player] = MAX_MINUTES
                capped_players.add(player)

        # No overflow - we're done
        if overflow_minutes == 0:
            break

        # Redistribute overflow to non-capped players
        non_capped = [p for p in teammates if p not in capped_players]

        if not non_capped:
            # Everyone capped - burn remaining minutes (conservative)
            break

        # Weight by original weighted baselines
        non_capped_total = sum(weighted_baselines[p] for p in non_capped)

        for player in non_capped:
            weight = weighted_baselines[player] / non_capped_total
            additional_boost = overflow_minutes * weight
            projections[player] += additional_boost

    # Track all beneficiaries in injury context
    final_projections = {}
    for player in teammates:
        final_projections[player.name] = projections[player]

        add_to_injury_context(
            player=player.name,
            team=player.team,
            status='BENEFICIARY',
            true_baseline=baselines[player],  # Store original baseline
            beneficiary_of=injured_player.name,
            injury_date=today
        )

    return final_projections
```

**Example Impact (with overflow redistribution):**
```
Embiid (35 min) injured, position overlap: Reed (15 min), Bamba (5 min), Young (0 min)

Initial distribution (with +3 constant):
- Reed: 18/(18+8+3) = 62% → 15 + 21.7 = 36.7 min
- Bamba: 8/(18+8+3) = 28% → 5 + 9.8 = 14.8 min
- Young: 3/(18+8+3) = 10% → 0 + 3.5 = 3.5 min

After 36-min cap & overflow redistribution:
- Reed: 36.7 → capped at 36 min (0.7 min overflow)
- Bamba: 14.8 + (0.7 × 8/11) = 15.3 min ✓ (gets Reed's overflow)
- Young: 3.5 + (0.7 × 3/11) = 3.7 min ✓ (gets Reed's overflow)

Total: 36 + 15.3 + 3.7 = 55 min ≈ 35 min redistributed ✓
```

**Cascading overflow example:**
```
LeBron (38 min) injured, deep bench available

Initial:
- Davis: 32 + 20 = 52 min → cap at 36 (16 overflow)
- Rui: 18 + 10 = 28 min
- Hayes: 8 + 5 = 13 min
- Christie: 0 + 3 = 3 min

After overflow round 1:
- Davis: 36 min (capped)
- Rui: 28 + (16 × 21/34) = 37.9 min → cap at 36 (1.9 overflow)
- Hayes: 13 + (16 × 11/34) = 18.2 min
- Christie: 3 + (16 × 3/34) = 4.4 min

After overflow round 2:
- Davis: 36 min (capped)
- Rui: 36 min (capped)
- Hayes: 18.2 + (1.9 × 11/14) = 19.7 min ✓
- Christie: 4.4 + (1.9 × 3/14) = 4.8 min ✓

Final: 36 + 36 + 19.7 + 4.8 = 96.5 min (conservative, some burned)
```

### 3. Handle Return from Injury

```python
def handle_return_from_injury(returning_player, team, injury_context):
    """When player returns, revert beneficiaries"""

    # Get returning player's info
    injury_record = injury_context[
        (injury_context['PLAYER'] == returning_player.name) &
        (injury_context['STATUS'] == 'INJURED')
    ]

    if injury_record.empty:
        return  # Not tracked as injured

    days_out = (today - injury_record['INJURY_DATE'].iloc[0]).days
    true_baseline = injury_record['TRUE_BASELINE'].iloc[0]

    # Project returning player
    if days_out <= 10:
        returning_player.projected_min = true_baseline
        returning_player.confidence = 'MEDIUM'
    else:
        returning_player.projected_min = true_baseline * 0.7
        returning_player.confidence = 'LOW'

    # Revert all beneficiaries
    beneficiaries = injury_context[
        (injury_context['BENEFICIARY_OF'] == returning_player.name)
    ]

    for _, beneficiary in beneficiaries.iterrows():
        # Find player in team
        player = next(p for p in team if p.name == beneficiary['PLAYER'])

        # Revert to pre-injury baseline
        player.projected_min = beneficiary['TRUE_BASELINE']
        player.confidence = 'LOW'
        player.note = 'REVERT_FROM_FILL_IN_ROLE'

    # Clear from injury context
    remove_from_injury_context(returning_player.name)
```

### 4. Daily Update Process

```python
def update_injury_context(todays_injuries, injury_context):
    """Update injury context daily"""

    today = datetime.date.today()

    # Mark new injuries
    for injured_player in todays_injuries:
        if injured_player.name not in injury_context['PLAYER'].values:
            add_to_injury_context(
                player=injured_player.name,
                team=injured_player.team,
                status='INJURED',
                injury_date=today,
                true_baseline=get_player_baseline(injured_player, injury_context)
            )

    # Detect returns (players no longer in injury report)
    active_injuries = injury_context[injury_context['STATUS'] == 'INJURED']
    for _, injury in active_injuries.iterrows():
        if injury['PLAYER'] not in [p.name for p in todays_injuries]:
            # Player returned
            remove_from_injury_context(injury['PLAYER'])

    # Save updated context
    save_to_s3(injury_context, 'data/injury_context/current.parquet')
```

---

## Edge Cases Handled

### ✅ **Long-term injuries (30+ games)**
- Uses frozen TRUE baseline, prevents compounding inflation

### ✅ **Player hasn't played this season (Tatum case)**
- Fallback: Previous season avg → Career avg → 5 min default (very conservative for inactive players)

### ✅ **Cascading injuries (beneficiary gets injured)**
```python
# Reed filling in for Embiid (30 min), then Reed gets injured
# Redistributes Reed's TRUE baseline (15 min), not inflated (30 min)
lost_minutes = get_player_baseline(Reed, injury_context)  # Returns 15, not 30
```

### ✅ **Multiple simultaneous injuries**
- Process star injuries first (sort by minutes)
- Cap each player at 36-38 min
- Burn leftover minutes (conservative)

### ✅ **Early season (< 5 games)**
- Use previous season or career averages

### ✅ **Traded players**
- Fallback to previous team stats or career average

### ✅ **Confidence status expiration**
- LOW confidence clears after 3 games played
- Allows Formula C to take over with actual data

---

## Confidence Management

### When to Set LOW Confidence:
1. **Returning player:** Set when player first returns from injury
2. **Fill-in reversion:** Set when star returns and bench player reverted

### When to Clear LOW Confidence:
```python
def update_confidence_status(player, injury_context, games_log):
    """
    Clear LOW confidence after player has played enough games

    Args:
        player: Player object
        injury_context: Injury context DataFrame
        games_log: Player's recent games

    Returns:
        Updated confidence level
    """
    CONFIDENCE_CLEARANCE_GAMES = 3  # Games needed to clear LOW confidence

    # Check if player has LOW confidence in injury context
    low_confidence_record = injury_context[
        (injury_context['PLAYER'] == player.name) &
        (injury_context['STATUS'].isin(['INJURED', 'BENEFICIARY']))
    ]

    if low_confidence_record.empty:
        return 'HIGH'  # Not in injury context

    # Get injury/revert date
    injury_date = low_confidence_record['INJURY_DATE'].iloc[0]

    # Count games played since injury date
    games_since = games_log[games_log['GAME_DATE'] > injury_date]
    games_played_since = len(games_since)

    if games_played_since >= CONFIDENCE_CLEARANCE_GAMES:
        # Enough games played - clear from injury context
        remove_from_injury_context(player.name)
        return 'HIGH'

    # Still in stabilization period
    if games_played_since == 0:
        return 'LOW'
    elif games_played_since < CONFIDENCE_CLEARANCE_GAMES:
        return 'MEDIUM'

    return 'HIGH'
```

**Example Timeline:**
```
Day 1 (Embiid returns):
- Embiid: LOW confidence (projected 24 min, plays 28 min actual)
- Reed: LOW confidence (projected 15 min, plays 18 min actual)

Day 2 (Game 1 after return):
- Embiid: MEDIUM confidence (Formula C uses game 1 data)
- Reed: MEDIUM confidence (Formula C uses game 1 data)

Day 4 (Game 3 after return):
- Embiid: HIGH confidence (3 games played, clear from injury context)
- Reed: HIGH confidence (3 games played, clear from injury context)
- Both now use normal Formula C projections
```

---

## Integration with Formula C

```python
def project_minutes(player, team, injury_context, injury_data):
    """Complete projection with injury handling"""

    # Step 1: Check if player is injured
    if player.status == 'OUT':
        return 0

    # Step 2: Check if player just returned
    if player.name in injury_context[injury_context['STATUS'] == 'INJURED']['PLAYER'].values:
        return handle_return_from_injury(player, team, injury_context)

    # Step 3: Update confidence status (clears LOW after 3 games)
    player.confidence = update_confidence_status(player, injury_context, player.games_log)

    # Step 4: Get baseline (might be frozen if benefiting from injury)
    baseline = get_player_baseline(player, injury_context)

    # Step 5: Check for teammate injuries (handle multiple injuries)
    team_injuries = [p for p in team if p.status == 'OUT']

    if team_injuries:
        # Sort by minutes (handle star injuries first)
        team_injuries_sorted = sorted(team_injuries, key=lambda x: x.season_avg_min or 0, reverse=True)

        # Apply redistribution for each injury sequentially
        current_projection = baseline
        for injured in team_injuries_sorted:
            injury_projections = redistribute_injury_minutes(injured, team, injury_context)
            if player.name in injury_projections:
                current_projection = injury_projections[player.name]
                break  # Found this player's projection

        return current_projection

    # Step 5: No injuries - use Formula C
    projected = (
        0.5 * baseline +
        0.3 * player.last_7_avg_min +
        0.2 * player.prev_game_min
    )

    return min(projected, 36)
```

---

## Testing Validation

**Critical scenarios to validate:**
- Long injury (40+ games): Baseline stays frozen, no inflation
- Player returns: Beneficiaries revert to pre-injury baseline
- Cascading injury: Redistribution uses frozen baseline (not inflated)
- Zero games played: Fallback chain works (prev_season → career → 10 min)
- Early season: Previous season data used correctly

---

## Key Principle

**Always use TRUE baselines for redistribution, never inflated current values.**

This prevents compounding inflation while maintaining accurate projections for DFS optimization.

---

## Helper Functions

### position_overlap()
```python
def position_overlap(pos1, pos2):
    """
    Check if two positions can substitute for each other

    Args:
        pos1, pos2: Position strings (e.g., 'PG', 'SG', 'SF', 'PF', 'C')

    Returns:
        bool: True if positions overlap
    """
    # Define position groups that can substitute
    position_groups = [
        {'PG', 'SG'},      # Guards can swap
        {'SG', 'SF'},      # Wings can swap
        {'SF', 'PF'},      # Forwards can swap
        {'PF', 'C'},       # Bigs can swap
    ]

    # Exact match
    if pos1 == pos2:
        return True

    # Check if in same group
    for group in position_groups:
        if pos1 in group and pos2 in group:
            return True

    return False
```

### add_to_injury_context()
```python
def add_to_injury_context(player, team, status, true_baseline, beneficiary_of, injury_date):
    """Add or update player in injury context"""
    global injury_context

    # Remove existing record if present
    injury_context = injury_context[injury_context['PLAYER'] != player]

    # Add new record
    new_record = pd.DataFrame([{
        'PLAYER': player,
        'TEAM': team,
        'STATUS': status,
        'INJURY_DATE': injury_date,
        'TRUE_BASELINE': true_baseline,
        'BENEFICIARY_OF': beneficiary_of,
        'UPDATED_DATE': datetime.date.today()
    }])

    injury_context = pd.concat([injury_context, new_record], ignore_index=True)
```

### remove_from_injury_context()
```python
def remove_from_injury_context(player_name):
    """Remove player and their beneficiaries from injury context"""
    global injury_context

    # Remove the injured player
    injury_context = injury_context[injury_context['PLAYER'] != player_name]

    # Remove all beneficiaries of this player
    injury_context = injury_context[injury_context['BENEFICIARY_OF'] != player_name]
```

---

## Multi-Model Comparison Framework

Run 4 parallel projection models daily to compare performance and identify the best approach.

### Models

1. **Complex Position Overlap (Primary Model)**
   - Uses position_overlap() with adjacent position substitution
   - EXACT_POSITION_MULTIPLIER = 2.0 for direct backups
   - TRUE baseline tracking to prevent inflation
   - Injury redistribution with Formula C

2. **Direct Position Exchange**
   - Only exact position matches (pos1 == pos2)
   - No adjacent position overlap
   - TRUE baseline tracking
   - Simpler redistribution, more conservative

3. **Formula C Baseline (No Injury Handling)**
   - Pure Formula C: 0.5 × season_avg + 0.3 × last_7 + 0.2 × prev_game
   - No injury redistribution at all
   - Benchmark for measuring injury adjustment value

4. **SportsLine Baseline**
   - Direct pull from SportsLine projections
   - Industry benchmark (4.74 MAE historically)
   - No custom logic

### Daily Output Structure

**Process:**
1. Generate minutes_projections.parquet using model-specific logic
2. Feed projections into **lineup-optimizer** (`lambda/lineup-optimizer/`) to generate optimal_lineup.parquet
3. Calculate projected_fantasy_points.parquet from lineup

**S3 Storage:**
```
data/model_comparison/{date}/
├── complex_position_overlap/
│   ├── minutes_projections.parquet       # Model output
│   ├── optimal_lineup.parquet            # From lineup-optimizer
│   └── projected_fantasy_points.parquet  # Calculated from lineup
├── direct_position_only/
│   ├── minutes_projections.parquet
│   ├── optimal_lineup.parquet
│   └── projected_fantasy_points.parquet
├── formula_c_baseline/
│   ├── minutes_projections.parquet
│   ├── optimal_lineup.parquet
│   └── projected_fantasy_points.parquet
└── sportsline_baseline/
    ├── minutes_projections.parquet
    ├── optimal_lineup.parquet
    └── projected_fantasy_points.parquet
```

**Note:** All models use the same lineup-optimizer downstream. This isolates projection quality from lineup construction.

### Comparison Metrics

After actual games complete, calculate for each model:
- **Minutes MAE**: Mean absolute error on minutes projections
- **Fantasy Points MAE**: Accuracy of total fantasy points projection
- **Lineup Success Rate**: % of days the optimal lineup would have been profitable
- **Bust Rate**: % of players projected 25+ min who played <15 min (catastrophic failures)

### Purpose

Isolate the value of injury redistribution logic by comparing against baselines. If Complex Position Overlap consistently outperforms Formula C Baseline, validates the injury handling approach. SportsLine provides industry benchmark.
