# Minutes Projection Strategies

## Current Baseline: SportsLine
- MAE: 4.74 minutes
- Used as our comparison benchmark
- Problem: 20% of predictions have ±8 min error

---

## Our Approach

### 1. Formula C (Core Projection)
```
projected_min = (0.5 × season_avg_min) + (0.3 × last_7_avg_min) + (0.2 × prev_game_min)
```

**Why this formula:**
- Balanced approach between season stability and recent trends
- Not too reactive, not too conservative
- Cap at 36 minutes to prevent unrealistic projections

---

### 2. Injury Adjustments (WHERE THE MONEY IS)

**Core Principle:** When a key player goes down, correctly identifying who absorbs the minutes = DFS edge.

---

#### Strategy A: Simple Position-Based Redistribution (Baseline)

Distribute lost minutes evenly among teammates at the same position.

```python
lost_minutes = injured_player.season_avg_min
teammates_at_position = filter_by_position(team, injured_player.position)

for teammate in teammates_at_position:
    boost = lost_minutes / len(teammates_at_position)
    teammate.projected_min += boost
```

**Pros:** Simple, interpretable
**Cons:** Ignores player similarity, role, and skill level. A backup point guard and third-string center aren't equally likely to absorb PG minutes.

---

#### Strategy B: Cluster-Based Redistribution ⭐ (RECOMMENDED)

Use player clusters to identify who is actually similar to the injured player.

```python
lost_minutes = injured_player.season_avg_min
injured_cluster = injured_player.cluster

# Find teammates in same cluster AND similar position
similar_players = [
    p for p in team
    if p.cluster == injured_cluster
    and position_overlap(p.position, injured_player.position)
]

# If no cluster matches, fall back to position only
if not similar_players:
    similar_players = filter_by_position(team, injured_player.position)

# Weight distribution by current minutes (better players get more)
total_current_min = sum(p.season_avg_min for p in similar_players)

for player in similar_players:
    weight = player.season_avg_min / total_current_min
    player.projected_min += lost_minutes * weight
# Set cap on minutes per player
```

**Why this works:**
- Clusters capture playstyle similarity (e.g., both are rim-runners, both are stretch-4s)
- Players in same cluster are more likely to fill similar roles
- Weighting by current minutes accounts for depth chart order

**Example:**
- Injured: Joel Embiid (35 min, Cluster: Elite Big)
- Cluster match: Paul Reed (15 min, Cluster: Elite Big)
- Position match but different cluster: Montrezl Harrell (12 min, Cluster: Energy Big)
- Reed gets MORE of the minutes (same cluster + more current minutes)

---


#### Strategy C: Tiered Distribution (Stars vs Role Players)

Different redistribution logic based on injured player's role.

```python
lost_minutes = injured_player.season_avg_min

if lost_minutes >= 32:  # Star player injury
    # Star injuries → big minutes shift to multiple players
    primary_backup = find_primary_backup(injured_player)  # Same cluster + position
    secondary_backups = find_secondary_backups(injured_player)

    primary_backup.projected_min += lost_minutes * 0.5  # Gets 50%

    for backup in secondary_backups:
        backup.projected_min += (lost_minutes * 0.5) / len(secondary_backups)

elif lost_minutes >= 20:  # Rotation player injury
    # Minutes mostly go to direct backup
    backup = find_primary_backup(injured_player)
    backup.projected_min += lost_minutes * 0.7

    # Rest distributed to cluster matches
    cluster_matches = find_cluster_matches(injured_player)
    for player in cluster_matches:
        player.projected_min += (lost_minutes * 0.3) / len(cluster_matches)

else:  # Deep bench injury (<20 min)
    # Minimal impact - might not redistribute at all
    pass
```

**Why this works:**
- Star injuries create bigger rotational shifts
- Deep bench injuries barely matter
- Matches real coaching behavior

---



### Multiple Injuries (Critical Edge Case)

When 2+ players are OUT, naive redistribution breaks:

```python
def handle_multiple_injuries(team, injuries):
    """
    Handle multiple concurrent injuries

    Key principle: Process in order of importance, with diminishing returns
    """
    # Sort by minutes (star injuries first)
    injuries_sorted = sorted(injuries, key=lambda x: x.season_avg_min, reverse=True)

    adjusted_minutes = {p: p.baseline_min for p in team}

    for injured in injuries_sorted:
        lost = injured.season_avg_min

        # Find beneficiaries (cluster + position)
        beneficiaries = find_cluster_and_position_matches(injured, team)

        for player in beneficiaries:
            current = adjusted_minutes[player]

            # Cap at 38 minutes (realistic maximum)
            if current < 38:
                # Weight by cluster similarity and current minutes
                weight = calculate_weight(player, injured)
                boost = min(lost * weight, 38 - current)
                adjusted_minutes[player] += boost

    return adjusted_minutes
```

**Key insights:**
- Process star injuries first
- Cap individual players at realistic maximums (36-38 min)
- Each subsequent injury has less impact (diminishing returns)

---

### Implementation Recommendation

1. Do them all and analyze results
---

## Known Issues

### Return-from-Injury Problem
**Issue:** When an injured player returns, our formula struggles to adjust.

**Why:** The players who filled in during the injury will have inflated recent minutes:
- Their `last_7_avg_min` is HIGH (from filling in)
- Their `prev_game_min` is HIGH (from filling in)
- Formula C gives 50% weight to recent data → projects them too high

**Example:**
- Star player out 2 weeks
- Backup plays 30 min/game during absence
- Backup's `last_7_avg` jumps from 15 → 28 minutes
- Star returns → Backup should drop back to ~15 min
- But Formula C projects: (15 × 0.5) + (28 × 0.3) + (30 × 0.2) = 21.9 min ❌
- Actual should be closer to 15 min ✓
- Even more nuance with multiple injuries at once(which is common)

**Solution: Conservative Return-from-Injury + DFS Risk Management**

**Core Philosophy:** In DFS, wrongly projecting a bench player at 30 min (include in lineup) when they actually play 13 min = catastrophic. Missing an opportunity (projecting 13 min when they play 20) = acceptable. **Be conservative.**

**1. Injury Tracking System**
- Create S3 table: `data/injury_tracking/current.parquet`
- Track: `PLAYER`, `DATE`, `STATUS` (OUT/ACTIVE), `DAYS_OUT` (consecutive)
- Update daily when scraping NBA injury reports

**2. Return Detection**
- Compare today's injury report vs yesterday's
- If player was OUT 3+ days and now ACTIVE → Return detected
- Find injury start date (last game before OUT status)

**3. Returning Star Player Adjustment**
```python
if days_out <= 10:  # Short injury
    projected_min = pre_injury_baseline
    confidence = 'MEDIUM'  # Might have rust
    variance = ±6 min
else:  # Long injury
    projected_min = pre_injury_baseline × 0.7  # Minutes restriction likely
    confidence = 'LOW'  # High uncertainty
    variance = ±10 min
```

**4. Fill-In Player Adjustment (THE CRITICAL PART)**
```python
# When star returns, check each teammate
injury_start_date = get_injury_start_date(returning_player)

for teammate in team:
    # Get their projection BEFORE the injury started
    pre_injury_baseline = get_projection_before_injury_date(
        teammate,
        injury_start_date
    )

    # Check if their role expanded during injury
    # Use max of season_avg or recent_avg (handles both short and long injuries)
    current_role = max(teammate.season_avg_min, teammate.recent_avg_min)

    if current_role - pre_injury_baseline >= 5:
        # Player filled in during injury - REVERT
        projected_min = pre_injury_baseline
        confidence = 'LOW'  # Flag as uncertain
        variance = ±8 min
        note = 'REVERT_FROM_FILL_IN_ROLE'

# Why this works for ALL injury lengths:
# - Short injury (10 days): recent_avg inflated, season_avg not yet → Caught
# - Long injury (50 days): season_avg inflated, recent_avg stable → Caught
# Example: Tatum out 50 games, Hauser goes from 15→25 min season avg → Caught (25-15=10)
```

**5. Lineup Optimizer Integration**
```python
# Avoid HIGH UNCERTAINTY players in DFS lineups
safe_players = [p for p in players if p['confidence'] != 'LOW']
risky_players = [p for p in players if p['confidence'] == 'LOW']

# Build lineup from safe players first
# Only use risky players if exceptional value AND no better options
```

**6. Self-Correction After 2-3 Games**
- Return day projections might be conservative (maybe bench player keeps 20 min)
- Formula C incorporates actual data from Games 1-2
- Confidence returns to HIGH after rotation stabilizes
- Accept missing short-term opportunity to avoid catastrophic failure

**Key Insight:** Better to **under-project** fill-in players and avoid them in lineups than risk including a dud. DFS downside (wasted roster spot) >> upside (catching occasional role change).