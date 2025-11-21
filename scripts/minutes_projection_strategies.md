# Minutes Projection Strategies - Testing Framework

## The Problem

**Current State:**
- SportsLine projections: 4.74 min MAE
- 20% of predictions have ±8 min error
- Examples of failures:
  - Sion James: 42 min projected → 23-25 actual (consistent over-projection)
  - Paul Reed: 13 min projected → 31 actual (missed opportunity)
  - Andre Drummond: 13 min projected → 35 actual (injury fill-in)

**Impact on FP Predictions:**
- MIN feature = 60% importance in model
- 4.74 min MAE × 60% = **2.85 FP error from minutes alone**
- Current model MAE: 8.02 FP
- If we reduce minutes MAE to 3.5 → save ~0.75 FP → **7.27 FP total** (beats DFF's 7.38!)

---

## Strategy Categories

### 1. Baseline Formulas (No Injury Data)

Simple rule-based projections using only historical player stats.

#### Formula A: Simple Season Average
```
projected_min = season_avg_min
```
**Pros**: Stable, simple
**Cons**: Ignores trends, slow to adapt

#### Formula B: Weighted Average (Conservative)
```
projected_min = 0.6 * season_avg_min + 0.3 * last_7_avg_min + 0.1 * prev_game_min
```
**Pros**: Stable but considers recent form
**Cons**: May miss recent role changes

#### Formula C: Weighted Average (Balanced)
```
projected_min = 0.5 * season_avg_min + 0.3 * last_7_avg_min + 0.2 * prev_game_min
```
**Pros**: Balanced approach
**Cons**: Still reactive

#### Formula D: Weighted Average (Reactive)
```
projected_min = 0.3 * season_avg_min + 0.4 * last_7_avg_min + 0.3 * prev_game_min
```
**Pros**: Captures trends quickly
**Cons**: More volatile, sensitive to outliers

#### Formula E: Last 7 Games Only
```
projected_min = last_7_avg_min
```
**Pros**: Most reactive
**Cons**: Volatile, misses long-term baseline

#### Formula F: Starter-Adjusted
```python
if is_starter:
    projected_min = max(28, weighted_average)
else:
    projected_min = weighted_average
```
**Pros**: Prevents under-projecting starters
**Cons**: Need accurate starter detection

**Test These**: Run all formulas on historical data (Oct 30+), calculate MAE for each.

---

### 2. Injury Adjustment Strategies

Requires: NBA injury report (Status: OUT)

#### Strategy 1: Simple Redistribution
```python
for injured_player in injuries:
    lost_minutes = injured_player.season_avg_min

    # Find teammates in same position
    teammates = filter_by_position(team, injured_player.position)

    # Distribute evenly
    for teammate in teammates:
        teammate.projected_min += lost_minutes / len(teammates)
```

**Pros**: Simple, interpretable
**Cons**: Ignores player quality, cluster similarity

---

#### Strategy 2: Cluster-Based Redistribution
```python
for injured_player in injuries:
    lost_minutes = injured_player.season_avg_min

    # Find teammates in same cluster/position
    similar_players = filter_by_cluster_and_position(
        team,
        injured_player.cluster,
        injured_player.position
    )

    # Distribute by inverse depth chart order
    for player in similar_players:
        # Backup gets most, then third-string, etc.
        weight = calculate_depth_weight(player)
        player.projected_min += lost_minutes * weight
```

**Pros**: More intelligent distribution
**Cons**: Cluster data might not capture role perfectly

---

#### Strategy 3: Historical Pattern Learning
```python
# For each historical injury:
# - Identify injured player and their typical minutes
# - Observe how minutes actually redistributed
# - Learn patterns by position/cluster

# For new injury:
# - Match to similar historical injury (same position, similar minutes)
# - Apply learned redistribution pattern
```

**Pros**: Data-driven, learns real patterns
**Cons**: Requires building historical injury database

---

#### Strategy 4: Tiered Redistribution (Your Idea)
```python
for injured_player in injuries:
    lost_minutes = injured_player.season_avg_min

    # Check injury recency
    if injury_days < 3:  # Recent injury
        # Look for teammates with boosted proj_min
        boosted = find_boosted_teammates(team)
        if not boosted:
            # Manual distribution
            star_boost = 4
            backup1_boost = 16
            backup2_boost = 16
            # Distribute to appropriate players
    else:  # Long-term injury
        # Analyze teammates' minutes shift since injury
        # Compare historical avg vs recent avg vs projected
        # Adjust projections based on observed shift
```

**Pros**: Nuanced, handles different injury scenarios
**Cons**: Complex, many edge cases

---

#### Strategy 5: Multiple Injuries Handler
```python
def handle_multiple_injuries(team, injuries):
    """
    When multiple players are out:
    1. Sort injuries by typical minutes (stars first)
    2. For each injury, redistribute considering:
       - Already-adjusted minutes from previous injuries
       - Avoid over-allocating to single player
       - Cap at 36-38 minutes per player
    """

    injuries_sorted = sorted(injuries, key=lambda x: x.season_avg_min, reverse=True)

    adjusted_minutes = {}
    for injured in injuries_sorted:
        available_teammates = [p for p in team if p not in injuries]

        # Distribute lost minutes
        lost = injured.season_avg_min
        for teammate in available_teammates:
            current = adjusted_minutes.get(teammate, teammate.baseline_min)
            if current < 36:  # Don't over-allocate
                boost = min(lost / len(available_teammates), 36 - current)
                adjusted_minutes[teammate] = current + boost
```

**Pros**: Handles complex scenarios
**Cons**: Very complex logic

---

## Testing Framework

### Data Requirements:
- **Historical box scores** (have this)
- **Daily predictions** with ACTUAL_MIN (have this, after Oct 30)
- **NBA injury reports** (need to scrape - available on nba.com)
- **Starting lineup data** (already scraping this)

### Test Script Structure:
```python
def test_strategy(strategy_name, projection_function):
    """
    Test a minutes projection strategy against historical data

    Args:
        strategy_name: Name of the strategy
        projection_function: Function that takes (player_stats, injuries) → projected_min

    Returns:
        MAE, detailed error breakdown
    """
    results = []

    for game in historical_games_after_oct_30:
        for player in game.players:
            # Get player's historical stats BEFORE this game
            stats = get_stats_before_game(player, game.date)

            # Get injuries for that day
            injuries = get_injuries_on_date(player.team, game.date)

            # Calculate projected minutes using strategy
            projected = projection_function(stats, injuries)

            # Compare to actual
            actual = player.actual_min
            error = abs(projected - actual)

            results.append({
                'player': player.name,
                'date': game.date,
                'projected': projected,
                'actual': actual,
                'error': error
            })

    # Calculate MAE
    mae = sum(r['error'] for r in results) / len(results)

    return mae, results
```

### Strategies to Test:

1. **SportsLine (Baseline)** - Current approach (4.74 MAE)
2. **Formula B** - 0.6*season + 0.3*last7 + 0.1*prev
3. **Formula C** - 0.5*season + 0.3*last7 + 0.2*prev
4. **Formula C + Cap 36** - Formula C with 36 min cap
5. **Formula C + Starter Baseline** - Formula C with 28 min baseline for starters
6. **Formula C + Simple Injury** - Formula C + Strategy 1 injury adjustment
7. **[Your custom strategy here]**

### Success Metrics:
- **Primary**: Overall MAE (target: <4.0 min)
- **Starters**: MAE for players with season_avg > 24 min
- **Bench**: MAE for players with season_avg < 24 min
- **Outliers**: % of predictions with >8 min error
- **DFS Edge**: Capture rate for +15 min opportunities (injury fill-ins)

---

## Your Ideas & Experiments

### Injury Adjustment Brainstorm:

**Key Insights:**
- Recent injuries (< 3 days): Look for projected minutes boosts in teammates
- Long-term injuries: Analyze historical minutes shifts
- Multiple injuries: Complex redistribution, avoid over-allocation

**Questions to Explore:**
1. How to determine if an injury is "recent" vs "established"?
   - Use injury report date vs game date?
   - Look at teammate minutes trends?

2. How to identify which teammates get the minutes?
   - Position matching?
   - Cluster similarity?
   - Depth chart order?

3. How much to boost?
   - Even split among backups?
   - Weighted by cluster/position similarity?
   - Based on historical patterns?

4. How to handle multiple injuries?
   - Sequential allocation?
   - Simultaneous redistribution with constraints?

**Your Notes:**
```
[Add your thoughts here as you experiment]

Example:
- Tested Formula C: 3.8 MAE (beat SportsLine!)
- Issue: Still missing injury opportunities
- Next: Try adding simple injury adjustment (+10 to backup)
```

---

## Injury Data Sources

### NBA Official Injury Report:
- **URL**: https://www.nba.com/injury-report
- **Format**: HTML table with Status (OUT, QUESTIONABLE, etc.)
- **Available**: Current day + next few days
- **Historical**: Not easily available (would need daily scraping)

### Alternative Sources:
- ESPN injury report
- Rotoworld
- FantasyPros

### Scraping Strategy:
```python
def scrape_nba_injuries(date=None):
    """
    Scrape NBA injury report for given date (default: today)

    Returns:
        List of {player, team, position, status, injury_note}
    """
    # Implementation here
    pass
```

---

## Implementation Roadmap

### Phase 1: Baseline Testing (Do First)
- [ ] Implement 5-6 baseline formulas
- [ ] Test on historical data (Oct 30+)
- [ ] Identify best baseline formula
- [ ] **Expected outcome**: Beat SportsLine's 4.74 MAE

### Phase 2: Injury Scraping (Do Second)
- [ ] Build NBA injury report scraper
- [ ] Test scraper reliability (does it work daily?)
- [ ] Store injury data structure

### Phase 3: Injury Adjustment (Do Third)
- [ ] Implement simple redistribution (Strategy 1)
- [ ] Test on historical injury cases
- [ ] Iterate on approach based on results
- [ ] **Expected outcome**: Capture +15 min opportunities

### Phase 4: Production Deployment
- [ ] Replace SportsLine with best formula
- [ ] Add injury adjustment layer
- [ ] Apply 36 min cap (already done)
- [ ] Monitor production MAE
- [ ] **Target**: <7.38 MAE total FP (beat DFF)

---

## Quick Wins vs Long-Term

### Quick Wins (This Week):
1. Test baseline formulas → find one that beats 4.74 MAE
2. Implement it in daily-predictions
3. Deploy and monitor

### Long-Term (Next Few Weeks):
1. Build injury scraper
2. Test injury adjustment strategies
3. Handle edge cases (multiple injuries, back-to-backs)
4. Iterate based on production data

---

## Notes & Iteration Log

**[Add your findings here as you test]**

Example:
```
2025-11-21:
- Tested Formula C (0.5/0.3/0.2): 3.9 MAE vs SportsLine's 4.74
- Starters: 3.2 MAE, Bench: 5.1 MAE
- Issue: Bench players have high variance
- Next: Try starter-adjusted version

2025-11-22:
- Added 28 min baseline for starters: 3.7 MAE overall
- Improved starter accuracy, slight cost on bench
- Decision: Deploy Formula C + starter baseline
```

---

## Final Thoughts

**Key Principle**: Start simple, add complexity only if needed.

**Testing is critical**: Don't assume a formula works - validate with historical data.

**Injury adjustments are optional**: Even without them, beating SportsLine's baseline will help significantly.

**Production monitoring**: After deploying any change, watch the daily MAE closely to confirm improvement.
