"""
Projection models and core projection logic
"""

import pandas as pd
import logging
from config import MAX_MINUTES
from injury_system import (
    get_player_baseline,
    update_confidence_status,
    add_to_injury_context,
    remove_from_injury_context
)

logger = logging.getLogger()


# ==================== Position Overlap Functions ====================

def position_overlap_complex(pos1, pos2):
    """
    Complex position overlap: Adjacent positions can substitute
    Used for: Complex Position Overlap model
    """
    position_groups = [
        {'PG', 'SG'},      # Guards can swap
        {'SG', 'SF'},      # Wings can swap
        {'SF', 'PF'},      # Forwards can swap
        {'PF', 'C'},       # Bigs can swap
    ]

    if pos1 == pos2:
        return True

    for group in position_groups:
        if pos1 in group and pos2 in group:
            return True

    return False


def position_overlap_exact(pos1, pos2):
    """
    Exact position match only
    Used for: Direct Position Exchange model
    """
    return pos1 == pos2


# ==================== Core Projection Logic ====================

def project_minutes_with_injuries(player_row, team_players, injury_context, games_log, today, box_scores,
                                   position_overlap_func, use_multiplier, injury_data, team_injury_redistributions=None):
    """
    Unified projection logic for both models with injury handling

    Args:
        team_injury_redistributions: Pre-calculated injury redistributions for the team (optional).
                                     If None, will calculate on-demand (legacy behavior).

    IMPORTANT: This function handles TWO DIFFERENT "returning from injury" scenarios:

    A. RETURNING TODAY (25% reduction):
       - Player still ON injury report (has RETURN_DATE_DT = today)
       - Gets normal projection (redistribution or Formula C) with 25% reduction
       - Applies when player listed as OUT/QUESTIONABLE but expected to play today

    B. RETURNING STATUS (10 MPG for 4 games):
       - Player with season-long injury (0 games, FROM_PREV_SEASON=True)
       - NO LONGER on injury report (STATUS != 'OUT')
       - Gets conservative 10 MPG for first 4 games
       - Applies to stars returning after missing entire season (e.g., Kawhi)

    These scenarios DON'T OVERLAP because:
    - Scenario A requires being ON injury report (RETURN_DATE_DT set)
    - Scenario B requires NOT being on injury report anymore (STATUS != 'OUT')

    Other scenarios:
    3. Active injuries (teammate currently OUT) - Redistribute their minutes
    4. Normal play (no injuries affecting this player) - Use Formula C

    Args:
        position_overlap_func: Either position_overlap_complex or position_overlap_exact
        use_multiplier: If True, apply 2x multiplier for exact position (complex model)
        injury_data: DataFrame of current injury report data (to filter out free agents)

    Returns:
        (projected_minutes, confidence, updated_injury_context)
    """
    player_name = player_row['PLAYER']

    # ========== CHECK: Is player RETURNING TODAY? ==========
    # Scenario A: Player still on injury report (RETURN_DATE_DT = today) but expected to play
    # Will apply 25% reduction later (after redistribution or Formula C calculation)
    # Example: LeBron listed as OUT yesterday, expected to return today
    returning_today = False
    if 'RETURN_DATE_DT' in player_row and pd.notna(player_row.get('RETURN_DATE_DT')):
        return_date = player_row['RETURN_DATE_DT']
        if isinstance(return_date, pd.Timestamp):
            return_date = return_date.date()
        if return_date == today:
            returning_today = True
            logger.info(f"{player_name}: Returning today - will apply 25% minutes reduction")

    # ========== STEP 1: Handle currently injured players (status OUT) ==========
    # If OUT but returning today, treat as playing (with 25% reduction applied later)
    if player_row.get('STATUS') == 'OUT' and not returning_today:
        return 0, 'HIGH', injury_context

    # ========== STEP 2: Check for RETURNING STATUS (Season-Long Returns) ==========
    # Scenario B: Star player (e.g., Kawhi) was out ALL SEASON (0 games), now healthy (STATUS != 'OUT')
    # Both returning star AND affected teammates get conservative 10 MPG for first 4 games
    # NOTE: This is DIFFERENT from "Returning TODAY" (Scenario A) - those players are still on injury report

    # First, check injury_context for ALREADY-TRACKED returning players on this team
    returning_players = injury_context[
        (injury_context['STATUS'] == 'RETURNING') &
        (injury_context['TEAM'] == player_row['TEAM'])
    ]

    # Track if current player is returning OR affected by a returning player
    is_returning_player = False      # True if current player is the one returning
    affected_by_return = False       # True if current player affected by teammate's return
    max_returning_player_games = None  # Games played by returning player (for 4-game threshold)

    # Loop through already-tracked returning players (from previous lambda runs)
    for _, returning_record in returning_players.iterrows():
        returning_player_name = returning_record['PLAYER']
        return_date = pd.to_datetime(returning_record['INJURY_DATE'])

        # Count games played by returning player since return
        returning_games = games_log[
            (games_log['PLAYER'] == returning_player_name) &
            (games_log['GAME_DATE'] >= return_date)
        ]
        games_since_return = len(returning_games)

        if games_since_return >= 4:
            # Cleanup: Remove from RETURNING tracking after 4 games (enough data for baseline)
            logger.info(f"{returning_player_name} has played {games_since_return} games since return - removing from RETURNING tracking")
            injury_context = remove_from_injury_context(injury_context, returning_player_name)
        else:
            # Still in conservative period (< 4 games) - check if current player is affected
            if player_name == returning_player_name:
                is_returning_player = True
                max_returning_player_games = games_since_return
                logger.info(f"{player_name}: Returning player with {games_since_return}/4 games")
            else:
                # Check position overlap with current player
                returning_player_row = team_players[team_players['PLAYER'] == returning_player_name]
                if not returning_player_row.empty:
                    returning_pos = returning_player_row['POSITION'].iloc[0]
                    if position_overlap_func(player_row.get('POSITION', 'SF'), returning_pos):
                        affected_by_return = True
                        # Track the max games of any returning player affecting this player
                        if max_returning_player_games is None or games_since_return > max_returning_player_games:
                            max_returning_player_games = games_since_return
                        logger.info(f"{player_name}: Affected by {returning_player_name} return ({games_since_return}/4 games)")

                        # Mark affected teammate as EX_BENEFICIARY to exclude inflated games from baseline
                        # Check if already tracked (avoid duplicate records)
                        already_ex_beneficiary = injury_context[
                            (injury_context['PLAYER'] == player_name) &
                            (injury_context['STATUS'] == 'EX_BENEFICIARY') &
                            (injury_context['BENEFICIARY_OF'] == returning_player_name)
                        ]

                        if already_ex_beneficiary.empty:
                            # Get injury date from returning player's data
                            returning_player_full = team_players[team_players['PLAYER'] == returning_player_name]
                            if not returning_player_full.empty:
                                injury_date = returning_player_full.iloc[0].get('ESTIMATED_INJURY_DATE')
                                if pd.isna(injury_date) or injury_date is None:
                                    # Estimate as start of season
                                    current_year = pd.Timestamp(today).year
                                    season_start = pd.Timestamp(f"{current_year}-10-15").date()
                                    if pd.Timestamp(today).month < 7:
                                        season_start = pd.Timestamp(f"{current_year - 1}-10-15").date()
                                    injury_date = season_start

                                # Mark as EX_BENEFICIARY to exclude inflated games
                                logger.info(f"{player_name}: Marking as EX_BENEFICIARY of {returning_player_name} (excluding games {injury_date} to {return_date})")
                                injury_context = add_to_injury_context(
                                    injury_context,
                                    player_name=player_name,
                                    team=player_row['TEAM'],
                                    status='EX_BENEFICIARY',
                                    true_baseline=0,
                                    beneficiary_of=returning_player_name,
                                    injury_date=injury_date,
                                    return_date=return_date
                                )
    # Get set of players on injury report (any status - OUT, returning, etc.)
    if not injury_data.empty:
        injured_or_returning_players = set(injury_data['PLAYER'].unique())
    else:
        injured_or_returning_players = set()
    # Detect NEW season-long returns (not yet tracked in injury_context)
    # Example: Kawhi Leonard - played 30 MPG last season, 0 games this season, now healthy
    for _, teammate in team_players.iterrows():
        if (teammate.get('GAMES_PLAYED', 0) == 0 and           # Haven't played this season
            teammate.get('FROM_PREV_SEASON', False) and        # Data is from last season
            teammate.get('STATUS') != 'OUT' and                # Not currently injured (now healthy!)
            teammate.get('CAREER_AVG_MIN', 0) >= 20 and        # Must be significant player (filter out bench warmers)
            teammate['PLAYER'] in injured_or_returning_players):  # ← Must be/was on injury report (filters out free agents)
            # This teammate was out all season but is now returning!
            returning_player_name = teammate['PLAYER']

            # Get injury date (when they first got injured)
            # Use ESTIMATED_INJURY_DATE if available, otherwise estimate as start of season
            injury_date = teammate.get('ESTIMATED_INJURY_DATE')
            if pd.isna(injury_date) or injury_date is None:
                # Estimate as start of current season (October 15th of appropriate year)
                current_year = pd.Timestamp(today).year
                season_start = pd.Timestamp(f"{current_year}-10-15").date()
                if pd.Timestamp(today).month < 7:  # Before July = last year's season
                    season_start = pd.Timestamp(f"{current_year - 1}-10-15").date()
                injury_date = season_start
                logger.info(f"{returning_player_name}: No ESTIMATED_INJURY_DATE, using season start {injury_date}")

            # Check if this return is already tracked
            already_tracked = injury_context[
                (injury_context['PLAYER'] == returning_player_name) &
                (injury_context['STATUS'] == 'RETURNING')
            ]

            if already_tracked.empty:
                # Track this return in injury context
                logger.info(f"NEW season-long return detected: {returning_player_name} (out since {injury_date}, 10 MPG for 4 games)")
                injury_context = add_to_injury_context(
                    injury_context,
                    player_name=returning_player_name,
                    team=player_row['TEAM'],
                    status='RETURNING',
                    true_baseline=0,
                    beneficiary_of=None,
                    injury_date=today
                )

                # Mark if current player is affected
                if player_name == returning_player_name:
                    is_returning_player = True
                    max_returning_player_games = 0  # Just detected, 0 games played
                elif position_overlap_func(player_row.get('POSITION', 'SF'), teammate.get('POSITION', 'SF')):
                    # Mark affected teammate as EX_BENEFICIARY to exclude inflated games from baseline
                    affected_by_return = True
                    max_returning_player_games = 0  # Just detected, 0 games played

                    # Add EX_BENEFICIARY record (skipping BENEFICIARY status)
                    logger.info(f"{player_name}: Marking as EX_BENEFICIARY of {returning_player_name} (excluding games {injury_date} to {today})")
                    injury_context = add_to_injury_context(
                        injury_context,
                        player_name=player_name,
                        team=player_row['TEAM'],
                        status='EX_BENEFICIARY',
                        true_baseline=0,  # Will be calculated from non-inflated games
                        beneficiary_of=returning_player_name,
                        injury_date=injury_date,  # When returning player got injured
                        return_date=today  # When returning player returns
                    )
                    logger.info(f"{player_name}: Affected by {returning_player_name} return (10 MPG for 4 games, then exclude inflated games from baseline)")

    # ========== STEP 3: Apply 10 MPG for RETURNING STATUS (Scenario B) ==========
    # Both returning star (Kawhi) AND affected teammates (Norman Powell) get 10 MPG for first 4 games
    # This prevents overestimation while we collect data on new rotation
    # NOTE: This is NOT for "Returning TODAY" players (Scenario A) - they get 25% reduction instead
    if is_returning_player or affected_by_return:
        if max_returning_player_games is not None and max_returning_player_games < 4:
            logger.info(f"{player_name}: RETURNING STATUS scenario - using conservative 10 MPG (returning player has {max_returning_player_games}/4 games)")
            return 10, 'LOW', injury_context

    # ========== STEP 4: Get baseline and check for ACTIVE injuries ==========
    # At this point, no season-long returns affecting this player
    # Check if any teammates are currently OUT and redistribute their minutes
    confidence = update_confidence_status(player_row, injury_context, games_log)
    baseline = get_player_baseline(player_row, injury_context, box_scores)  # EX_BENEFICIARY logic applied here

    # Use pre-calculated injury redistributions if provided, otherwise calculate on-demand (legacy)
    if team_injury_redistributions is None:
        # Legacy behavior: Calculate on-demand (less efficient)
        from injury_system import redistribute_injury_minutes

        team_injuries = team_players[team_players['STATUS'] == 'OUT']
        if 'RETURN_DATE_DT' in team_injuries.columns:
            teammates_returning_today = team_injuries['RETURN_DATE_DT'] <= today
            team_injuries = team_injuries[~teammates_returning_today]
            excluded_count = teammates_returning_today.sum()
            if excluded_count > 0:
                logger.debug(f"Excluded {excluded_count} teammates marked OUT but returning today (won't redistribute their minutes)")

        if not team_injuries.empty:
            team_injuries_sorted = team_injuries.sort_values('SEASON_AVG_MIN', ascending=False)

            # Try to find injury redistribution for this player
            for _, injured in team_injuries_sorted.iterrows():
                injury_projections, injury_context = redistribute_injury_minutes(
                    injured, team_players, injury_context, today, box_scores,
                    position_overlap_func, use_multiplier=use_multiplier
                )
                if player_name in injury_projections:
                    projected_min = injury_projections[player_name]

                    # Apply 25% reduction for RETURNING TODAY (Scenario A)
                    if returning_today:
                        projected_min = projected_min * 0.75
                        logger.info(f"{player_name}: RETURNING TODAY (Scenario A) - Applied 25% reduction: {injury_projections[player_name]:.1f} -> {projected_min:.1f} MPG")

                    return projected_min, confidence, injury_context
    else:
        # Optimized path: Use pre-calculated redistributions (no redundant calls)
        # Check if this player benefits from any injured teammate's minutes
        for _, injury_projections in team_injury_redistributions.items():
            if player_name in injury_projections:
                # Player benefits from this injury - use redistribution
                projected_min = injury_projections[player_name]

                # Apply 25% reduction for RETURNING TODAY (Scenario A)
                # Player still on injury report but expected to play today
                if returning_today:
                    projected_min = projected_min * 0.75
                    logger.info(f"{player_name}: RETURNING TODAY (Scenario A) - Applied 25% reduction: {injury_projections[player_name]:.1f} -> {projected_min:.1f} MPG")

                return projected_min, confidence, injury_context

        # Team has injuries but player doesn't benefit - fall through to Formula C

    # Use Formula C (either no injuries or player not affected by them)
    last_7_avg = player_row.get('LAST_7_AVG_MIN', baseline)
    prev_game = player_row.get('PREV_GAME_MIN', baseline)

    projected = 0.5 * baseline + 0.3 * last_7_avg + 0.2 * prev_game
    projected = min(projected, MAX_MINUTES)

    # Apply 25% reduction for RETURNING TODAY (Scenario A)
    # Player still on injury report but expected to play today
    if returning_today:
        original_proj = projected
        projected = projected * 0.75
        logger.info(f"{player_name}: RETURNING TODAY (Scenario A) - Applied 25% reduction: {original_proj:.1f} -> {projected:.1f} MPG")

    return projected, confidence, injury_context


# ==================== Model 1: Complex Position Overlap ====================

def project_minutes_complex(player_row, team_players, injury_context, games_log, today, box_scores, injury_data, team_injury_redistributions=None):
    """Complex position overlap with 2x multiplier for direct backups"""
    return project_minutes_with_injuries(
        player_row, team_players, injury_context, games_log, today, box_scores,
        position_overlap_func=position_overlap_complex,
        use_multiplier=True,
        injury_data=injury_data,
        team_injury_redistributions=team_injury_redistributions
    )


# ==================== Model 2: Direct Position Exchange ====================

def project_minutes_direct(player_row, team_players, injury_context, games_log, today, box_scores, injury_data, team_injury_redistributions=None):
    """Exact position match only, no 2x multiplier"""
    return project_minutes_with_injuries(
        player_row, team_players, injury_context, games_log, today, box_scores,
        position_overlap_func=position_overlap_exact,
        use_multiplier=False,
        injury_data=injury_data,
        team_injury_redistributions=team_injury_redistributions
    )


# ==================== Model 3: Formula C Baseline (No Injury Handling) ====================

def project_minutes_formula_c(player_row, today):
    """Pure Formula C - no injury redistribution"""
    # Check if player is returning today (still on injury report but expected to play)
    returning_today = False
    if 'RETURN_DATE_DT' in player_row and pd.notna(player_row.get('RETURN_DATE_DT')):
        return_date = player_row['RETURN_DATE_DT']
        if isinstance(return_date, pd.Timestamp):
            return_date = return_date.date()
        if return_date == today:
            returning_today = True

    # If OUT but returning today, treat as playing (with 25% reduction)
    if player_row.get('STATUS') == 'OUT' and not returning_today:
        return 0, 'HIGH'

    # Get baseline with fallbacks (conservative approach)
    if player_row.get('GAMES_PLAYED', 0) >= 4:
        # Reliable: 4+ games this season
        baseline = player_row.get('SEASON_AVG_MIN', 10)
    elif player_row.get('FROM_PREV_SEASON', False):
        # Season-long injury (0 games this season) - check if same team
        prev_team = player_row.get('PREV_TEAM')
        current_team = player_row.get('TEAM')

        if prev_team == current_team:
            # Same team - use previous season baseline (reliable)
            baseline = player_row.get('SEASON_AVG_MIN', 10)
        else:
            # TRADED - previous season data unreliable, use conservative fallback
            baseline = 10
    else:
        # < 4 games this season - UNRELIABLE, use conservative fallback
        baseline = 10

    last_7_avg = player_row.get('LAST_7_AVG_MIN', baseline)
    prev_game = player_row.get('PREV_GAME_MIN', baseline)

    # Formula C: 50% season, 30% last 7, 20% prev game
    projected = 0.5 * baseline + 0.3 * last_7_avg + 0.2 * prev_game
    projected = min(projected, MAX_MINUTES)

    # Apply 25% reduction if returning today
    if returning_today:
        original_proj = projected
        projected = projected * 0.75
        logger.info(f"{player_row['PLAYER']}: Applied 25% reduction (returning today): {original_proj:.1f} -> {projected:.1f} MPG")

    return projected, 'HIGH'
