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


# ==================== Helper Functions ====================

def build_projection_dict(player_row, projected_min, confidence, today):
    """
    Build standard projection dictionary

    Args:
        player_row: Player data row
        projected_min: Projected minutes
        confidence: Confidence level
        today: Date for projection

    Returns:
        Dictionary with standard projection fields
    """
    return {
        'DATE': today,
        'PLAYER': player_row['PLAYER'],
        'TEAM': player_row['TEAM'],
        'POSITION': player_row.get('POSITION', 'UNKNOWN'),
        'PROJECTED_MIN': round(projected_min, 1),
        'ACTUAL_MIN': None,
        'ACTUAL_FP': None,
        'CONFIDENCE': confidence
    }


def generate_team_based_projections(
    player_stats,
    injury_context,
    today,
    box_scores,
    injury_data,
    games_log,
    position_overlap_func,
    project_func,
    use_multiplier
):
    """
    Generate projections for team-based models (Complex and Direct)

    This helper handles the common pattern:
    1. Loop through teams
    2. Calculate team injury redistributions once per team
    3. Generate projections for each player on the team
    4. Build projection dictionaries

    Args:
        player_stats: DataFrame with all player stats
        injury_context: Current injury context DataFrame
        today: Date for projections
        box_scores: Box score data
        injury_data: Current injury report
        games_log: Games log for confidence tracking
        position_overlap_func: position_overlap_complex or position_overlap_exact
        project_func: project_minutes_complex or project_minutes_direct
        use_multiplier: Boolean for 2x multiplier (True for complex, False for direct)

    Returns:
        Tuple of (projections_list, updated_injury_context)
    """
    from injury_system import calculate_team_injury_redistributions

    teams = player_stats['TEAM'].unique()
    projections = []
    injury_ctx = injury_context.copy()

    for team in teams:
        team_players = player_stats[player_stats['TEAM'] == team].copy()

        # Calculate injury redistributions ONCE per team (avoids redundant calculations)
        team_redistributions, injury_ctx = calculate_team_injury_redistributions(
            team_players, injury_ctx, today, box_scores,
            position_overlap_func=position_overlap_func,
            use_multiplier=use_multiplier
        )

        # Generate projections for each player on the team
        for _, player in team_players.iterrows():
            projected_min, confidence, injury_ctx = project_func(
                player, team_players, injury_ctx, games_log, today, box_scores, injury_data,
                team_injury_redistributions=team_redistributions
            )

            projections.append(build_projection_dict(player, projected_min, confidence, today))

    return projections, injury_ctx


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

    Handles two returning-from-injury scenarios:

    A. EXTENDED ABSENCE (25% reduction):
       - Missed 10+ team games → Apply 25% reduction to projection (first game back only)

    B. SEASON-LONG RETURN (10 MPG for 3 games):
       - Star (20+ MPG) missed entire season → Returning star + affected teammates get 10 MPG for 3 games

    Other scenarios:
    3. Active injuries (teammate OUT) → Redistribute their minutes
    4. Normal play → Use Formula C

    Args:
        position_overlap_func: Either position_overlap_complex or position_overlap_exact
        use_multiplier: If True, apply 2x multiplier for exact position (complex model)
        injury_data: DataFrame of current injury report data (to filter out free agents)

    Returns:
        (projected_minutes, confidence, updated_injury_context)
    """
    player_name = player_row['PLAYER']
    player_team = player_row['TEAM']

    # ========== CHECK: Is player returning from extended absence (10+ games)? ==========
    # If player hasn't played in 10+ team games, apply reduction for first game back
    returning_from_absence = False
    games_missed = 0

    if not games_log.empty and player_row.get('STATUS') != 'OUT':
        # Get player's most recent game
        player_games = games_log[games_log['PLAYER'] == player_name].copy()

        if not player_games.empty and 'GAME_DATE' in player_games.columns:
            player_games['GAME_DATE'] = pd.to_datetime(player_games['GAME_DATE'])
            last_player_game = player_games['GAME_DATE'].max()

            # Get team's games since player's last appearance
            team_games = games_log[games_log['TEAM'] == player_team].copy()
            if not team_games.empty and 'GAME_DATE' in team_games.columns:
                team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'])
                games_since_last_appearance = team_games[team_games['GAME_DATE'] > last_player_game]

                # Count unique game dates (in case multiple players from same team in one game)
                games_missed = games_since_last_appearance['GAME_DATE'].nunique()

                if games_missed >= 10:
                    returning_from_absence = True
                    logger.info(f"{player_name}: Returning from {games_missed}-game absence - will apply 25% minutes reduction")

    # ========== STEP 1: Handle currently injured players (status OUT) ==========
    if player_row.get('STATUS') == 'OUT':
        return 0, 'HIGH', injury_context

    # ========== STEP 2: Season-Long Returns (10 MPG for 3 games) ==========
    # Example: Kawhi 0 games this season, now healthy → Kawhi + affected teammates get 10 MPG for 3 games
    # Two loops: (1) Process already-tracked, (2) Detect new

    # Loop 1: Already-tracked (marked RETURNING in previous runs)
    returning_players = injury_context[
        (injury_context['STATUS'] == 'RETURNING') &
        (injury_context['TEAM'] == player_row['TEAM'])
    ]

    # Track if current player is returning OR affected by a returning player
    is_returning_player = False
    affected_by_return = False
    max_returning_player_games = None
    for _, returning_record in returning_players.iterrows():
        returning_player_name = returning_record['PLAYER']
        return_date = pd.to_datetime(returning_record['INJURY_DATE'])

        # Count games played since return
        returning_games = games_log[
            (games_log['PLAYER'] == returning_player_name) &
            (games_log['GAME_DATE'] >= return_date)
        ]
        games_since_return = len(returning_games)

        # Check if player still on team and is a star (20+ MPG)
        returning_player_full = team_players[team_players['PLAYER'] == returning_player_name]
        if returning_player_full.empty or returning_player_full.iloc[0].get('CAREER_AVG_MIN', 0) < 20:
            # Player traded/cut or not a star → Remove stale entry
            injury_context = remove_from_injury_context(injury_context, returning_player_name)
            continue

        if games_since_return >= 3:
            # Played 3+ games → Remove from tracking (enough data now)
            logger.info(f"{returning_player_name} played {games_since_return} games since return - removing from tracking")
            injury_context = remove_from_injury_context(injury_context, returning_player_name)
        else:
            # Still in 3-game conservative period

            if player_name == returning_player_name:
                # Current player IS the returning star
                is_returning_player = True
                max_returning_player_games = games_since_return
                logger.info(f"{player_name}: Returning star with {games_since_return}/3 games played")
            else:
                # Check if current player affected by returning star (position overlap)
                returning_pos = returning_player_full.iloc[0].get('POSITION', 'SF')
                if position_overlap_func(player_row.get('POSITION', 'SF'), returning_pos):
                    affected_by_return = True
                    if max_returning_player_games is None or games_since_return > max_returning_player_games:
                        max_returning_player_games = games_since_return
                    logger.info(f"{player_name}: Affected by {returning_player_name} return ({games_since_return}/3 games)")

                    # Mark as EX_BENEFICIARY to exclude inflated games from future baseline
                    already_ex_beneficiary = injury_context[
                        (injury_context['PLAYER'] == player_name) &
                        (injury_context['STATUS'] == 'EX_BENEFICIARY') &
                        (injury_context['BENEFICIARY_OF'] == returning_player_name)
                    ]

                    if already_ex_beneficiary.empty:
                        # Get when returning star got injured
                        injury_date = returning_player_full.iloc[0].get('ESTIMATED_INJURY_DATE')
                        if pd.isna(injury_date) or injury_date is None:
                            # Estimate as start of season (Oct 15)
                            current_year = pd.Timestamp(today).year
                            season_start = pd.Timestamp(f"{current_year}-10-15").date()
                            if pd.Timestamp(today).month < 7:
                                season_start = pd.Timestamp(f"{current_year - 1}-10-15").date()
                            injury_date = season_start

                        logger.info(f"{player_name}: Marking as EX_BENEFICIARY of {returning_player_name} (exclude {injury_date} to {return_date})")
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
    # Loop 2: Detect new (first time seeing this return)
    if not injury_data.empty:
        injured_or_returning_players = set(injury_data['PLAYER'].unique())
    else:
        injured_or_returning_players = set()
    for _, teammate in team_players.iterrows():
        # Check if player qualifies as season-long return
        if (teammate.get('GAMES_PLAYED', 0) == 0 and           # 0 games this season
            teammate.get('FROM_PREV_SEASON', False) and        # Has data from last season
            teammate.get('STATUS') != 'OUT' and                # Now healthy
            teammate.get('CAREER_AVG_MIN', 0) >= 20 and        # Star (20+ MPG career)
            teammate['PLAYER'] in injured_or_returning_players):  # Was on injury report

            returning_player_name = teammate['PLAYER']

            # Check if already tracked from previous run
            already_tracked = injury_context[
                (injury_context['PLAYER'] == returning_player_name) &
                (injury_context['STATUS'] == 'RETURNING')
            ]

            if already_tracked.empty:
                # First time detecting this return → Add to tracking
                injury_date = teammate.get('ESTIMATED_INJURY_DATE')
                if pd.isna(injury_date) or injury_date is None:
                    # No injury date → Estimate as season start (Oct 15)
                    current_year = pd.Timestamp(today).year
                    season_start = pd.Timestamp(f"{current_year}-10-15").date()
                    if pd.Timestamp(today).month < 7:
                        season_start = pd.Timestamp(f"{current_year - 1}-10-15").date()
                    injury_date = season_start
                    logger.info(f"{returning_player_name}: No injury date, using season start {injury_date}")

                logger.info(f"NEW season-long return: {returning_player_name} (10 MPG for 3 games)")
                injury_context = add_to_injury_context(
                    injury_context,
                    player_name=returning_player_name,
                    team=player_row['TEAM'],
                    status='RETURNING',
                    true_baseline=0,
                    beneficiary_of=None,
                    injury_date=today
                )

                # Check if current player is affected
                if player_name == returning_player_name:
                    # Current player IS the returning star
                    is_returning_player = True
                    max_returning_player_games = 0
                elif position_overlap_func(player_row.get('POSITION', 'SF'), teammate.get('POSITION', 'SF')):
                    # Current player affected by returning star (position overlap)
                    affected_by_return = True
                    max_returning_player_games = 0

                    logger.info(f"{player_name}: Affected by {returning_player_name} return (10 MPG for 3 games)")
                    injury_context = add_to_injury_context(
                        injury_context,
                        player_name=player_name,
                        team=player_row['TEAM'],
                        status='EX_BENEFICIARY',
                        true_baseline=0,
                        beneficiary_of=returning_player_name,
                        injury_date=injury_date,
                        return_date=today
                    )

    # ========== STEP 3: Apply 10 MPG for RETURNING STATUS ==========
    # If returning star or affected teammate → 10 MPG for first 3 games
    if is_returning_player or affected_by_return:
        if max_returning_player_games is not None and max_returning_player_games < 3:
            logger.info(f"{player_name}: RETURNING STATUS - 10 MPG ({max_returning_player_games}/3 games)")
            return 10, 'LOW', injury_context

    # ========== STEP 4: Normal projection (Formula C or injury redistribution) ==========
    confidence = update_confidence_status(player_row, injury_context, games_log)
    baseline = get_player_baseline(player_row, injury_context, box_scores)  # EX_BENEFICIARY logic applied here

    # Use pre-calculated injury redistributions if provided, otherwise calculate on-demand (legacy)
    if team_injury_redistributions is None:
        # Legacy behavior: Calculate on-demand (less efficient)
        from injury_system import redistribute_injury_minutes

        team_injuries = team_players[team_players['STATUS'] == 'OUT']

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

                    # Apply 25% reduction if returning from extended absence (10+ games)
                    if returning_from_absence:
                        projected_min = projected_min * 0.75
                        logger.info(f"{player_name}: Returning from {games_missed}-game absence - Applied 25% reduction: {injury_projections[player_name]:.1f} -> {projected_min:.1f} MPG")

                    return projected_min, confidence, injury_context
    else:
        # Optimized path: Use pre-calculated redistributions (no redundant calls)
        # Check if this player benefits from any injured teammate's minutes
        for _, injury_projections in team_injury_redistributions.items():
            if player_name in injury_projections:
                # Player benefits from this injury - use redistribution
                projected_min = injury_projections[player_name]

                # Apply 25% reduction if returning from extended absence (10+ games)
                if returning_from_absence:
                    projected_min = projected_min * 0.75
                    logger.info(f"{player_name}: Returning from {games_missed}-game absence - Applied 25% reduction: {injury_projections[player_name]:.1f} -> {projected_min:.1f} MPG")

                return projected_min, confidence, injury_context

        # Team has injuries but player doesn't benefit - fall through to Formula C

    # Use Formula C (either no injuries or player not affected by them)
    last_7_avg = player_row.get('LAST_7_AVG_MIN', baseline)
    prev_game = player_row.get('PREV_GAME_MIN', baseline)

    projected = 0.5 * baseline + 0.3 * last_7_avg + 0.2 * prev_game
    projected = min(projected, MAX_MINUTES)

    # Apply 25% reduction if returning from extended absence (10+ games)
    if returning_from_absence:
        original_proj = projected
        projected = projected * 0.75
        logger.info(f"{player_name}: Returning from {games_missed}-game absence - Applied 25% reduction: {original_proj:.1f} -> {projected:.1f} MPG")

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

def project_minutes_formula_c(player_row, today, games_log):
    """Pure Formula C - no injury redistribution"""
    # Check if player is currently injured
    if player_row.get('STATUS') == 'OUT':
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

    # Apply starter floor if player is listed as starter (DailyFantasyFuel data)
    starter_status = player_row.get('STARTER_STATUS')
    player_name = player_row['PLAYER']
    if starter_status == 'CONFIRMED' and baseline < 24:
        logger.info(f"{player_name}: CONFIRMED starter floor applied to Formula C baseline ({baseline:.1f} -> 24.0 MPG)")
        baseline = 24
    elif starter_status == 'EXPECTED' and baseline < 20:
        logger.info(f"{player_name}: EXPECTED starter floor applied to Formula C baseline ({baseline:.1f} -> 20.0 MPG)")
        baseline = 20

    last_7_avg = player_row.get('LAST_7_AVG_MIN', baseline)
    prev_game = player_row.get('PREV_GAME_MIN', baseline)

    # Formula C: 50% season, 30% last 7, 20% prev game
    projected = 0.5 * baseline + 0.3 * last_7_avg + 0.2 * prev_game
    projected = min(projected, MAX_MINUTES)

    # Check if returning from extended absence (10+ games) and apply reduction
    # Note: Formula C doesn't have full injury context, so this is a simplified check
    player_name = player_row['PLAYER']
    player_team = player_row['TEAM']

    if not games_log.empty:
        player_games = games_log[games_log['PLAYER'] == player_name].copy()

        if not player_games.empty and 'GAME_DATE' in player_games.columns:
            player_games['GAME_DATE'] = pd.to_datetime(player_games['GAME_DATE'])
            last_player_game = player_games['GAME_DATE'].max()

            # Get team's games since player's last appearance
            team_games = games_log[games_log['TEAM'] == player_team].copy()
            if not team_games.empty and 'GAME_DATE' in team_games.columns:
                team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'])
                games_since_last_appearance = team_games[team_games['GAME_DATE'] > last_player_game]
                games_missed = games_since_last_appearance['GAME_DATE'].nunique()

                if games_missed >= 10:
                    original_proj = projected
                    projected = projected * 0.75
                    logger.info(f"{player_name}: Returning from {games_missed}-game absence - Applied 25% reduction: {original_proj:.1f} -> {projected:.1f} MPG")

    return projected, 'HIGH'
