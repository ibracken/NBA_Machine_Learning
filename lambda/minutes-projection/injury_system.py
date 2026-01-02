"""
Injury tracking and baseline calculation system
"""

import pandas as pd
import numpy as np
import logging
import pytz
from datetime import datetime
from config import MAX_MINUTES, BENCH_OPPORTUNITY_CONSTANT, EXACT_POSITION_MULTIPLIER, CONFIDENCE_CLEARANCE_GAMES

logger = logging.getLogger()


# ==================== Injury Context Helper Functions ====================

def transition_beneficiaries_to_ex(injury_context, currently_injured, today):
    """
    Transition statuses when injured player returns (MID-SEASON INJURIES ONLY):
    - BENEFICIARY → EX_BENEFICIARY (will revert to pre-injury baseline)
    - ROLE_DECREASED → Keep unchanged (will keep using recent games)

    This handles the normal flow where:
    1. Player gets injured mid-season → teammates marked as BENEFICIARY
    2. Player returns → this function transitions BENEFICIARY → EX_BENEFICIARY

    DOES NOT HANDLE: Season-long returns (e.g., Kawhi out all season, never had BENEFICIARY status)
    Those are detected in project_minutes_with_injuries() which has access to player_stats
    and can check for GAMES_PLAYED=0, FROM_PREV_SEASON=True conditions.

    Args:
        injury_context: DataFrame of injury context to process
        currently_injured: Set of player names who are still injured
        today: Current date (for storing return_date)

    Returns:
        Updated injury_context DataFrame
    """
    if injury_context.empty:
        return injury_context

    # Handle BENEFICIARY transitions
    beneficiaries = injury_context[injury_context['STATUS'] == 'BENEFICIARY']

    for _, beneficiary in beneficiaries.iterrows():
        injured_player = beneficiary['BENEFICIARY_OF']

        if injured_player not in currently_injured:
            # Injured player returned - transition this beneficiary relationship
            logger.info(f"{beneficiary['PLAYER']}: Transitioning BENEFICIARY -> EX_BENEFICIARY ({injured_player} returned)")

            # Create new EX_BENEFICIARY record
            injury_context = add_to_injury_context(
                injury_context,
                player_name=beneficiary['PLAYER'],
                team=beneficiary['TEAM'],
                status='EX_BENEFICIARY',
                true_baseline=beneficiary['TRUE_BASELINE'],
                beneficiary_of=beneficiary['BENEFICIARY_OF'],
                injury_date=beneficiary['INJURY_DATE'],
                return_date=today
            )

            # Remove only the specific BENEFICIARY record for this returned player
            injury_context = injury_context[
                ~((injury_context['PLAYER'] == beneficiary['PLAYER']) &
                  (injury_context['STATUS'] == 'BENEFICIARY') &
                  (injury_context['BENEFICIARY_OF'] == injured_player))
            ].copy()

    # KEEP ROLE_DECREASED status even when injured player returns
    # These players permanently lost minutes - continue using recent games (don't revert to old baseline)
    # Example: Kennard lost minutes when Trae injured, should stay at reduced role even after Trae returns

    return injury_context


def add_to_injury_context(injury_context, player_name, team, status, true_baseline, beneficiary_of, injury_date, return_date=None):
    """
    Add or update player in injury context

    For EX_BENEFICIARY and BENEFICIARY: Allows multiple rows per player (multiple injury periods or multiple injuries benefited from)
    For RETURNING: Only replaces existing RETURNING rows (keeps BENEFICIARY and EX_BENEFICIARY)
    For other statuses: Replaces existing row
    """
    # For EX_BENEFICIARY and BENEFICIARY, allow multiple rows per player
    # For RETURNING, only remove old RETURNING rows (keep BENEFICIARY and EX_BENEFICIARY)
    # For other statuses, replace existing row
    if status == 'RETURNING':
        # Keep BENEFICIARY and EX_BENEFICIARY, only remove old RETURNING
        injury_context = injury_context[
            ~((injury_context['PLAYER'] == player_name) &
              (injury_context['STATUS'] == 'RETURNING'))
        ].copy()
    elif status not in ['EX_BENEFICIARY', 'BENEFICIARY']:
        injury_context = injury_context[injury_context['PLAYER'] != player_name].copy()

    new_record = pd.DataFrame([{
        'PLAYER': player_name,
        'TEAM': team,
        'STATUS': status,
        'INJURY_DATE': injury_date,
        'RETURN_DATE': return_date,
        'TRUE_BASELINE': true_baseline,
        'BENEFICIARY_OF': beneficiary_of,
        'UPDATED_DATE': datetime.now(pytz.timezone('US/Eastern'))
    }])

    return pd.concat([injury_context, new_record], ignore_index=True)


def remove_from_injury_context(injury_context, player_name):
    """Remove player and their beneficiaries from injury context"""
    injury_context = injury_context[injury_context['PLAYER'] != player_name].copy()
    injury_context = injury_context[injury_context['BENEFICIARY_OF'] != player_name].copy()
    return injury_context


def get_pre_injury_baseline(player_name, box_scores, injury_date):
    """
    Calculate baseline EXCLUDING games played after injury_date

    This prevents the TRUE_BASELINE from being inflated by games where the player
    was already filling in for an injured teammate.

    Args:
        player_name: Name of the player
        box_scores: Full season box scores
        injury_date: Date when teammate got injured (exclude games after this)

    Returns:
        Baseline minutes (float) or None if insufficient data
    """
    player_games = box_scores[box_scores['PLAYER'] == player_name].copy()

    if player_games.empty or 'GAME_DATE' not in player_games.columns:
        return None

    player_games['GAME_DATE'] = pd.to_datetime(player_games['GAME_DATE'])
    injury_date = pd.to_datetime(injury_date)

    # Only use games BEFORE the injury occurred
    pre_injury_games = player_games[player_games['GAME_DATE'] < injury_date].copy()

    if len(pre_injury_games) == 0:
        # No games before injury - can't calculate baseline
        return None

    # Use weighted average (recent games get 2x weight)
    today = pd.Timestamp.now()
    recent_cutoff = today - pd.Timedelta(days=14)
    pre_injury_games['weight'] = pre_injury_games['GAME_DATE'].apply(
        lambda d: 2.0 if d >= recent_cutoff else 1.0
    )

    baseline = (pre_injury_games['MIN'] * pre_injury_games['weight']).sum() / pre_injury_games['weight'].sum()

    logger.info(f"{player_name}: Pre-injury baseline = {baseline:.1f} MPG "
                f"(using {len(pre_injury_games)} games before {injury_date.date()})")
    return baseline


def get_player_baseline(player_row, injury_context, box_scores):
    """
    Get player's TRUE baseline minutes with fallback chain

    For EX_BENEFICIARY players: Excludes inflation periods (games between INJURY_DATE and RETURN_DATE)
    to prevent compounding minutes problem

    Conservative approach: Use 10 MPG fallback when baseline is unreliable (< 4 games, traded players)
    """
    player_name = player_row['PLAYER']

    # Check if player is CURRENTLY benefiting from injury (active beneficiary)
    active_beneficiary = injury_context[
        (injury_context['PLAYER'] == player_name) &
        (injury_context['STATUS'] == 'BENEFICIARY')
    ]

    if not active_beneficiary.empty:
        # Return frozen baseline from before they started filling in
        baseline = active_beneficiary['TRUE_BASELINE'].iloc[0]

        # Apply starter floor if applicable
        starter_status = player_row.get('STARTER_STATUS')
        if starter_status == 'CONFIRMED' and baseline < 24:
            logger.info(f"{player_name}: CONFIRMED starter floor applied to BENEFICIARY baseline ({baseline:.1f} -> 24.0 MPG)")
            baseline = 24
        elif starter_status == 'EXPECTED' and baseline < 20:
            logger.info(f"{player_name}: EXPECTED starter floor applied to BENEFICIARY baseline ({baseline:.1f} -> 20.0 MPG)")
            baseline = 20

        return baseline

    # Check for ROLE_DECREASED status (player squeezed out during teammate's injury)
    # These players should NOT revert to pre-injury baseline when injured player returns
    # Use recent games instead to reflect their new, reduced role
    role_decreased = injury_context[
        (injury_context['PLAYER'] == player_name) &
        (injury_context['STATUS'] == 'ROLE_DECREASED')
    ]

    if not role_decreased.empty:
        # Use recent performance, not pre-injury baseline
        # This player's role permanently decreased during the injury period
        recent_baseline = player_row.get('LAST_7_AVG_MIN')
        if recent_baseline is not None and recent_baseline > 0:
            logger.info(f"{player_name}: ROLE_DECREASED status - using recent baseline {recent_baseline:.1f} MPG (not reverting to pre-injury)")
            baseline = recent_baseline
        else:
            # Fallback to season average if no recent games
            baseline = player_row.get('SEASON_AVG_MIN', 10)

        # Apply starter floor if applicable
        starter_status = player_row.get('STARTER_STATUS')
        if starter_status == 'CONFIRMED' and baseline < 24:
            logger.info(f"{player_name}: CONFIRMED starter floor applied to ROLE_DECREASED baseline ({baseline:.1f} -> 24.0 MPG)")
            baseline = 24
        elif starter_status == 'EXPECTED' and baseline < 20:
            logger.info(f"{player_name}: EXPECTED starter floor applied to ROLE_DECREASED baseline ({baseline:.1f} -> 20.0 MPG)")
            baseline = 20

        return baseline

    # Check if player WAS a beneficiary (ex-beneficiary with historical INJURY_DATE)
    # Can have MULTIPLE rows if player filled in during multiple injury periods
    ex_beneficiary_records = injury_context[
        (injury_context['PLAYER'] == player_name) &
        (injury_context['STATUS'] == 'EX_BENEFICIARY')
    ]

    if not ex_beneficiary_records.empty:
        # Ex-beneficiary - calculate baseline excluding ALL inflation periods
        player_games = box_scores[box_scores['PLAYER'] == player_name].copy()

        if not player_games.empty and 'GAME_DATE' in player_games.columns:
            player_games['GAME_DATE'] = pd.to_datetime(player_games['GAME_DATE'])

            # Start with all games valid, then exclude each inflation period
            valid_games_mask = pd.Series([True] * len(player_games), index=player_games.index)

            for _, record in ex_beneficiary_records.iterrows():
                injury_date = pd.to_datetime(record['INJURY_DATE'])
                return_date = pd.to_datetime(record['RETURN_DATE'])

                # Exclude games in this inflation period
                inflation_mask = (
                    (player_games['GAME_DATE'] >= injury_date) &
                    (player_games['GAME_DATE'] < return_date)
                )
                valid_games_mask = valid_games_mask & ~inflation_mask

            valid_games = player_games[valid_games_mask].copy()

            if len(valid_games) >= 1:
                # Weight recent games (last 14 days) at 2.0x for faster convergence
                today = pd.Timestamp.now()
                recent_cutoff = today - pd.Timedelta(days=14)
                valid_games['weight'] = valid_games['GAME_DATE'].apply(
                    lambda d: 2.0 if d >= recent_cutoff else 1.0
                )

                # Weighted average of minutes
                historical_baseline = (valid_games['MIN'] * valid_games['weight']).sum() / valid_games['weight'].sum()
                excluded_count = len(player_games) - len(valid_games)
                recent_games = (valid_games['GAME_DATE'] >= recent_cutoff).sum()

                logger.info(f"{player_name}: Ex-beneficiary baseline = {historical_baseline:.1f} MPG "
                           f"(excluding {excluded_count} inflated games, {recent_games} recent games weighted 2x)")
                baseline = historical_baseline
            else:
                # No valid games - use conservative fallback
                logger.warning(f"{player_name}: No valid games for ex-beneficiary - using conservative 10 MPG")
                baseline = 10

            # Apply starter floor if applicable
            starter_status = player_row.get('STARTER_STATUS')
            if starter_status == 'CONFIRMED' and baseline < 24:
                logger.info(f"{player_name}: CONFIRMED starter floor applied to EX_BENEFICIARY baseline ({baseline:.1f} -> 24.0 MPG)")
                baseline = 24
            elif starter_status == 'EXPECTED' and baseline < 20:
                logger.info(f"{player_name}: EXPECTED starter floor applied to EX_BENEFICIARY baseline ({baseline:.1f} -> 20.0 MPG)")
                baseline = 20

            return baseline

    # Not a beneficiary - normal baseline calculation
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
            # TRADED - previous season data unreliable (different role on old team)
            logger.warning(f"{player_name}: Season-long injury on new team after trade ({prev_team} -> {current_team}) - using conservative 10 MPG baseline")
            baseline = 10  # Conservative fallback
    else:
        # < 4 games this season - UNRELIABLE
        # Don't use CAREER_AVG_MIN (could be inflated from old team/role)
        logger.debug(f"{player_name}: Insufficient games ({player_row.get('GAMES_PLAYED', 0)}) - using conservative 10 MPG baseline")
        baseline = 10  # Conservative fallback

    # Apply starter floor if player is listed as starter (DailyFantasyFuel data)
    # This catches: bench-to-starter promotions, rookies/new players starting, traded players starting
    # Does NOT interfere with high-minute starters (max preserves their historical baseline)
    starter_status = player_row.get('STARTER_STATUS')
    if starter_status == 'CONFIRMED':
        if baseline < 24:
            logger.info(f"{player_name}: CONFIRMED starter floor applied ({baseline:.1f} -> 24.0 MPG)")
            baseline = 24
    elif starter_status == 'EXPECTED':
        if baseline < 20:
            logger.info(f"{player_name}: EXPECTED starter floor applied ({baseline:.1f} -> 20.0 MPG)")
            baseline = 20

    return baseline


def update_confidence_status(player_row, injury_context, games_log):
    """
    Update confidence level based on games played since injury

    NOTE: Confidence is METADATA ONLY and does NOT affect projection calculations.
    It's returned alongside projections for monitoring/logging purposes.
    """
    player_name = player_row['PLAYER']

    low_confidence_record = injury_context[
        (injury_context['PLAYER'] == player_name) &
        (injury_context['STATUS'].isin(['INJURED', 'BENEFICIARY']))
    ]

    if low_confidence_record.empty:
        return 'HIGH'

    injury_date = pd.to_datetime(low_confidence_record['INJURY_DATE'].iloc[0])
    games_since = games_log[games_log['GAME_DATE'] > injury_date]
    games_played_since = len(games_since)

    if games_played_since >= CONFIDENCE_CLEARANCE_GAMES:
        return 'HIGH'
    elif games_played_since == 0:
        return 'LOW'
    elif games_played_since < CONFIDENCE_CLEARANCE_GAMES:
        return 'MEDIUM'

    return 'HIGH'


# ==================== Injury Impact Analysis ====================

def analyze_injury_impact(teammates, box_scores, injury_date, injury_context, min_games=3):
    """
    Analyze how teammates' minutes changed during an injury period

    Compares pre-injury baseline vs during-injury performance to identify:
    - Players who gained minutes (true beneficiaries)
    - Players who lost minutes (squeezed out by rotation changes)
    - Players unaffected

    Args:
        teammates: DataFrame of teammates to analyze
        box_scores: Full season box scores
        injury_date: Date when injury occurred
        injury_context: Current injury tracking context (to check for existing beneficiaries)
        min_games: Minimum games required in each period for reliable comparison

    Returns:
        role_increased: List of (player_name, pre_baseline, during_avg, delta)
        role_decreased: List of (player_name, pre_baseline, during_avg, delta)
        unchanged: List of player_name
    """
    injury_date = pd.to_datetime(injury_date)
    role_increased = []
    role_decreased = []
    unchanged = []

    for _, player in teammates.iterrows():
        player_name = player['PLAYER']
        player_games = box_scores[box_scores['PLAYER'] == player_name].copy()

        if player_games.empty or 'GAME_DATE' not in player_games.columns:
            unchanged.append(player_name)
            continue

        player_games['GAME_DATE'] = pd.to_datetime(player_games['GAME_DATE'])

        # Split games into pre-injury and during-injury periods
        pre_injury_games = player_games[player_games['GAME_DATE'] < injury_date]
        during_injury_games = player_games[player_games['GAME_DATE'] >= injury_date]

        # Need minimum games in both periods for reliable comparison
        if len(pre_injury_games) < min_games or len(during_injury_games) < min_games:
            unchanged.append(player_name)
            continue

        # Check if player is ALREADY a beneficiary of another injury (stacking injuries)
        # If so, use their frozen TRUE_BASELINE instead of calculating from recent games
        existing_beneficiary = injury_context[
            (injury_context['PLAYER'] == player_name) &
            (injury_context['STATUS'] == 'BENEFICIARY')
        ]

        if not existing_beneficiary.empty:
            # Player already benefiting from another injury - use frozen baseline
            pre_baseline = existing_beneficiary['TRUE_BASELINE'].iloc[0]
            logger.info(f"{player_name}: Already BENEFICIARY of another injury - using frozen baseline {pre_baseline:.1f} MPG (prevents inflation from stacking injuries)")
        else:
            # Calculate baseline from pre-injury games
            pre_baseline = pre_injury_games['MIN'].mean()

        # Calculate post-injury average
        during_avg = during_injury_games['MIN'].mean()
        delta = during_avg - pre_baseline

        # Categorize based on change (threshold: ±3 MPG)
        if delta > 3:
            role_increased.append((player_name, pre_baseline, during_avg, delta))
            logger.info(f"{player_name}: Role increased during injury (+{delta:.1f} MPG: {pre_baseline:.1f} -> {during_avg:.1f})")
        elif delta < -3:
            role_decreased.append((player_name, pre_baseline, during_avg, delta))
            logger.info(f"{player_name}: Role decreased during injury ({delta:.1f} MPG: {pre_baseline:.1f} -> {during_avg:.1f})")
        else:
            unchanged.append(player_name)

    return role_increased, role_decreased, unchanged


# ==================== Injury Redistribution (Complex & Direct Models) ====================

def redistribute_injury_minutes(injured_player, team_players, injury_context, today, box_scores,
                                position_overlap_func, use_multiplier=True):
    """
    Redistribute minutes using TRUE baselines

    Args:
        box_scores: Full season box scores for baseline calculations
        position_overlap_func: Either position_overlap_complex or position_overlap_exact
        use_multiplier: If True, apply 2x multiplier for exact position (complex model)
    """
    # Skip redistribution for season-long injuries (0 games this season, using prev season data)
    # Their teammates' SEASON_AVG_MIN already accounts for their absence all season
    if injured_player.get('GAMES_PLAYED', 1) == 0 and injured_player.get('FROM_PREV_SEASON', False):
        logger.debug(
            f"Season-long injury detected: {injured_player['PLAYER']} (0 games this season). "
            f"Skipping redistribution - teammates' baselines already account for absence."
        )
        return {}, injury_context

    # ========== ONGOING INJURY CHECK (>= 2 games) ==========
    # For ongoing injuries, use post-injury Formula C instead of predictive redistribution
    # This adapts to actual rotation changes (handles positionless basketball)
    injury_date_to_use = injured_player.get('ESTIMATED_INJURY_DATE', today)
    if pd.isna(injury_date_to_use):
        injury_date_to_use = today

    # Count games since injury by finding MAX games played by any teammate
    # This handles cases where some teammates also got injured
    games_since_injury = 0
    for _, teammate in team_players.iterrows():
        if teammate['PLAYER'] == injured_player['PLAYER']:
            continue  # Skip the injured player

        teammate_games = box_scores[
            (box_scores['PLAYER'] == teammate['PLAYER']) &
            (box_scores['GAME_DATE'] >= pd.to_datetime(injury_date_to_use))
        ]

        # Track the maximum - this represents how many games the team has played
        games_since_injury = max(games_since_injury, len(teammate_games))

    if games_since_injury >= 2:
        logger.info(
            f"{injured_player['PLAYER']} out for {games_since_injury} games - using post-injury Formula C "
            f"(calculated from games after injury, adapts to actual rotation)"
        )

        # Find ALL active teammates (not just overlapping positions)
        # This handles positionless basketball - let the data show who actually got minutes
        # Example: McBride (PG) might absorb minutes from OG (SF) if coach goes small
        teammates = team_players[
            (team_players['PLAYER'] != injured_player['PLAYER']) &
            (team_players['STATUS'] != 'OUT')
        ].copy()

        if not teammates.empty:
            # Analyze how teammates' roles actually changed during injury
            role_increased, role_decreased, unchanged = analyze_injury_impact(
                teammates, box_scores, injury_date_to_use, injury_context, min_games=3
            )

            # Track role changes in injury_context (for when injured player returns)
            for player_name, pre_baseline, during_avg, delta in role_increased:
                # Check if already tracked
                existing = injury_context[
                    (injury_context['PLAYER'] == player_name) &
                    (injury_context['STATUS'] == 'BENEFICIARY') &
                    (injury_context['BENEFICIARY_OF'] == injured_player['PLAYER'])
                ]

                if existing.empty:
                    # Mark as BENEFICIARY with clean pre-injury baseline
                    injury_context = add_to_injury_context(
                        injury_context,
                        player_name=player_name,
                        team=injured_player['TEAM'],
                        status='BENEFICIARY',
                        true_baseline=pre_baseline,  # Clean pre-injury baseline
                        beneficiary_of=injured_player['PLAYER'],
                        injury_date=injury_date_to_use
                    )
                    logger.info(f"{player_name}: Tracked as BENEFICIARY (baseline {pre_baseline:.1f} MPG, currently {during_avg:.1f} MPG)")

            # Track role_decreased players so they don't revert to old baseline when injury ends
            for player_name, pre_baseline, during_avg, delta in role_decreased:
                # Check if already tracked
                existing = injury_context[
                    (injury_context['PLAYER'] == player_name) &
                    (injury_context['STATUS'] == 'ROLE_DECREASED') &
                    (injury_context['BENEFICIARY_OF'] == injured_player['PLAYER'])
                ]

                if existing.empty:
                    # Mark as ROLE_DECREASED
                    injury_context = add_to_injury_context(
                        injury_context,
                        player_name=player_name,
                        team=injured_player['TEAM'],
                        status='ROLE_DECREASED',
                        true_baseline=pre_baseline,  # Store original baseline for reference
                        beneficiary_of=injured_player['PLAYER'],
                        injury_date=injury_date_to_use
                    )
                    logger.info(f"{player_name}: Tracked as ROLE_DECREASED (baseline {pre_baseline:.1f} MPG, currently {during_avg:.1f} MPG)")

        # Calculate post-injury Formula C for all teammates
        # This uses ONLY games played after the injury to reflect actual rotation changes
        projections = {}
        for _, player in teammates.iterrows():
            player_name = player['PLAYER']

            # Get post-injury games only
            player_post_injury = box_scores[
                (box_scores['PLAYER'] == player_name) &
                (box_scores['GAME_DATE'] >= pd.to_datetime(injury_date_to_use))
            ].copy()

            if len(player_post_injury) >= 3:
                # Enough post-injury data - use post-injury Formula C
                post_injury_avg = player_post_injury['MIN'].mean()
                post_injury_last_7 = player_post_injury.tail(7)['MIN'].mean()
                post_injury_prev = player_post_injury.iloc[-1]['MIN'] if not player_post_injury.empty else post_injury_avg

                projected = 0.5 * post_injury_avg + 0.3 * post_injury_last_7 + 0.2 * post_injury_prev
                projected = min(projected, MAX_MINUTES)
                projections[player_name] = projected

                logger.debug(f"{player_name}: Post-injury Formula C = {projected:.1f} MPG (avg:{post_injury_avg:.1f}, last7:{post_injury_last_7:.1f}, prev:{post_injury_prev:.1f})")

        return projections, injury_context

    # ========== NEW INJURY (<= 2 games): Use predictive redistribution ==========
    logger.info(f"{injured_player['PLAYER']} recently injured ({games_since_injury} games) - using predictive redistribution")

    # Get injured player's baseline
    if injured_player.get('GAMES_PLAYED', 0) >= 4:
        # Reliable: 4+ games this season
        lost_minutes = injured_player.get('SEASON_AVG_MIN', 5)
    elif injured_player.get('FROM_PREV_SEASON', False):
        # Season-long injury (0 games this season) - check if same team
        prev_team = injured_player.get('PREV_TEAM')
        current_team = injured_player.get('TEAM')

        if prev_team == current_team:
            # Same team - use previous season baseline (reliable)
            lost_minutes = injured_player.get('SEASON_AVG_MIN', 5)
        else:
            # TRADED - previous season data unreliable, use conservative fallback
            logger.warning(f"{injured_player['PLAYER']}: Season-long injury after trade - using conservative 5 MPG baseline")
            lost_minutes = 5
    else:
        # < 4 games this season - UNRELIABLE, use conservative fallback
        logger.debug(f"{injured_player['PLAYER']}: Insufficient games ({injured_player.get('GAMES_PLAYED', 0)}) - using conservative 5 MPG baseline")
        lost_minutes = 5

    # Find teammates at same/overlapping position
    injured_position = injured_player.get('POSITION', 'SF')
    teammates = team_players[
        (team_players['PLAYER'] != injured_player['PLAYER']) &
        (team_players['STATUS'] != 'OUT') &
        (team_players['POSITION'].apply(lambda pos: position_overlap_func(pos, injured_position)))
    ].copy()

    if teammates.empty:
        logger.debug(f"No teammates available to absorb {injured_player['PLAYER']}'s minutes")
        return {}, injury_context

    # Calculate TRUE baselines (with pre-injury logic to prevent inflation)
    # Note: injury_date_to_use already calculated at top of function
    baselines = {}
    for _, player in teammates.iterrows():
        player_name = player['PLAYER']

        # FIRST: Check if player already has a frozen baseline (existing beneficiary)
        active_beneficiary = injury_context[
            (injury_context['PLAYER'] == player_name) &
            (injury_context['STATUS'] == 'BENEFICIARY')
        ]

        if not active_beneficiary.empty:
            # Player already tracked - use frozen baseline
            baselines[player_name] = active_beneficiary['TRUE_BASELINE'].iloc[0]
        else:
            # NEW beneficiary - calculate baseline excluding post-injury games
            pre_injury_baseline = get_pre_injury_baseline(player_name, box_scores, injury_date_to_use)

            if pre_injury_baseline is not None:
                # Successfully calculated clean baseline
                baselines[player_name] = pre_injury_baseline
            else:
                # No pre-injury games - use standard baseline (handles rookies, recent trades)
                baselines[player_name] = get_player_baseline(player, injury_context, box_scores)

    # Apply weighting
    weighted_baselines = {}
    for player_name, baseline in baselines.items():
        base_weight = baseline + BENCH_OPPORTUNITY_CONSTANT
        player_position = teammates[teammates['PLAYER'] == player_name]['POSITION'].iloc[0]

        # Apply 2x multiplier for exact position (only if use_multiplier=True)
        if use_multiplier and player_position == injured_position:
            weighted_baselines[player_name] = base_weight * EXACT_POSITION_MULTIPLIER
        else:
            weighted_baselines[player_name] = base_weight

    total_weighted = sum(weighted_baselines.values())

    # Initial distribution
    projections = {}
    for player_name, weight in weighted_baselines.items():
        distribution_weight = weight / total_weighted
        boost = lost_minutes * distribution_weight
        projected = baselines[player_name] + boost
        projections[player_name] = projected

    # Handle overflow from 36-min cap
    for _ in range(7):
        overflow_minutes = 0
        capped_players = set()

        for player_name, projected in projections.items():
            if projected > MAX_MINUTES:
                overflow_minutes += (projected - MAX_MINUTES)
                projections[player_name] = MAX_MINUTES
                capped_players.add(player_name)

        if overflow_minutes == 0:
            break

        non_capped = [p for p in projections.keys() if p not in capped_players]
        if not non_capped:
            break

        non_capped_total = sum(weighted_baselines[p] for p in non_capped)
        for player_name in non_capped:
            weight = weighted_baselines[player_name] / non_capped_total
            additional_boost = overflow_minutes * weight
            projections[player_name] = min(projections[player_name] + additional_boost, MAX_MINUTES)

    # Cap individual boosts to 50% of lost minutes
    # Prevents one player from absorbing entire injury impact (unrealistic)
    # Example: If OG (30 MPG) is out, no single player should get more than 15 MPG boost
    # Allows multiple injuries to stack (realistic scenario)
    max_individual_boost = lost_minutes * 0.5

    for player_name in list(projections.keys()):
        boost = projections[player_name] - baselines[player_name]
        if boost > max_individual_boost:
            logger.info(f"{player_name}: Capping boost from {injured_player['PLAYER']} injury: {boost:.1f} -> {max_individual_boost:.1f} MPG")
            projections[player_name] = baselines[player_name] + max_individual_boost

    # Track beneficiaries in injury context (only if boost > 3 MPG)
    # Note: injury_date_to_use already calculated at top of function
    for player_name in projections.keys():
        boost = projections[player_name] - baselines[player_name]

        # Only track as BENEFICIARY if boost > 3 MPG
        # Prevents stars from being tracked for minimal boosts (e.g., +1 MPG after 36-min cap)
        if boost > 3:
            # Check if this beneficiary relationship already exists (prevent duplicates)
            existing_beneficiary = injury_context[
                (injury_context['PLAYER'] == player_name) &
                (injury_context['STATUS'] == 'BENEFICIARY') &
                (injury_context['BENEFICIARY_OF'] == injured_player['PLAYER'])
            ]

            if existing_beneficiary.empty:
                injury_context = add_to_injury_context(
                    injury_context,
                    player_name=player_name,
                    team=injured_player['TEAM'],
                    status='BENEFICIARY',
                    true_baseline=baselines[player_name],
                    beneficiary_of=injured_player['PLAYER'],
                    injury_date=injury_date_to_use
                )
                logger.info(f"{player_name}: Marked as BENEFICIARY (+{boost:.1f} MPG from {injured_player['PLAYER']} injury, baseline: {baselines[player_name]:.1f} -> projected: {projections[player_name]:.1f})")
            else:
                logger.debug(f"{player_name}: Already tracked as BENEFICIARY of {injured_player['PLAYER']} (current projection: {projections[player_name]:.1f} MPG)")
        else:
            logger.debug(f"{player_name}: Boost too small (+{boost:.1f} MPG), not tracking as BENEFICIARY")

    return projections, injury_context


# ==================== Shared Projection Logic ====================

def calculate_team_injury_redistributions(team_players, injury_context, today, box_scores,
                                          position_overlap_func, use_multiplier):
    """
    Calculate injury redistributions for ALL injured players on a team ONCE.
    This avoids redundant calculations when looping through each active player.

    Returns:
        tuple: (team_redistributions, updated_injury_context)
            - team_redistributions: dict of {injured_player_name: {player_name: projected_min}}
            - updated_injury_context: injury_context with beneficiaries tracked
    """
    team_redistributions = {}

    # Find teammates currently OUT (exclude those returning today - they're playing!)
    team_injuries = team_players[team_players['STATUS'] == 'OUT']
    if 'RETURN_DATE_DT' in team_injuries.columns:
        teammates_returning_today = team_injuries['RETURN_DATE_DT'] <= today
        team_injuries = team_injuries[~teammates_returning_today]
        excluded_count = teammates_returning_today.sum()
        if excluded_count > 0:
            logger.debug(f"Excluded {excluded_count} teammates marked OUT but returning today (won't redistribute their minutes)")

    if not team_injuries.empty:
        team_injuries_sorted = team_injuries.sort_values('SEASON_AVG_MIN', ascending=False)

        # Calculate redistribution for each injured player ONCE
        for _, injured in team_injuries_sorted.iterrows():
            injury_projections, injury_context = redistribute_injury_minutes(
                injured, team_players, injury_context, today, box_scores,
                position_overlap_func, use_multiplier=use_multiplier
            )
            if injury_projections:  # Only store if there are projections
                team_redistributions[injured['PLAYER']] = injury_projections

    return team_redistributions, injury_context
