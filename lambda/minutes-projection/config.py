"""
Configuration and constants for minutes projection lambda
"""

import boto3

# S3 client
s3_client = boto3.client('s3')
# === NEW: Add SNS Client ===
sns_client = boto3.client('sns')
SNS_TOPIC_ARN = 'arn:aws:sns:us-east-1:349928386418:lineup-optimizer-notifications'
BUCKET_NAME = 'nba-prediction-ibracken'

# Algorithm constants
BENCH_OPPORTUNITY_CONSTANT = 0.1  # Small boost for deep bench players
EXACT_POSITION_MULTIPLIER = 2.0  # Direct backups get 2x weight (complex model only)
MAX_MINUTES = 37  # Cap individual player minutes
CONFIDENCE_CLEARANCE_GAMES = 3  # Games needed to clear LOW confidence

# Injury redistribution is directionally right but ~3x oversized: beneficiaries historically
# realized only 30-50% of the boost handed to them, at every boost size. Blending the
# redistributed projection back toward the no-injury projection removes the resulting
# over-projection bias. 0.35 was fit on Nov-Dec 2025 and validated on Dec 26-Jan 19.
INJURY_ADJUSTMENT_WEIGHT = 0.35

# An EX_BENEFICIARY record permanently excludes a date range from a player's baseline, so it is
# only worth writing when the player was genuinely inflated during it. Measured over 1,871
# historical windows, 44% showed no effect and 13% covered games where the player played LESS -
# excluding those biased baselines in both directions. Require this much lift (MPG inside the
# window minus outside) before recording one.
EX_BENEFICIARY_MIN_LIFT = 3.0

# FP_PER_MIN uses the career rate until a player has more than this many games this season, then
# the season rate. Must match supervised-learning, which uses the career rate for a player's first
# 2 games of a season (SEASON_GAME_NUM <= 2, i.e. at most 1 prior game).
CAREER_RATE_MAX_PRIOR_GAMES = 1

# Early-season pause: in-house projections (and their lineups) are skipped until every NBA team has
# played this many games this season. Before that, most players have no or <4 current-season games,
# so they are dropped or fall to the 10 MPG fallback, and offseason movers sit on their old team.
# 4 matches the "reliable baseline" cutoff in projection_models/injury_system. The DFF lineup still runs.
# Mirrored in llm-analyst/config.py - keep in sync.
PROJECTION_START_MIN_TEAM_GAMES = 4
NBA_TEAM_COUNT = 30
