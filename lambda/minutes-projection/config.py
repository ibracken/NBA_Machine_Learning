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

# MAX_MINUTES and CAREER_RATE_MAX_PRIOR_GAMES serve the barebones FP model (lineup_optimizer), which is kept
# but switched off. The complex model's constants were removed with it (2026-10-18).
MAX_MINUTES = 37  # Cap individual player minutes

# FP_PER_MIN uses the career rate until a player has more than this many games this season, then
# the season rate. Must match supervised-learning, which uses the career rate for a player's first
# 2 games of a season (SEASON_GAME_NUM <= 2, i.e. at most 1 prior game).
CAREER_RATE_MAX_PRIOR_GAMES = 1

# Early-season pause: in-house projections (and their lineups) are skipped until every NBA team has
# played this many games this season. Before that, most players have no or <4 current-season games,
# so they are dropped or fall to the 10 MPG fallback, and offseason movers sit on their old team.
# 4 matches Formula C's "reliable baseline" cutoff. The DFF lineup still runs.
# Mirrored in llm-analyst/config.py - keep in sync.
PROJECTION_START_MIN_TEAM_GAMES = 1   # 4 until 2026-10-25 (Ian): the early-season blend
NBA_TEAM_COUNT = 30
