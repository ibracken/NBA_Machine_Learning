"""
NBA Injury Scraper Lambda (PDF Version)
Scrapes injury data from NBA Official PDFs and saves to S3
Output: data/injuries/current.parquet
Columns: PLAYER, TEAM, STATUS, RETURN_DATE, RETURN_DATE_DT, ESTIMATED_INJURY_DATE
"""

import boto3
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pytz
from io import BytesIO
import logging
from unidecode import unidecode
import pdfplumber
import re
import os
import tempfile

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add console handler for local testing
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# S3 client
s3 = boto3.client('s3')
BUCKET_NAME = 'nba-prediction-ibracken'

# S3 utility functions
def save_dataframe_to_s3(df, key):
    """Save DataFrame as Parquet to S3"""
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=parquet_buffer.getvalue())
    logger.info(f"Saved {len(df)} records to s3://{BUCKET_NAME}/{key}")

def normalize_name(name):
    """Normalize player name for matching with other datasets"""
    # Convert "Last, First" to "first last"
    if ',' in name:
        parts = name.split(',')
        if len(parts) == 2:
            last = parts[0].strip()
            first = parts[1].strip()
            name = f"{first} {last}"

    # Normalize suffixes: ensure space before Jr./II/III/IV/Sr.
    # PDFs have "NanceJr." but box scores have "nance jr."
    name = re.sub(r'([a-z])(jr\.|ii|iii|iv|sr\.)', r'\1 \2', name, flags=re.IGNORECASE)

    return unidecode(name.strip().lower())

def get_latest_nba_injury_pdf_url():
    """
    Scrapes the NBA official injury report page to find the most recent PDF link.
    Returns the URL of the latest report.
    """
    page_url = "https://official.nba.com/nba-injury-report-2025-26-season/"

    try:
        logger.info(f"Fetching NBA official injury report page: {page_url}")
        response = requests.get(page_url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all PDF links
        pdf_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'Injury-Report_' in href and href.endswith('.pdf'):
                pdf_links.append(href)

        if not pdf_links:
            raise Exception("No injury report PDFs found on the page")

        # The links are in chronological order, so the last one is the most recent
        latest_url = pdf_links[-1]

        # Extract the time from the URL for display
        time_match = re.search(r'_(\d{2}[AP]M)\.pdf', latest_url)
        time_str = time_match.group(1) if time_match else "unknown time"

        logger.info(f"Found latest NBA official report: {time_str} - {latest_url}")
        return latest_url

    except Exception as e:
        logger.error(f"Error fetching NBA official injury report page: {e}")
        raise

def extract_team_abbr_from_matchup(matchup, team_name):
    """
    Extract team abbreviation from matchup code.
    matchup format: "AAA@BBB" where AAA is away team, BBB is home team
    team_name: full team name from PDF (e.g., "WashingtonWizards")

    Returns the 3-letter abbreviation by matching against matchup.
    """
    if not matchup or '@' not in matchup:
        return None

    parts = matchup.split('@')
    if len(parts) != 2:
        return None

    away_abbr, home_abbr = parts[0].strip(), parts[1].strip()

    # Try to match team name with abbreviation
    # Simple heuristic: check if team name starts with abbreviation letters
    team_upper = team_name.replace(' ', '').upper()

    # Check if away or home team
    if team_upper.startswith(away_abbr[:2]):
        return away_abbr
    elif team_upper.startswith(home_abbr[:2]):
        return home_abbr
    else:
        # Fallback: try matching first letters
        initials = ''.join([word[0] for word in re.findall(r'[A-Z][a-z]*', team_name)])
        if initials.startswith(away_abbr):
            return away_abbr
        elif initials.startswith(home_abbr):
            return home_abbr

    return away_abbr if team_upper.startswith(away_abbr[0]) else home_abbr

def parse_nba_injury_pdf(pdf_path):
    """
    Parse NBA official injury report PDF to extract injury data.
    Returns: DataFrame with columns [PLAYER, TEAM, STATUS, REASON]
    """
    logger.info(f"Parsing NBA official injury PDF: {pdf_path}")

    data = []

    # These variables hold the value of merged cells as we read down the page
    current_date = None
    current_time = None
    current_matchup = None
    current_team = None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract raw text since table detection is failing
                text = page.extract_text()
                if not text:
                    continue

                lines = text.split('\n')

                for line in lines:
                    line = line.strip()

                    # Skip empty lines, headers, and page numbers
                    if not line or 'Injury Report:' in line or 'Page' in line:
                        continue
                    if 'GameDate' in line or 'Game Date' in line:
                        continue

                    # Pattern 1: Full line with date, time, matchup, team, player, status, reason (optional)
                    full_match = re.match(
                        r'^(\d{1,2}/\d{1,2}/\d{4})\s+(\d{2}:\d{2}\s*\([A-Z]+\))\s+([A-Z]{3}@[A-Z]{3})\s+(.+?)\s+([A-Z][a-zA-Z\'\.]+(?:\s*(?:Jr\.|II|III|IV|Sr\.))?,\s*[A-Z][a-zA-Z\']+)\s+(Out|Questionable|Probable|Available)(?:\s+(.+))?$',
                        line
                    )
                    if full_match:
                        current_date = full_match.group(1)
                        current_time = full_match.group(2)
                        current_matchup = full_match.group(3)
                        current_team = full_match.group(4).strip()
                        player = full_match.group(5).strip()
                        status = full_match.group(6).strip()
                        reason = full_match.group(7).strip() if full_match.group(7) else ""

                        # Only include if status is "Out" and NOT G League related
                        if status == "Out" and "GLeague" not in reason and "G League" not in reason:
                            team_abbr = extract_team_abbr_from_matchup(current_matchup, current_team)
                            data.append({
                                "PLAYER": normalize_name(player),
                                "TEAM": team_abbr,
                                "STATUS": "OUT",
                                "RETURN_DATE": None  # No return date in PDF
                            })
                        continue

                    # Pattern 2: Time, matchup, team, player, status, reason (optional, same date as previous)
                    time_match = re.match(
                        r'^(\d{2}:\d{2}\s*\([A-Z]+\))\s+([A-Z]{3}@[A-Z]{3})\s+(.+?)\s+([A-Z][a-zA-Z\'\.]+(?:\s*(?:Jr\.|II|III|IV|Sr\.))?,\s*[A-Z][a-zA-Z\']+)\s+(Out|Questionable|Probable|Available)(?:\s+(.+))?$',
                        line
                    )
                    if time_match:
                        current_time = time_match.group(1)
                        current_matchup = time_match.group(2)
                        current_team = time_match.group(3).strip()
                        player = time_match.group(4).strip()
                        status = time_match.group(5).strip()
                        reason = time_match.group(6).strip() if time_match.group(6) else ""

                        # Only include if status is "Out" and NOT G League related
                        if status == "Out" and "GLeague" not in reason and "G League" not in reason:
                            team_abbr = extract_team_abbr_from_matchup(current_matchup, current_team)
                            data.append({
                                "PLAYER": normalize_name(player),
                                "TEAM": team_abbr,
                                "STATUS": "OUT",
                                "RETURN_DATE": None
                            })
                        continue

                    # Pattern 2.5: Matchup, team, player, status, reason (optional, same time/date as previous)
                    matchup_team_match = re.match(
                        r'^([A-Z]{3}@[A-Z]{3})\s+(.+?)\s+([A-Z][a-zA-Z\'\.]+(?:\s*(?:Jr\.|II|III|IV|Sr\.))?,\s*[A-Z][a-zA-Z\']+)\s+(Out|Questionable|Probable|Available)(?:\s+(.+))?$',
                        line
                    )
                    if matchup_team_match:
                        current_matchup = matchup_team_match.group(1)
                        current_team = matchup_team_match.group(2).strip()
                        player = matchup_team_match.group(3).strip()
                        status = matchup_team_match.group(4).strip()
                        reason = matchup_team_match.group(5).strip() if matchup_team_match.group(5) else ""

                        # Only include if status is "Out" and NOT G League related
                        if status == "Out" and "GLeague" not in reason and "G League" not in reason:
                            team_abbr = extract_team_abbr_from_matchup(current_matchup, current_team)
                            data.append({
                                "PLAYER": normalize_name(player),
                                "TEAM": team_abbr,
                                "STATUS": "OUT",
                                "RETURN_DATE": None
                            })
                        continue

                    # Pattern 3: Team, player, status, reason (optional, same matchup)
                    team_match = re.match(
                        r'^([A-Z][a-zA-Z\s]+?)\s+([A-Z][a-zA-Z\'\.]+(?:\s*(?:Jr\.|II|III|IV|Sr\.))?,\s*[A-Z][a-zA-Z\']+)\s+(Out|Questionable|Probable|Available)(?:\s+(.+))?$',
                        line
                    )
                    if team_match:
                        current_team = team_match.group(1).strip()
                        player = team_match.group(2).strip()
                        status = team_match.group(3).strip()
                        reason = team_match.group(4).strip() if team_match.group(4) else ""

                        # Only include if status is "Out" and NOT G League related
                        if status == "Out" and "GLeague" not in reason and "G League" not in reason:
                            team_abbr = extract_team_abbr_from_matchup(current_matchup, current_team)
                            data.append({
                                "PLAYER": normalize_name(player),
                                "TEAM": team_abbr,
                                "STATUS": "OUT",
                                "RETURN_DATE": None
                            })
                        continue

                    # Pattern 4: Player, status, reason (optional, same team)
                    player_match = re.match(
                        r'^([A-Z][a-zA-Z\'\.]+(?:\s*(?:Jr\.|II|III|IV|Sr\.))?,\s*[A-Z][a-zA-Z\']+)\s+(Out|Questionable|Probable|Available)(?:\s+(.+))?$',
                        line
                    )
                    if player_match:
                        player = player_match.group(1).strip()
                        status = player_match.group(2).strip()
                        reason = player_match.group(3).strip() if player_match.group(3) else ""

                        # Only include if status is "Out" and NOT G League related
                        if status == "Out" and "GLeague" not in reason and "G League" not in reason:
                            team_abbr = extract_team_abbr_from_matchup(current_matchup, current_team)
                            data.append({
                                "PLAYER": normalize_name(player),
                                "TEAM": team_abbr,
                                "STATUS": "OUT",
                                "RETURN_DATE": None
                            })
                        continue

                    # Pattern 5: "NOT YET SUBMITTED" cases - skip these
                    if 'NOT YET SUBMITTED' in line:
                        continue

        df = pd.DataFrame(data)
        logger.info(f"Parsed {len(df)} OUT injuries from NBA official PDF (excluding G League)")

        return df

    except Exception as e:
        logger.error(f"Error parsing NBA official PDF: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def scrape_nba_official_injuries():
    """
    Scrape injury data from NBA official website PDF
    Returns: DataFrame with columns [PLAYER, TEAM, STATUS, RETURN_DATE]
    """
    logger.info("Starting NBA official injury scraping")

    try:
        # Get latest PDF URL
        pdf_url = get_latest_nba_injury_pdf_url()

        # Download PDF to temp directory (works on both Windows and Lambda)
        # On Lambda: /tmp, on Windows: C:\Users\...\AppData\Local\Temp
        pdf_filename = pdf_url.split('/')[-1]
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, pdf_filename)

        logger.info(f"Downloading PDF to: {pdf_path}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        with open(pdf_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"PDF downloaded successfully ({len(response.content) / 1024:.2f} KB)")

        # Parse PDF
        df = parse_nba_injury_pdf(pdf_path)

        # Clean up temp file
        try:
            os.remove(pdf_path)
            logger.info(f"Cleaned up temp file: {pdf_path}")
        except Exception as e:
            logger.warning(f"Could not remove temp file: {e}")

        return df

    except Exception as e:
        logger.error(f"Error scraping NBA official injuries: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def estimate_injury_dates(df_injuries):
    """
    Estimate when injuries started by finding last game date for each player
    - Load current season box scores from S3
    - Fallback to previous season for season-long injuries (0 games this season)
    - Find last game date for each injured player
    - Set ESTIMATED_INJURY_DATE = last_game_date + 1 day for OUT players
    """
    logger.info("Estimating injury dates from box scores")

    try:
        # Load current season box scores from S3
        try:
            response = s3.get_object(Bucket=BUCKET_NAME, Key='data/box_scores/current.parquet')
            box_scores = pd.read_parquet(BytesIO(response['Body'].read()))
            logger.info(f"Loaded {len(box_scores)} current season box score records")
        except Exception as e:
            logger.warning(f"Could not load current season box scores: {e}. Setting ESTIMATED_INJURY_DATE to None")
            df_injuries['ESTIMATED_INJURY_DATE'] = None
            return df_injuries

        # Normalize player names in box scores
        box_scores['PLAYER'] = box_scores['PLAYER'].apply(normalize_name)

        # Convert GAME_DATE to datetime if needed
        if 'GAME_DATE' in box_scores.columns:
            box_scores['GAME_DATE'] = pd.to_datetime(box_scores['GAME_DATE'])
        else:
            logger.warning("No GAME_DATE column in box scores")
            df_injuries['ESTIMATED_INJURY_DATE'] = None
            return df_injuries

        # Get last game date for each player from current season
        last_games_current = box_scores.groupby('PLAYER')['GAME_DATE'].max().reset_index()
        last_games_current.columns = ['PLAYER', 'LAST_GAME_DATE']

        # Merge with injuries
        df_injuries = df_injuries.merge(last_games_current, on='PLAYER', how='left')

        # For players not found in current season (season-long injuries), fallback to previous season
        players_without_dates = df_injuries[df_injuries['LAST_GAME_DATE'].isna()]['PLAYER'].unique()

        if len(players_without_dates) > 0:
            logger.info(f"Found {len(players_without_dates)} players with no current season games - checking previous season")

            try:
                # Load previous season box scores
                response = s3.get_object(Bucket=BUCKET_NAME, Key='data/box_scores/2024-25.parquet')
                prev_box_scores = pd.read_parquet(BytesIO(response['Body'].read()))
                logger.info(f"Loaded {len(prev_box_scores)} previous season box score records")

                # Normalize player names
                prev_box_scores['PLAYER'] = prev_box_scores['PLAYER'].apply(normalize_name)

                # Convert GAME_DATE to datetime
                if 'GAME_DATE' in prev_box_scores.columns:
                    prev_box_scores['GAME_DATE'] = pd.to_datetime(prev_box_scores['GAME_DATE'])

                    # Get last game date for season-long injured players from previous season
                    prev_season_players = prev_box_scores[prev_box_scores['PLAYER'].isin(players_without_dates)]
                    last_games_prev = prev_season_players.groupby('PLAYER')['GAME_DATE'].max().reset_index()
                    last_games_prev.columns = ['PLAYER', 'LAST_GAME_DATE_PREV']

                    # Update LAST_GAME_DATE for players found in previous season
                    for idx, row in df_injuries[df_injuries['LAST_GAME_DATE'].isna()].iterrows():
                        player_name = row['PLAYER']
                        prev_game = last_games_prev[last_games_prev['PLAYER'] == player_name]

                        if not prev_game.empty:
                            df_injuries.loc[idx, 'LAST_GAME_DATE'] = prev_game['LAST_GAME_DATE_PREV'].iloc[0]
                            logger.info(f"{player_name}: Using previous season last game date (season-long injury)")

            except s3.exceptions.NoSuchKey:
                logger.warning("Previous season box scores not found in S3 (data/box_scores/2024-25.parquet)")
            except Exception as e:
                logger.warning(f"Could not load previous season box scores: {e}")

        # Filter out players who haven't played in either current or previous season
        # These are likely G-League/rookie players with no NBA games - their injuries don't affect minutes
        players_still_without_dates = df_injuries[df_injuries['LAST_GAME_DATE'].isna()]['PLAYER'].tolist()
        if len(players_still_without_dates) > 0:
            logger.info(f"Excluding {len(players_still_without_dates)} players with no NBA games (G-League/rookies): {players_still_without_dates}")
            df_injuries = df_injuries[df_injuries['LAST_GAME_DATE'].notna()].copy()

        # For all injured players, estimate injury date as last_game_date + 1 day
        def calculate_injury_date(row):
            if pd.notna(row.get('LAST_GAME_DATE')):
                last_game = pd.to_datetime(row['LAST_GAME_DATE'])
                return (last_game + pd.Timedelta(days=1)).date()
            return None

        df_injuries['ESTIMATED_INJURY_DATE'] = df_injuries.apply(calculate_injury_date, axis=1)

        # Drop temporary column
        df_injuries = df_injuries.drop(columns=['LAST_GAME_DATE'])

        # Log summary
        with_dates = df_injuries['ESTIMATED_INJURY_DATE'].notna().sum()
        without_dates = df_injuries['ESTIMATED_INJURY_DATE'].isna().sum()
        logger.info(f"Estimated injury dates for {with_dates}/{len(df_injuries)} injured players ({without_dates} without dates)")

        return df_injuries

    except Exception as e:
        logger.warning(f"Error estimating injury dates: {e}")
        df_injuries['ESTIMATED_INJURY_DATE'] = None
        return df_injuries

def run_injury_scraper():
    """Main function to run injury scraping pipeline"""
    logger.info("Starting injury scraper (PDF version)")

    try:
        # Scrape injuries from NBA official PDF
        df_injuries = scrape_nba_official_injuries()

        if df_injuries.empty:
            logger.warning("No injury data scraped from PDF")
            return {
                'success': False,
                'error': 'No injury data scraped from PDF'
            }

        # Add RETURN_DATE_DT column (set to None since we don't have return dates)
        df_injuries['RETURN_DATE_DT'] = None

        # Estimate injury start dates from box scores
        df_injuries = estimate_injury_dates(df_injuries)

        # Save to S3
        save_dataframe_to_s3(df_injuries, 'data/injuries/current.parquet')

        logger.info("Injury scraper (PDF version) completed successfully")

        return {
            'success': True,
            'total_injuries': len(df_injuries),
            'status_breakdown': df_injuries['STATUS'].value_counts().to_dict()
        }

    except Exception as e:
        logger.error(f"Error in injury scraper: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    logger.info("Injury scraper Lambda function started (PDF version)")

    try:
        result = run_injury_scraper()

        if result['success']:
            return {
                'statusCode': 200,
                'body': {
                    'message': 'Injury scraping (PDF) completed successfully',
                    'total_injuries': result['total_injuries'],
                    'status_breakdown': result.get('status_breakdown', {})
                }
            }
        else:
            return {
                'statusCode': 500,
                'body': {
                    'message': 'Injury scraping (PDF) failed',
                    'error': result['error']
                }
            }

    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'message': 'Lambda execution failed',
                'error': str(e)
            }
        }

# For local testing
if __name__ == "__main__":
    result = run_injury_scraper()
    print(f"Injury scraper result: {result}")
