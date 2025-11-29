"""
NBA Injury Scraper Lambda
Scrapes injury data from ESPN and saves to S3
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
    return unidecode(name.strip().lower())

# NBA Teams mapping: full name -> abbreviation
# Team name is extracted from each div at: div/div[1]/div/span
TEAM_NAME_TO_ABBR = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}

def scrape_espn_injuries():
    """
    Scrape injury data from ESPN using XPath structure
    Returns: DataFrame with columns [PLAYER, TEAM, STATUS, RETURN_DATE]
    """
    logger.info("Starting ESPN injury scraping")

    url = "https://www.espn.com/nba/injuries"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')

        injuries = []

        # Base XPath: /html/body/div[1]/div/div/div/div/main/div[2]/div[2]/div/div/section/div/section
        # Convert XPath to CSS-like navigation
        base_section = soup.find('body').find('div').find('div').find('div').find('div').find('main')
        if not base_section:
            logger.error("Could not navigate to main section")
            return pd.DataFrame()

        # Navigate to div[2]/div[2]/div/div/section/div/section
        divs = base_section.find_all('div', recursive=False)
        if len(divs) < 2:
            logger.error("Could not find div[2] under main")
            return pd.DataFrame()

        section_parent = divs[1]  # div[2]
        inner_divs = section_parent.find_all('div', recursive=False)
        if len(inner_divs) < 2:
            logger.error("Could not find div[2]/div[2]")
            return pd.DataFrame()

        # Navigate deeper to find the section container
        section_container = inner_divs[1]  # div[2]
        # Continue navigation: div/div/section/div/section
        for _ in range(2):  # div/div
            section_container = section_container.find('div', recursive=False)
            if not section_container:
                logger.error("Failed navigation to section container")
                return pd.DataFrame()

        section = section_container.find('section', recursive=False)
        if not section:
            logger.error("Could not find section element")
            return pd.DataFrame()

        inner_section_div = section.find('div', recursive=False)
        if not inner_section_div:
            logger.error("Could not find div inside section")
            return pd.DataFrame()

        final_section = inner_section_div.find('section', recursive=False)
        if not final_section:
            logger.error("Could not find final section element")
            return pd.DataFrame()

        # Now get all team divs
        team_divs = final_section.find_all('div', recursive=False)
        logger.info(f"Found {len(team_divs)} team divs")

        # Iterate through all team divs and extract team name dynamically
        for idx, team_div in enumerate(team_divs):
            team_name = None  # Initialize for exception handling
            try:
                # Extract team name from div[1]/div/span
                inner_divs = team_div.find_all('div', recursive=False)
                if len(inner_divs) < 1:
                    logger.warning(f"Div {idx}: Could not find div[1] for team name")
                    continue

                first_div = inner_divs[0]  # div[1]
                nested_div = first_div.find('div', recursive=False)
                if not nested_div:
                    logger.warning(f"Div {idx}: Could not find nested div for team name")
                    continue

                team_name_span = nested_div.find('span', recursive=False)
                if not team_name_span:
                    logger.warning(f"Div {idx}: Could not find span with team name")
                    continue

                team_name = team_name_span.get_text(strip=True)

                # Look up team abbreviation
                team_abbr = TEAM_NAME_TO_ABBR.get(team_name)
                if not team_abbr:
                    logger.warning(f"Div {idx}: Unknown team name '{team_name}' - skipping")
                    continue

                logger.info(f"Processing {team_name} ({team_abbr}) - div[{idx}]")

                # Navigate to table: div[2]/div/div[2]/table/tbody
                # XPath structure: team_div/div[2]/div/div[2]/table/tbody
                # Re-get inner_divs to access div[2]
                team_inner_divs = team_div.find_all('div', recursive=False)
                if len(team_inner_divs) < 2:
                    logger.info(f"{team_name}: No div[2] found (team has no injuries)")
                    continue

                second_div = team_inner_divs[1]  # div[2]

                # Navigate: div/div[2]
                nested_div = second_div.find('div', recursive=False)
                if not nested_div:
                    logger.info(f"{team_name}: No nested div found (team has no injuries)")
                    continue

                nested_divs = nested_div.find_all('div', recursive=False)
                if len(nested_divs) < 2:
                    logger.info(f"{team_name}: No div[2] in nested structure (team has no injuries)")
                    continue

                table_container = nested_divs[1]  # div[2]

                # Find table/tbody
                table = table_container.find('table')
                if not table:
                    logger.info(f"{team_name}: No injury table found (team has no injuries)")
                    continue

                tbody = table.find('tbody')
                if not tbody:
                    logger.info(f"{team_name}: No tbody found in table (team has no injuries)")
                    continue

                # Find all player rows
                player_rows = tbody.find_all('tr', recursive=False)
                logger.info(f"{team_name}: Found {len(player_rows)} injured players")

                for row in player_rows:
                    try:
                        cells = row.find_all('td', recursive=False)
                        if len(cells) < 4:
                            logger.warning(f"{team_name}: Row has insufficient cells ({len(cells)})")
                            continue

                        # Player name: td[1]/a
                        name_cell = cells[0]  # td[1] (0-indexed)
                        name_link = name_cell.find('a')
                        if name_link:
                            player_name = name_link.get_text(strip=True)
                        else:
                            player_name = name_cell.get_text(strip=True)

                        if not player_name:
                            logger.warning(f"{team_name}: Empty player name, skipping row")
                            continue

                        # Return date: td[3]
                        return_date_cell = cells[2]  # td[3] (0-indexed)
                        return_date_raw = return_date_cell.get_text(strip=True)
                        return_date = return_date_raw if return_date_raw else None

                        # Status: td[4]/span
                        status_cell = cells[3]  # td[4] (0-indexed)
                        status_span = status_cell.find('span')
                        if status_span:
                            status = status_span.get_text(strip=True)
                        else:
                            status = status_cell.get_text(strip=True)

                        if not status:
                            status = 'OUT'  # Default if missing

                        # Normalize status
                        status_upper = status.upper()
                        if 'OUT' in status_upper:
                            status_normalized = 'OUT'
                        elif 'QUESTIONABLE' in status_upper:
                            status_normalized = 'QUESTIONABLE'
                        elif 'DOUBTFUL' in status_upper:
                            status_normalized = 'DOUBTFUL'
                        elif 'PROBABLE' in status_upper:
                            status_normalized = 'PROBABLE'
                        elif 'DAY-TO-DAY' in status_upper or 'DTD' in status_upper:
                            status_normalized = 'DAY-TO-DAY'
                        else:
                            status_normalized = status_upper

                        injuries.append({
                            'PLAYER': normalize_name(player_name),
                            'TEAM': team_abbr,
                            'STATUS': status_normalized,
                            'RETURN_DATE': return_date
                        })

                        logger.debug(f"Found injury: {player_name} ({team_abbr}) - {status_normalized} - Return: {return_date}")

                    except Exception as e:
                        logger.warning(f"Error parsing player row in {team_name}: {e}")
                        continue

            except Exception as e:
                team_identifier = team_name if team_name else f"div[{idx}]"
                logger.warning(f"Error processing team {team_identifier}: {e}")
                continue

        df = pd.DataFrame(injuries)
        logger.info(f"Successfully scraped {len(df)} injury records")

        if not df.empty:
            # Log summary by status
            status_counts = df['STATUS'].value_counts()
            logger.info(f"Injury status breakdown: {status_counts.to_dict()}")

        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch ESPN injury page: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error scraping ESPN injuries: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def process_return_dates(df_injuries):
    """
    Process and filter injury return dates
    - Parse return dates to datetime
    - Filter out stale injuries (return date <= today)
    """
    logger.info("Processing return dates")

    try:
        eastern = pytz.timezone('US/Eastern')
        today = datetime.now(eastern).date()

        # Parse return dates
        if 'RETURN_DATE' in df_injuries.columns:
            # Try to parse various date formats from ESPN
            def parse_return_date(date_str):
                if not date_str or pd.isna(date_str):
                    return None

                date_str = str(date_str).strip()

                # Common ESPN formats: "Nov 24", "Expected Nov 24", "2025-11-24", etc.
                try:
                    # Try parsing "Nov 24" format
                    parsed = pd.to_datetime(date_str, format='%b %d', errors='coerce')
                    if pd.notna(parsed):
                        # Add current year
                        parsed = parsed.replace(year=today.year)
                        parsed_date = parsed.date()

                        # If the date is more than 30 days in the past, assume it's referring to next year
                        # (e.g., "April 1" in November means next April, not last April)
                        # But if it's only a few days past, it's truly stale data
                        days_in_past = (today - parsed_date).days
                        if days_in_past > 30:
                            parsed = parsed.replace(year=today.year + 1)
                            parsed_date = parsed.date()

                        return parsed_date
                except:
                    pass

                try:
                    # Try standard datetime parsing
                    parsed = pd.to_datetime(date_str, errors='coerce')
                    if pd.notna(parsed):
                        return parsed.date()
                except:
                    pass

                return None

            df_injuries['RETURN_DATE_PARSED'] = df_injuries['RETURN_DATE'].apply(parse_return_date)

            # Filter out stale injuries (return date < today)
            # Keep players returning TODAY (they're still on injury report, might play limited minutes)
            stale_mask = (df_injuries['RETURN_DATE_PARSED'].notna()) & (df_injuries['RETURN_DATE_PARSED'] < today)
            stale_count = stale_mask.sum()

            if stale_count > 0:
                logger.info(f"Filtering out {stale_count} stale injury reports (return date < today)")
                df_injuries = df_injuries[~stale_mask].copy()

            # Log players returning today (will get 25% minutes reduction in projection)
            returning_today_mask = (df_injuries['RETURN_DATE_PARSED'].notna()) & (df_injuries['RETURN_DATE_PARSED'] == today)
            returning_today_count = returning_today_mask.sum()
            if returning_today_count > 0:
                logger.info(f"Keeping {returning_today_count} players returning today (will apply 25% minutes reduction)")

            # Keep the parsed date for minutes-projection use
            df_injuries = df_injuries.rename(columns={'RETURN_DATE_PARSED': 'RETURN_DATE_DT'})
        else:
            # No RETURN_DATE column, add empty column
            df_injuries['RETURN_DATE_DT'] = None

        return df_injuries

    except Exception as e:
        logger.warning(f"Error processing return dates: {e}")
        return df_injuries

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
        # This includes OUT, QUESTIONABLE, DAY-TO-DAY, etc. since status can change during recovery
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
    logger.info("Starting injury scraper")

    try:
        # Scrape injuries from ESPN
        df_injuries = scrape_espn_injuries()

        if df_injuries.empty:
            logger.warning("No injury data scraped")
            return {
                'success': False,
                'error': 'No injury data scraped'
            }

        # Process and filter return dates
        df_injuries = process_return_dates(df_injuries)

        if df_injuries.empty:
            logger.error("No injuries remaining after filtering stale reports - this is unusual, there should always be injuries")
            return {
                'success': False,
                'error': 'No injuries remaining after filtering. Scraping may have failed or all injuries filtered as stale.'
            }

        # Estimate injury start dates from box scores
        df_injuries = estimate_injury_dates(df_injuries)

        # Save to S3
        save_dataframe_to_s3(df_injuries, 'data/injuries/current.parquet')

        logger.info("Injury scraper completed successfully")

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
    logger.info("Injury scraper Lambda function started")

    try:
        result = run_injury_scraper()

        if result['success']:
            return {
                'statusCode': 200,
                'body': {
                    'message': 'Injury scraping completed successfully',
                    'total_injuries': result['total_injuries'],
                    'status_breakdown': result.get('status_breakdown', {})
                }
            }
        else:
            return {
                'statusCode': 500,
                'body': {
                    'message': 'Injury scraping failed',
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
