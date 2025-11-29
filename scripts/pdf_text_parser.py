import pdfplumber
import pandas as pd
import re

def parse_nba_injury_report(pdf_path):
    data = []

    # These variables hold the value of merged cells as we read down the page
    current_date = None
    current_time = None
    current_matchup = None
    current_team = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract raw text since table detection is failing
            text = page.extract_text()
            lines = text.split('\n')

            for line in lines:
                line = line.strip()

                # Skip empty lines, headers, and page numbers
                if not line or 'Injury Report:' in line or 'Page' in line:
                    continue
                if 'GameDate' in line or 'Game Date' in line:
                    continue

                # Pattern 1: Full line with date, time, matchup, team, player, status, reason
                # Example: "11/29/2025 05:00 (ET) BOS@MIN Boston Celtics Brown, Jaylen Questionable Injury/Illness - Low Back; Spasms"
                full_match = re.match(
                    r'^(\d{1,2}/\d{1,2}/\d{4})\s+(\d{2}:\d{2}\s*\([A-Z]+\))\s+([A-Z]{3}@[A-Z]{3})\s+(.+?)\s+([A-Z][a-zA-Z\']+(?:\s+(?:Jr\.|II|III|IV|Sr\.))?,\s*[A-Z][a-zA-Z]+)\s+(Out|Questionable|Probable|Available)\s+(.+)$',
                    line
                )
                if full_match:
                    current_date = full_match.group(1)
                    current_time = full_match.group(2)
                    current_matchup = full_match.group(3)
                    current_team = full_match.group(4).strip()
                    player = full_match.group(5).strip()
                    status = full_match.group(6).strip()
                    reason = full_match.group(7).strip()

                    data.append({
                        "Game Date": current_date,
                        "Game Time": current_time,
                        "Matchup": current_matchup,
                        "Team": current_team,
                        "Player Name": player,
                        "Current Status": status,
                        "Reason": reason
                    })
                    continue

                # Pattern 2: Time, matchup, team, player, status, reason (same date as previous)
                time_match = re.match(
                    r'^(\d{2}:\d{2}\s*\([A-Z]+\))\s+([A-Z]{3}@[A-Z]{3})\s+(.+?)\s+([A-Z][a-zA-Z\']+(?:\s+(?:Jr\.|II|III|IV|Sr\.))?,\s*[A-Z][a-zA-Z]+)\s+(Out|Questionable|Probable|Available)\s+(.+)$',
                    line
                )
                if time_match:
                    current_time = time_match.group(1)
                    current_matchup = time_match.group(2)
                    current_team = time_match.group(3).strip()
                    player = time_match.group(4).strip()
                    status = time_match.group(5).strip()
                    reason = time_match.group(6).strip()

                    data.append({
                        "Game Date": current_date,
                        "Game Time": current_time,
                        "Matchup": current_matchup,
                        "Team": current_team,
                        "Player Name": player,
                        "Current Status": status,
                        "Reason": reason
                    })
                    continue

                # Pattern 3: Team, player, status, reason (same matchup)
                team_match = re.match(
                    r'^([A-Z][a-zA-Z\s]+?)\s+([A-Z][a-zA-Z\']+(?:\s+(?:Jr\.|II|III|IV|Sr\.))?,\s*[A-Z][a-zA-Z]+)\s+(Out|Questionable|Probable|Available)\s+(.+)$',
                    line
                )
                if team_match:
                    current_team = team_match.group(1).strip()
                    player = team_match.group(2).strip()
                    status = team_match.group(3).strip()
                    reason = team_match.group(4).strip()

                    data.append({
                        "Game Date": current_date,
                        "Game Time": current_time,
                        "Matchup": current_matchup,
                        "Team": current_team,
                        "Player Name": player,
                        "Current Status": status,
                        "Reason": reason
                    })
                    continue

                # Pattern 4: Player, status, reason (same team)
                player_match = re.match(
                    r'^([A-Z][a-zA-Z\']+(?:\s+(?:Jr\.|II|III|IV|Sr\.))?,\s*[A-Z][a-zA-Z]+)\s+(Out|Questionable|Probable|Available)\s+(.+)$',
                    line
                )
                if player_match:
                    player = player_match.group(1).strip()
                    status = player_match.group(2).strip()
                    reason = player_match.group(3).strip()

                    data.append({
                        "Game Date": current_date,
                        "Game Time": current_time,
                        "Matchup": current_matchup,
                        "Team": current_team,
                        "Player Name": player,
                        "Current Status": status,
                        "Reason": reason
                    })
                    continue

                # Pattern 5: "NOT YET SUBMITTED" cases
                if 'NOT YET SUBMITTED' in line:
                    # Check if there's a team name before NOT YET SUBMITTED
                    team_not_submitted = re.match(r'^(.+?)\s+NOT YET SUBMITTED', line)
                    if team_not_submitted:
                        current_team = team_not_submitted.group(1).strip()

                    data.append({
                        "Game Date": current_date,
                        "Game Time": current_time,
                        "Matchup": current_matchup,
                        "Team": current_team,
                        "Player Name": "N/A",
                        "Current Status": "NOT YET SUBMITTED",
                        "Reason": "Report not submitted"
                    })
                    continue

    df = pd.DataFrame(data)
    return df

# Usage
pdf_filename = "scripts/Injury-Report_2025-11-29_02PM.pdf"

try:
    df_injuries = parse_nba_injury_report(pdf_filename)

    print(f"Successfully parsed {len(df_injuries)} injury records")
    print("\nFirst 20 rows:")
    print(df_injuries.head(20).to_string())

    # Save to CSV
    df_injuries.to_csv("parsed_injuries.csv", index=False)
    print("\nSuccessfully saved to parsed_injuries.csv")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
