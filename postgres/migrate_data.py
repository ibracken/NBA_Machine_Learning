import pandas as pd
from sqlalchemy.orm import Session
from postgres.config import SessionLocal
from postgres.models import AdvancedPlayerStats, ClusteredPlayers, BoxScores, TestPlayerPredictions, DailyPlayerPredictions
from unidecode import unidecode
import math



def safe_float(value):
    try:
        return float(value)
    except ValueError:
        return None


def normalize_name(name):
    return unidecode(name.strip().lower())

# # Load the data
# data = pd.read_excel("data/NBAStats.xlsx")

# # Create a new session
session = SessionLocal()

# # Insert data into the database
# for _, row in data.iterrows():
#     normalized_name = normalize_name(row["PLAYER"])
#     existing_player = session.query(AdvancedPlayerStats).filter_by(PLAYER=normalized_name).first()
#     if existing_player:
#         existing_player.TEAM=row["TEAM"],
#         existing_player.AGE=row["AGE"],
#         existing_player.GP=row["GP"],
#         existing_player.W=row["W"],
#         existing_player.L=row["L"],
#         existing_player.MIN=row["MIN"],
#         existing_player.OFFRTG=row["OFFRTG"],
#         existing_player.DEFRTG=row["DEFRTG"],
#         existing_player.NETRTG=row["NETRTG"],
#         existing_player.AST_PERCENT=row["AST%"],
#         existing_player.AST_TO=row["AST/TO"],
#         existing_player.AST_RATIO=row["AST RATIO"],
#         existing_player.OREB_PERCENT=row["OREB%"],
#         existing_player.DREB_PERCENT=row["DREB%"],
#         existing_player.REB_PERCENT=row["REB%"],
#         existing_player.TO_RATIO=row["TO RATIO"],
#         existing_player.EFG_PERCENT=row["EFG%"],
#         existing_player.TS_PERCENT=row["TS%"],
#         existing_player.USG_PERCENT=row["USG%"],
#         existing_player.PACE=row["PACE"],
#         existing_player.PIE=row["PIE"],
#         existing_player.POSS=row["POSS"],
#         existing_player.FGA2P_PERCENT=row["%FGA2PT"],
#         existing_player.FGA3P_PERCENT=row["%FGA3PT"],
#         existing_player.PTS2P_PERCENT=row["%PTS2PT"],
#         existing_player.PTS2P_MR_PERCENT=row["%PTS2PT MR"],
#         existing_player.PTS3P_PERCENT=row["%PTS3PT"],
#         existing_player.PTSFBPS_PERCENT=row["%PTSFBPS"],
#         existing_player.PTSFT_PERCENT=row["%PTSFT"],
#         existing_player.PTS_OFFTO_PERCENT=row["%PTSOFFTO"],
#         existing_player.PTSPITP_PERCENT=row["%PTSPITP"],
#         existing_player.FG2M_AST_PERCENT=row["2FGM%AST"],
#         existing_player.FG2M_UAST_PERCENT=row["2FGM%UAST"],
#         existing_player.FG3M_AST_PERCENT=row["3FGM%AST"],
#         existing_player.FG3M_UAST_PERCENT=row["3FGM%UAST"],
#         existing_player.FGM_AST_PERCENT=row["FGM%AST"],
#         existing_player.FGM_UAST_PERCENT=row["FGM%UAST"],
#         existing_player.DEF_RTG=row["DEF RTG"],
#         existing_player.DREB=row["DREB"],
#         existing_player.DREB_PERCENT_TEAM=row["DREB%TEAM"],
#         existing_player.STL=row["STL"],
#         existing_player.STL_PERCENT=row["STL%"],
#         existing_player.BLK=row["BLK"],
#         existing_player.BLK_PERCENT=row["%BLK"],
#         existing_player.OPP_PTS_OFFTO=row["OPP PTSOFF TOV"],
#         existing_player.OPP_PTS_2ND_CHANCE=row["OPP PTS2ND CHANCE"],
#         existing_player.OPP_PTS_FB=row["OPP PTSFB"],
#         existing_player.OPP_PTS_PAINT=row["OPP PTSPAINT"],
#         existing_player.DEFWS=row["DEFWS"]

#     if not existing_player:
#         player_stat = AdvancedPlayerStats(
#             PLAYER=normalized_name,
#             TEAM=row["TEAM"],
#             AGE=row["AGE"],
#             GP=row["GP"],
#             W=row["W"],
#             L=row["L"],
#             MIN=row["MIN"],
#             OFFRTG=row["OFFRTG"],
#             DEFRTG=row["DEFRTG"],
#             NETRTG=row["NETRTG"],
#             AST_PERCENT=row["AST%"],
#             AST_TO=row["AST/TO"],
#             AST_RATIO=row["AST RATIO"],
#             OREB_PERCENT=row["OREB%"],
#             DREB_PERCENT=row["DREB%"],
#             REB_PERCENT=row["REB%"],
#             TO_RATIO=row["TO RATIO"],
#             EFG_PERCENT=row["EFG%"],
#             TS_PERCENT=row["TS%"],
#             USG_PERCENT=row["USG%"],
#             PACE=row["PACE"],
#             PIE=row["PIE"],
#             POSS=row["POSS"],
#             FGA2P_PERCENT=row["%FGA2PT"],
#             FGA3P_PERCENT=row["%FGA3PT"],
#             PTS2P_PERCENT=row["%PTS2PT"],
#             PTS2P_MR_PERCENT=row["%PTS2PT MR"],
#             PTS3P_PERCENT=row["%PTS3PT"],
#             PTSFBPS_PERCENT=row["%PTSFBPS"],
#             PTSFT_PERCENT=row["%PTSFT"],
#             PTS_OFFTO_PERCENT=row["%PTSOFFTO"],
#             PTSPITP_PERCENT=row["%PTSPITP"],
#             FG2M_AST_PERCENT=row["2FGM%AST"],
#             FG2M_UAST_PERCENT=row["2FGM%UAST"],
#             FG3M_AST_PERCENT=row["3FGM%AST"],
#             FG3M_UAST_PERCENT=row["3FGM%UAST"],
#             FGM_AST_PERCENT=row["FGM%AST"],
#             FGM_UAST_PERCENT=row["FGM%UAST"],
#             DEF_RTG=row["DEF RTG"],
#             DREB=row["DREB"],
#             DREB_PERCENT_TEAM=row["DREB%TEAM"],
#             STL=row["STL"],
#             STL_PERCENT=row["STL%"],
#             BLK=row["BLK"],
#             BLK_PERCENT=row["%BLK"],
#             OPP_PTS_OFFTO=row["OPP PTSOFF TOV"],
#             OPP_PTS_2ND_CHANCE=row["OPP PTS2ND CHANCE"],
#             OPP_PTS_FB=row["OPP PTSFB"],
#             OPP_PTS_PAINT=row["OPP PTSPAINT"],
#             DEFWS=row["DEFWS"],
#         )
#         session.add(player_stat)

data = pd.read_excel("data/clusteredPlayers.xlsx")

for _, row in data.iterrows():
    normalized_name = normalize_name(row["Player"])
    existing_player = session.query(ClusteredPlayers).filter_by(PLAYER=normalized_name).first()
    if existing_player:
        existing_player.CLUSTER = row["Cluster"]
    else:
        clustered_player = ClusteredPlayers(
            PLAYER=normalized_name,
            CLUSTER=row["Cluster"],
        )
        session.add(clustered_player)


data = pd.read_excel("data/boxScores.xlsx")

for _, row in data.iterrows():
    normalized_name = normalize_name(row["PLAYER"])
    existing_player_game = session.query(BoxScores).filter_by(
        PLAYER=normalized_name,
        GAME_DATE=row["GAME DATE"]
    ).first()

    if existing_player_game:
        existing_player_game.TEAM = row["TEAM"]
        existing_player_game.MATCH_UP = row["MATCH UP"]
        existing_player_game.WL = row["W/L"]
        existing_player_game.MIN = safe_float(row["MIN"])
        existing_player_game.PTS = safe_float(row["PTS"])
        existing_player_game.FGM = safe_float(row["FGM"])
        existing_player_game.FGA = safe_float(row["FGA"])
        existing_player_game.FG_PERCENT = safe_float(row["FG%"])
        existing_player_game.THREE_PM = safe_float(row["3PM"])
        existing_player_game.THREE_PA = safe_float(row["3PA"])
        existing_player_game.THREE_PERCENT = safe_float(row["3P%"])
        existing_player_game.FTM = safe_float(row["FTM"])
        existing_player_game.FTA = safe_float(row["FTA"])
        existing_player_game.FT_PERCENT = safe_float(row["FT%"])
        existing_player_game.OREB = safe_float(row["OREB"])
        existing_player_game.DREB = safe_float(row["DREB"])
        existing_player_game.REB = safe_float(row["REB"])
        existing_player_game.AST = safe_float(row["AST"])
        existing_player_game.STL = safe_float(row["STL"])
        existing_player_game.BLK = safe_float(row["BLK"])
        existing_player_game.TOV = safe_float(row["TOV"])
        existing_player_game.PF = safe_float(row["PF"])
        existing_player_game.PLUS_MINUS = safe_float(row["+/-"])
        existing_player_game.FP = safe_float(row["FP"])
        existing_player_game.CLUSTER = safe_float( row["CLUSTER"])
        existing_player_game.Last3_FP_Avg = safe_float(row["Last3_FP_Avg"])
        existing_player_game.Last5_FP_Avg = safe_float(row["Last5_FP_Avg"])
        existing_player_game.Last7_FP_Avg = safe_float(row["Last7_FP_Avg"])
        existing_player_game.Season_FP_Avg = safe_float(row["Season_FP_Avg"])
    else:
        # Add new record
        box_score = BoxScores(
            PLAYER=normalized_name,
            TEAM=row["TEAM"],
            MATCH_UP=row["MATCH UP"],
            GAME_DATE=row["GAME DATE"],
            WL=row["W/L"],
            MIN=safe_float(row["MIN"]),
            PTS=safe_float(row["PTS"]),
            FGM=safe_float(row["FGM"]),
            FGA=safe_float(row["FGA"]),
            FG_PERCENT=safe_float(row["FG%"]),
            THREE_PM=safe_float(row["3PM"]),
            THREE_PA=safe_float(row["3PA"]),
            THREE_PERCENT=safe_float(row["3P%"]),
            FTM=safe_float(row["FTM"]),
            FTA=safe_float(row["FTA"]),
            FT_PERCENT=safe_float(row["FT%"]),
            OREB=safe_float(row["OREB"]),
            DREB=safe_float(row["DREB"]),
            REB=safe_float(row["REB"]),
            AST=safe_float(row["AST"]),
            STL=safe_float(row["STL"]),
            BLK=safe_float(row["BLK"]),
            TOV=safe_float(row["TOV"]),
            PF=safe_float(row["PF"]),
            PLUS_MINUS=safe_float(row["+/-"]),
            FP=safe_float(row["FP"]),
            CLUSTER=row["CLUSTER"],
            Last3_FP_Avg=safe_float(row["Last3_FP_Avg"]),
            Last5_FP_Avg=safe_float(row["Last5_FP_Avg"]),
            Last7_FP_Avg=safe_float(row["Last7_FP_Avg"]),
            Season_FP_Avg=safe_float(row["Season_FP_Avg"]),
        )
        session.add(box_score)

data = pd.read_excel("data/testPlayerPredictions.xlsx")

for _, row in data.iterrows():
    normalized_name = normalize_name(row["PLAYER"])
    existing_entry = session.query(TestPlayerPredictions).filter_by(
        PLAYER=normalized_name,
        GAME_DATE=row["GAME DATE"]
    ).first()

    if existing_entry:
        # Update existing entry
        existing_entry.Last3_FP_Avg = row["Last3_FP_Avg"]
        existing_entry.Last5_FP_Avg = row["Last5_FP_Avg"]
        existing_entry.Last7_FP_Avg = row["Last7_FP_Avg"]
        existing_entry.Season_FP_Avg = row["Season_FP_Avg"]
        existing_entry.CLUSTER = row["CLUSTER"]
        existing_entry.ACTUAL = row["Actual"]
        existing_entry.PREDICTED = row["Predicted"]
        existing_entry.ERROR = row["Error"]
    else:
        # Add new entry
        new_entry = TestPlayerPredictions(
            PLAYER=normalized_name,
            GAME_DATE=row["GAME DATE"],
            Last3_FP_Avg=row["Last3_FP_Avg"],
            Last5_FP_Avg=row["Last5_FP_Avg"],
            Last7_FP_Avg=row["Last7_FP_Avg"],
            Season_FP_Avg=row["Season_FP_Avg"],
            CLUSTER=row["CLUSTER"],
            ACTUAL=row["Actual"],
            PREDICTED=row["Predicted"],
            ERROR=row["Error"],
        )
        session.add(new_entry)

data = pd.read_excel("data/dailyPredictions.xlsx")

# To handle null bools
data["My Model Closer Prediction"] = data["My Model Closer Prediction"].apply(
    lambda x: None if isinstance(x, float) and math.isnan(x) else bool(x)
)

for _, row in data.iterrows():
    normalized_name = normalize_name(row["PLAYER"])
    existing_entry = session.query(DailyPlayerPredictions).filter_by(
        PLAYER=normalized_name,
        GAME_DATE=row["GAME DATE"]
    ).first()
    if existing_entry:
        # Update existing entry
        existing_entry.PPG_PROJECTION = safe_float(row["PPG Projection"])
        existing_entry.MY_MODEL_PREDICTED_FP = safe_float(row["My Model Predicted FP"])
        existing_entry.GAME_DATE = row["GAME DATE"]
        existing_entry.ACTUAL_FP = safe_float(row["ACTUAL FP"])
        existing_entry.MY_MODEL_CLOSER_PREDICTION= row["My Model Closer Prediction"]
    else:
        # Add new entry
        new_entry = DailyPlayerPredictions(
            PLAYER=normalized_name,
            PPG_PROJECTION = safe_float(row["PPG Projection"]),
            MY_MODEL_PREDICTED_FP = safe_float(row["My Model Predicted FP"]),
            GAME_DATE = row["GAME DATE"],
            ACTUAL_FP = safe_float(row["ACTUAL FP"]),
            MY_MODEL_CLOSER_PREDICTION= row["My Model Closer Prediction"],
        )
        session.add(new_entry)

# Commit and close
session.commit()
session.close()
print("Data migrated successfully!")
