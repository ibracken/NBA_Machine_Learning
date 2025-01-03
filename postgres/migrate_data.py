import pandas as pd
from sqlalchemy.orm import Session
from config import SessionLocal
from models import AdvancedPlayerStats

# Load the data
data = pd.read_excel("data/NBAStats.xlsx")  # Replace with your file path

# Create a new session
session = SessionLocal()

# Insert data into the database
for _, row in data.iterrows():
    player_stat = AdvancedPlayerStats(
        PLAYER=row["PLAYER"],
        TEAM=row["TEAM"],
        AGE=row["AGE"],
        GP=row["GP"],
        W=row["W"],
        L=row["L"],
        MIN=row["MIN"],
        OFFRTG=row["OFFRTG"],
        DEFRTG=row["DEFRTG"],
        NETRTG=row["NETRTG"],
        AST_PERCENT=row["AST%"],
        AST_TO=row["AST/TO"],
        AST_RATIO=row["AST RATIO"],
        OREB_PERCENT=row["OREB%"],
        DREB_PERCENT=row["DREB%"],
        REB_PERCENT=row["REB%"],
        TO_RATIO=row["TO RATIO"],
        EFG_PERCENT=row["EFG%"],
        TS_PERCENT=row["TS%"],
        USG_PERCENT=row["USG%"],
        PACE=row["PACE"],
        PIE=row["PIE"],
        POSS=row["POSS"],
        FGA2P_PERCENT=row["%FGA2PT"],
        FGA3P_PERCENT=row["%FGA3PT"],
        PTS2P_PERCENT=row["%PTS2PT"],
        PTS2P_MR_PERCENT=row["%PTS2PT MR"],
        PTS3P_PERCENT=row["%PTS3PT"],
        PTSFBPS_PERCENT=row["%PTSFBPS"],
        PTSFT_PERCENT=row["%PTSFT"],
        PTS_OFFTO_PERCENT=row["%PTSOFFTO"],
        PTSPITP_PERCENT=row["%PTSPITP"],
        FG2M_AST_PERCENT=row["2FGM%AST"],
        FG2M_UAST_PERCENT=row["2FGM%UAST"],
        FG3M_AST_PERCENT=row["3FGM%AST"],
        FG3M_UAST_PERCENT=row["3FGM%UAST"],
        FGM_AST_PERCENT=row["FGM%AST"],
        FGM_UAST_PERCENT=row["FGM%UAST"],
        DEF_RTG=row["DEF RTG"],
        DREB=row["DREB"],
        DREB_PERCENT_TEAM=row["DREB%TEAM"], #TODO
        STL=row["STL"],
        STL_PERCENT=row["STL%"],
        BLK=row["BLK"],
        BLK_PERCENT=row["%BLK"],
        OPP_PTS_OFFTO=row["OPP PTSOFF TOV"],
        OPP_PTS_2ND_CHANCE=row["OPP PTS2ND CHANCE"],
        OPP_PTS_FB=row["OPP PTSFB"],
        OPP_PTS_PAINT=row["OPP PTSPAINT"],
        DEFWS=row["DEFWS"],
    )
    session.add(player_stat)

# Commit and close
session.commit()
session.close()
print("Data migrated successfully!")
