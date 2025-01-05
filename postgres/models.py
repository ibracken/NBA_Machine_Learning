from sqlalchemy import Column, Integer, String, Float, ForeignKey, Date, UniqueConstraint, Boolean
from postgres.config import Base
from sqlalchemy.orm import relationship

class AdvancedPlayerStats(Base):
    __tablename__ = "advancedPlayerStats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    PLAYER = Column(String, unique = True, nullable=False)
    TEAM = Column(String, nullable=False)
    AGE = Column(Integer, nullable=False)
    GP = Column(Integer)
    W = Column(Integer)
    L = Column(Integer)
    MIN = Column(Float)
    OFFRTG = Column(Float)
    DEFRTG = Column(Float)
    NETRTG = Column(Float)
    AST_PERCENT = Column(Float)
    AST_TO = Column(Float)
    AST_RATIO = Column(Float)
    OREB_PERCENT = Column(Float)
    DREB_PERCENT = Column(Float)
    REB_PERCENT = Column(Float)
    TO_RATIO = Column(Float)
    EFG_PERCENT = Column(Float)
    TS_PERCENT = Column(Float)
    USG_PERCENT = Column(Float)
    PACE = Column(Float)
    PIE = Column(Float)
    POSS = Column(Integer)
    FGA2P_PERCENT = Column(Float)
    FGA3P_PERCENT = Column(Float)
    PTS2P_PERCENT = Column(Float)
    PTS2P_MR_PERCENT = Column(Float)
    PTS3P_PERCENT = Column(Float)
    PTSFBPS_PERCENT = Column(Float)
    PTSFT_PERCENT = Column(Float)
    PTS_OFFTO_PERCENT = Column(Float)
    PTSPITP_PERCENT = Column(Float)
    FG2M_AST_PERCENT = Column(Float)
    FG2M_UAST_PERCENT = Column(Float)
    FG3M_AST_PERCENT = Column(Float)
    FG3M_UAST_PERCENT = Column(Float)
    FGM_AST_PERCENT = Column(Float)
    FGM_UAST_PERCENT = Column(Float)
    DEF_RTG = Column(Float)
    DREB = Column(Float)
    DREB_PERCENT_TEAM = Column(Float)
    STL = Column(Float)
    STL_PERCENT = Column(Float)
    BLK = Column(Float)
    BLK_PERCENT = Column(Float)
    OPP_PTS_OFFTO = Column(Float)
    OPP_PTS_2ND_CHANCE = Column(Float)
    OPP_PTS_FB = Column(Float)
    OPP_PTS_PAINT = Column(Float)
    DEFWS = Column(Float)

    clusters = relationship("ClusteredPlayers", back_populates="player_stats")
    box_scores = relationship("BoxScores", back_populates="player_stats")
    test_predictions = relationship("TestPlayerPredictions", back_populates="player_stats")
    daily_predictions = relationship("DailyPlayerPredictions", back_populates="player_stats")



class ClusteredPlayers(Base):
    __tablename__ = "ClusteredPlayers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    PLAYER = Column(String, ForeignKey("advancedPlayerStats.PLAYER"), unique = True, nullable=False)
    CLUSTER = Column(String, nullable=False)

    player_stats = relationship("AdvancedPlayerStats", back_populates="clusters")



class BoxScores(Base):
    __tablename__ = "boxScores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    PLAYER = Column(String, ForeignKey("advancedPlayerStats.PLAYER"), nullable=False)
    TEAM = Column(String, nullable=False)
    MATCH_UP = Column(String, nullable=False)
    GAME_DATE = Column(Date, nullable=False)
    WL = Column(String, nullable=False)  # Win/Loss as a single string
    MIN = Column(Float, nullable=True)
    PTS = Column(Float, nullable=True)
    FGM = Column(Float, nullable=True)
    FGA = Column(Float, nullable=True)
    FG_PERCENT = Column(Float, nullable=True)
    THREE_PM = Column(Float, nullable=True)  # 3-point made
    THREE_PA = Column(Float, nullable=True)  # 3-point attempted
    THREE_PERCENT = Column(Float, nullable=True)
    FTM = Column(Float, nullable=True)
    FTA = Column(Float, nullable=True)
    FT_PERCENT = Column(Float, nullable=True)
    OREB = Column(Float, nullable=True)
    DREB = Column(Float, nullable=True)
    REB = Column(Float, nullable=True)
    AST = Column(Float, nullable=True)
    STL = Column(Float, nullable=True)
    BLK = Column(Float, nullable=True)
    TOV = Column(Float, nullable=True)  # Turnovers
    PF = Column(Float, nullable=True)   # Personal fouls
    PLUS_MINUS = Column(Float, nullable=True)
    FP = Column(Float, nullable=True)  # Fantasy points
    CLUSTER = Column(String, nullable=True)
    Last3_FP_Avg = Column(Float, nullable=True)
    Last5_FP_Avg = Column(Float, nullable=True)
    Last7_FP_Avg = Column(Float, nullable=True)
    Season_FP_Avg = Column(Float, nullable=True)


    # Unique constraint for player and game date
    __table_args__ = (
        UniqueConstraint("PLAYER", "GAME_DATE", name="uq_player_game_date"),
    )

    # Relationship with AdvancedPlayerStats
    player_stats = relationship("AdvancedPlayerStats", back_populates="box_scores")

class TestPlayerPredictions(Base):
    __tablename__ = "testPlayerPredictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    PLAYER = Column(String, ForeignKey("advancedPlayerStats.PLAYER"), nullable=False)
    Last3_FP_Avg = Column(Float, nullable=True)
    Last5_FP_Avg = Column(Float, nullable=True)
    Last7_FP_Avg = Column(Float, nullable=True)
    Season_FP_Avg = Column(Float, nullable=True)
    CLUSTER = Column(String, nullable=True)
    GAME_DATE = Column(Date, nullable=False)
    ACTUAL = Column(Float, nullable=True)
    PREDICTED = Column(Float, nullable=True)
    ERROR = Column(Float, nullable=True)
    
    
    __table_args__ = (
        UniqueConstraint("PLAYER", "GAME_DATE", name="uq_test_player_game_date"),
    )

    player_stats = relationship("AdvancedPlayerStats", back_populates="test_predictions")


class DailyPlayerPredictions(Base):
    __tablename__ = "dailyPlayerPredictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    PLAYER = Column(String, ForeignKey("advancedPlayerStats.PLAYER"), nullable=False)
    PPG_PROJECTION = Column(Float, nullable=True)
    MY_MODEL_PREDICTED_FP = Column(Float, nullable=True)
    GAME_DATE = Column(Date, nullable=False)
    ACTUAL_FP = Column(Float, nullable=True)
    MY_MODEL_CLOSER_PREDICTION = Column(Boolean, nullable=True)
    __table_args__ = (
        UniqueConstraint("PLAYER", "GAME_DATE", name="uq_daily_player_game_date"),
    )

    player_stats = relationship("AdvancedPlayerStats", back_populates="daily_predictions")