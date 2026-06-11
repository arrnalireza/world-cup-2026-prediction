import numpy as np
import pandas as pd

HOME_ADVANTAGE= 100
K_FRIENDLY= 20
K_QUALIFIER= 40
K_TOURNAMENT= 60
K_WORLDCUP= 70
GD_CAP= 7

QUALIFIER_KEYWORDS= ["qualification", "qualifier", "eliminatoria", "preliminary"]
TOURNAMENT_KEYWORDS= [
    "world cup", "copa america", "african cup", "africa cup",
    "asian cup", "gold cup", "nations league", "nations cup",
    "uefa european", "euro 20", "confederation", "olympic",
]

def rank_to_elo(rank: float) -> float:
    return 1800 - (rank - 1) * 5

def get_k(match_type: str) -> float:
    if match_type=="q":
        return K_QUALIFIER
    elif match_type=="w":
        return K_WORLDCUP
    elif match_type=="f":
        return K_FRIENDLY
    return K_TOURNAMENT

def expected(r_a: float, r_b: float) -> float:
    return 1/(1+10**((r_b - r_a)/400))

def gd_mult(goal_diff: int) -> float:
    return max(1, np.log1p(min(abs(goal_diff), GD_CAP)) / np.log(2))

def match_outcome(home_score: int, away_score: int):
    if home_score > away_score: return 1, 0
    if home_score < away_score: return 0, 1
    return 0.5, 0.5

def add_elo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ratings = {}   

    def get_rating(team, rank):
        if team not in ratings:
            ratings[team] = rank_to_elo(rank)
        return ratings[team]
    
    elo_home_pre, elo_away_pre= [], []
    for row in df.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        
        r_home = get_rating(home, row.home_rank)
        r_away = get_rating(away, row.away_rank)
        elo_home_pre.append(round(r_home, 1))
        elo_away_pre.append(round(r_away, 1))

        r_home_eff = r_home + (0 if row.neutral else HOME_ADVANTAGE)
        e_home = expected(r_home_eff, r_away)

        s_home, _ = match_outcome(row.home_score, row.away_score)
        k = get_k(row.match_type)
        mult = gd_mult(row.gd)
        delta= k * mult * (s_home - e_home)
        ratings[home]= r_home+delta
        ratings[away]= r_away-delta

    df["elo_home_pre"]= elo_home_pre
    df["elo_away_pre"]= elo_away_pre
    return df
