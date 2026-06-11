import numpy as np
import pandas as pd
GROUPS_ADDR= "../data/tournament_groups.csv"
MAX_REST_DAY= 14
HOST_TEAMS= ["Mexico", "USA", "Canada"]

df= pd.read_csv(GROUPS_ADDR)
GROUPS= df.groupby("group")["team"].apply(list).to_dict()

GROUP_MATCH_DATES = {
    "A": [pd.Timestamp(d) for d in ["2026-06-11", "2026-06-18", "2026-06-24"]],
    "B": [pd.Timestamp(d) for d in ["2026-06-12", "2026-06-18", "2026-06-24"]],
    "C": [pd.Timestamp(d) for d in ["2026-06-13", "2026-06-19", "2026-06-24"]],
    "D": [pd.Timestamp(d) for d in ["2026-06-13", "2026-06-19", "2026-06-25"]],
    "E": [pd.Timestamp(d) for d in ["2026-06-14", "2026-06-20", "2026-06-25"]],
    "F": [pd.Timestamp(d) for d in ["2026-06-14", "2026-06-20", "2026-06-25"]],
    "G": [pd.Timestamp(d) for d in ["2026-06-15", "2026-06-21", "2026-06-26"]],
    "H": [pd.Timestamp(d) for d in ["2026-06-15", "2026-06-21", "2026-06-26"]],
    "I": [pd.Timestamp(d) for d in ["2026-06-16", "2026-06-22", "2026-06-26"]],
    "J": [pd.Timestamp(d) for d in ["2026-06-16", "2026-06-22", "2026-06-27"]],
    "K": [pd.Timestamp(d) for d in ["2026-06-17", "2026-06-23", "2026-06-27"]],
    "L": [pd.Timestamp(d) for d in ["2026-06-17", "2026-06-23", "2026-06-27"]]
}
R32_MATCHES = [
    (("R","A"), ("R","B")),  
    (("W","E"), ("T","ABCDF")), 
    (("W","F"), ("R","C")),   
    (("W","C"), ("R","F")),  
    (("W","I"), ("T","CDFGH")),  
    (("R","E"), ("R","I")),   
    (("W","A"), ("T","CEFHI")), 
    (("W","L"), ("T","EHIJK")),  
    (("W","D"), ("T","BEFIJ")), 
    (("W","G"), ("T","AEHIJ")), 
    (("R","K"), ("R","L")), 
    (("W","H"), ("R","J")),  
    (("W","B"), ("T","EFGIJ")), 
    (("W","J"), ("R","H")),  
    (("W","K"), ("T","DEIJL")), 
    (("R","D"), ("R","G")),
]
R16_PAIRS = [
    (1, 4),   
    (0, 2),   
    (3, 5),   
    (6, 7),   
    (11, 10), 
    (8, 9),   
    (13, 15), 
    (12, 14), 
]
QF_PAIRS = [
    (0, 1),
    (4, 5),
    (2, 3),
    (6, 7), 
]
SF_PAIRS = [
    (0, 1), 
    (2, 3),  
]
R32_DATES = [pd.Timestamp(d) for d in [
    "2026-06-28", "2026-06-29", "2026-06-29", "2026-06-29",
    "2026-06-30", "2026-06-30", "2026-06-30", "2026-07-01",
    "2026-07-01", "2026-07-01", "2026-07-02", "2026-07-02",
    "2026-07-02", "2026-07-03", "2026-07-03", "2026-07-03" 
]]
R16_DATES = [pd.Timestamp(d) for d in [
    "2026-07-04", "2026-07-04", "2026-07-05", "2026-07-05",
    "2026-07-06", "2026-07-06", "2026-07-07", "2026-07-07"
]]
QF_DATES = [pd.Timestamp(d) for d in [
    "2026-07-09", "2026-07-10", "2026-07-11", "2026-07-11"
]]
SF_DATES = [pd.Timestamp(d) for d in ["2026-07-14", "2026-07-15"]]
FINAL_DATE = pd.Timestamp("2026-07-19")

def is_neutral(home, away, host_teams):
    neutral= True
    if home in host_teams:
        neutral= False
    elif away in host_teams:
        home, away = away, home
        neutral= False
    return home, away, neutral
def compute_h2h_features(home, away, h2h_history):
    matches = h2h_history[home][away]
    total= len(matches)
    is_first= 1 if total == 0 else 0

    if is_first:
        return {
            "h2h_last5_home_winrate":0.5,
            "h2h_last5_avg_gd":0,
            "h2h_total_matches": 0,
            "h2h_is_first_meeting": 1,
        }
    last5 = matches[-5:]
    gd_sum= sum(last5)
    wins = sum(1 for gd in last5 if gd > 0)

    return {
        "h2h_last5_home_winrate": wins/len(last5),
        "h2h_last5_avg_gd": gd_sum / len(last5),
        "h2h_total_matches": total,
        "h2h_is_first_meeting": is_first
    }
def compute_rest_features(hf, af, current_date):
    res= {}
    res["home_rest"]= min((current_date-hf["last_match_date"]).days, MAX_REST_DAY)
    res["away_rest"]= min((current_date-af["last_match_date"]).days, MAX_REST_DAY)
    res["rest_diff"]= res["home_rest"]-res["away_rest"]
    return res       
def group_config_maker():
    configs= {}
    match_idx= 0
    matchdays= [
        [(0, 1), (2, 3)],
        [(0, 2), (3, 1)],
        [(3, 0), (1, 2)]
    ]
    for group_letter, group_teams in GROUPS.items():
        dates= GROUP_MATCH_DATES[group_letter]
        for idx, pairings in enumerate(matchdays):
            current_date= dates[idx]
            for (i, j) in pairings:
                home= group_teams[i]
                away= group_teams[j]
                home, away, neutral= is_neutral(home, away, HOST_TEAMS)
                configs[match_idx]= {
                    "home_team": home,
                    "away_team": away,
                    "date": current_date,
                    "neutral": neutral,
                    "group_letter": group_letter
                }
                match_idx += 1
    return home, away, configs
def predict_gd(model_ensemble, team_features, weights, configs, h2h_history):
    rows= []
    for d in configs.values():
        home= d["home_team"]
        away= d["away_team"]
        neutral= d["neutral"]
        date= d["date"]

        hf= team_features[home]
        af= team_features[away]
        elo_home_eff= hf["elo"] + (0 if neutral else 100)
        e_home= 1/(1+ 10**((af["elo"]-elo_home_eff)/400))

        h2h_features= compute_h2h_features(home, away, h2h_history)
        rest_features= compute_rest_features(hf, af, date)
        team_features[home]["last_match_date"]= date
        team_features[away]["last_match_date"]= date

        row= {
            "neutral": neutral,
            "home_rank": hf["rank"],
            "home_fifa_points": hf["fifa_points"],
            "away_rank": af["rank"],
            "away_fifa_points": af["fifa_points"],
            "home_avg_age": hf["avg_age"],
            "home_avg_value": hf["avg_value"],
            "away_avg_age": af["avg_age"],
            "away_avg_value": af["avg_value"],
            "home_rank_tier": hf["rank_tier"],
            "away_rank_tier": af["rank_tier"],
            "rank_diff": af["rank"] - hf["rank"],
            "home_last5_winrate": hf["last5_winrate"],
            "home_last5_avg_sd": hf["last5_avg_sd"],
            "away_last5_winrate": af["last5_winrate"],
            "away_last5_avg_sd": af["last5_avg_sd"],
            "match_type_ordinal": 3,
            "is_friendly": 0,
            "elo_home_pre": hf["elo"],
            "elo_away_pre": af["elo"],
            "elo_diff": hf["elo"]-af["elo"],
            "elo_win_prob": e_home,
            "fifa_points_diff": hf["fifa_points"]-af["fifa_points"],
            "rank_ratio": af["rank"]/hf["rank"],
            "tier_diff": af["rank_tier"]-hf["rank_tier"],
            "winrate_diff": hf["last5_winrate"]-af["last5_winrate"],
            "avg_sd_diff": hf["last5_avg_sd"]-af["last5_avg_sd"],
            "value_diff": hf["avg_value"]-af["avg_value"],
            "value_ratio": hf["avg_value"]/af["avg_value"],
            "age_diff": hf["avg_age"]-af["avg_age"],
            "h2h_last5_home_winrate":h2h_features["h2h_last5_home_winrate"],
            "h2h_last5_avg_gd":h2h_features["h2h_last5_avg_gd"],
            "h2h_total_matches": h2h_features["h2h_total_matches"],
            "h2h_is_first_meeting": h2h_features["h2h_is_first_meeting"],
            "home_days_rest": rest_features["home_rest"],
            "away_days_rest": rest_features["away_rest"],
            "rest_diff": rest_features["rest_diff"],
        }
        rows.append(row)
    X = pd.DataFrame(rows)

    xgb_m, lgb_m, cat_m = model_ensemble
    w = weights

    xgb_pred= xgb_m.predict(X)
    lgb_pred= lgb_m.predict(X)
    cat_pred= cat_m.predict(X)
    
    pred = w[0]*xgb_pred + w[1]*lgb_pred + w[2]*cat_pred
    return pred
def simulate_group(group_letters, model_ensemble, weights, team_features, sd_df, h2h_history):
    rows= []
    stats= {l: {t: {"pts": 0, "gd": 0} for t in GROUPS[l]} for l in group_letters}
    h2h= {l: {t1: {t2: 0 for t2 in GROUPS[l]} for t1 in GROUPS[l]} for l in group_letters}
    home, away, configs= group_config_maker()
    diff= predict_gd(model_ensemble, team_features, weights=weights, configs=configs, h2h_history= h2h_history)
    diff+= np.random.normal(0, 0.8, size=len(diff))
    diff_round= np.round(diff).astype(int)

    for i in range(len(diff_round)):
        c= configs[i]
        home= c["home_team"]
        away= c["away_team"]
        group= c["group_letter"]
        date= c["date"]

        if diff_round[i] > 0:
            stats[group][home]["pts"] += 3
            h2h[group][home][away]= 3
            h2h[group][away][home]= 0

        elif diff_round[i] == 0:
            stats[group][home]["pts"] += 1
            stats[group][away]["pts"] += 1
            h2h[group][home][away]= 1
            h2h[group][away][home]= 1

        else:
            stats[group][away]["pts"] += 3
            h2h[group][away][home]= 3
            h2h[group][home][away]= 0

        stats[group][home]["gd"] += diff_round[i]
        stats[group][away]["gd"] -= diff_round[i]
        rows.append({
            "home_team": home,
            "away_team": away,
            "date": date,
            "gd": diff_round[i],
        })
    sd_df= pd.concat([sd_df, pd.DataFrame(rows)], ignore_index=True)

    standings_history = []
    for group in group_letters:
        all_teams = list(GROUPS[group])


        all_teams.sort(
            key=lambda t: (
                -stats[group][t]["pts"],
                -stats[group][t]["gd"],
                team_features[t]["rank"]
            )
        )
        for i in range(len(all_teams) - 1):
            t1, t2 = all_teams[i], all_teams[i + 1]
            if(stats[group][t1]["pts"] == stats[group][t2]["pts"] and stats[group][t1]["gd"] == stats[group][t2]["gd"]):
                if h2h[group][t2][t1] > h2h[group][t1][t2]:
                    all_teams[i], all_teams[i + 1] = all_teams[i + 1], all_teams[i]
                    
        standings_history.append({
            "group": group,
            "table": [
                {
                    "team": t,
                    "pts": stats[group][t]["pts"],
                    "gd": stats[group][t]["gd"],
                    "rank": team_features[t]["rank"]
                }
                for t in all_teams
            ]
        })

    return standings_history, sd_df
def pick_best_thirds(third_place_teams):
    ranked= sorted(third_place_teams, key=lambda x: (-x["pts"], -x["gd"], x["rank"]))
    return ranked[:8]
def resolve_slot(slot, winners, runners_up, t_mapping):
    slot_type, group_code= slot
    if slot_type == "W":
        return winners[group_code]
    elif slot_type == "R":
        return runners_up[group_code]
    elif slot_type == "T":
        return t_mapping[group_code]    
def simulate_match_knockout(predicted_diff):
    if predicted_diff > 0:
        return "home"
    elif predicted_diff < 0:
        return "away"
    else:
        return "home" if np.random.random() < 0.5 else "away"
def knockout_config_maker(matches, matches_date):
    configs= {}
    match_idx= 0
    for (home, away), match_date in zip(matches, matches_date):
        home, away, neutral= is_neutral(home, away, HOST_TEAMS)
        configs[match_idx] = {
            "home_team": home,
            "away_team": away,
            "date": match_date,
            "neutral": neutral,
        }
        match_idx+= 1
    return configs
def simulate_knockout_round(matches, matches_date, model_ensemble, weights, sd_df, team_features, h2h_history):
    winners= []
    rows= []
    configs= knockout_config_maker(matches, matches_date)
    diff= predict_gd(model_ensemble, team_features, weights=weights, configs= configs, h2h_history= h2h_history)
    diff += np.random.normal(0, 1.2, size=len(diff))
    diff_round= np.round(diff).astype(int)

    for i in range(len(diff_round)):
        c= configs[i]
        home= c["home_team"]
        away= c["away_team"]
        match_date= c["date"]
        result= simulate_match_knockout(diff_round[i])
        winners.append(home if result == "home" else away)
        rows.append({
            "home_team": home,
            "away_team": away,
            "gd": diff_round[i],
            "date": match_date,
        })
    sd_df = sd_df = pd.concat([sd_df, pd.DataFrame(rows)], ignore_index=True)
    return winners, sd_df
def map_third_place_teams(best_thirds, r32_matches):
    t_slots = []
    for home, away in r32_matches:
        if home[0] == "T": t_slots.append(home[1])
        if away[0] == "T": t_slots.append(away[1])

    def solve(slot_idx, used_indices):
        if slot_idx == len(t_slots):
            return []
        current_constraint= t_slots[slot_idx]
        for i, t in enumerate(best_thirds):
            if i not in used_indices and t["group"] in current_constraint:
                res= solve(slot_idx + 1, used_indices | {i})
                if res is not None:
                    return [t["team"]] + res
        return None
    ordered_teams = solve(0, set())
    result= {t_slots[i]: ordered_teams[i] for i in range(len(t_slots))}
    return result
def compute_group_result(model_ensemble, weights, team_features, sd_df, h2h_history):
    winners= {}
    runners_up = {} 
    thirds= []  
    standings_history, sd_df = simulate_group(list(GROUPS.keys()), model_ensemble, weights, team_features, sd_df, h2h_history)
    for group_data in standings_history:
        group_letter= group_data["group"]
        table= group_data["table"]
        winners[group_letter] = table[0]["team"]
        runners_up[group_letter] = table[1]["team"]
        thirds.append({**table[2], "group": group_letter})
    best_thirds= pick_best_thirds(thirds)
    t_mapping= map_third_place_teams(best_thirds, R32_MATCHES)
    return winners, runners_up, t_mapping
def simulate_final(finalist_home, finalist_away, model_ensemble, team_features, weights, sd_df, h2h_history):
    finalist_home, finalist_away, neutral= is_neutral(finalist_home, finalist_away, HOST_TEAMS)
    configs= {
        0: {
            "home_team": finalist_home,
            "away_team": finalist_away,
            "date": FINAL_DATE,
            "neutral": neutral,
        }
    }
    diff= predict_gd(model_ensemble, team_features, weights=weights, configs= configs, h2h_history= h2h_history)
    diff += np.random.normal(0, 1.2)
    diff_round = np.round(diff[0]).astype(int)
    result= simulate_match_knockout(diff_round)
    champion= finalist_home if result == "home" else finalist_away
    runner= finalist_away if result == "home" else finalist_home
    return runner, champion
def r32_slot_resolver(winners, runners_up, t_mapping):
    r32_matches = []
    for home_slot, away_slot in R32_MATCHES:
        home_team = resolve_slot(home_slot, winners, runners_up, t_mapping)
        away_team = resolve_slot(away_slot, winners, runners_up, t_mapping)
        r32_matches.append((home_team, away_team))
    return r32_matches
def simulate_tournament(model_ensemble, weights, sd_df, team_features, h2h_history):
    winners, runners_up, t_mapping= compute_group_result(model_ensemble, weights, team_features, sd_df, h2h_history)
    advanced= (list(winners.values()) + list(runners_up.values()) + list(t_mapping.values()))

    r32_matches= r32_slot_resolver(winners, runners_up, t_mapping) 
    r32_winners, sd_df =simulate_knockout_round(r32_matches, R32_DATES, model_ensemble, weights, sd_df, team_features, h2h_history)

    r16_matches= [(r32_winners[h], r32_winners[a]) for h, a in R16_PAIRS]
    r16_winners, sd_df= simulate_knockout_round(r16_matches, R16_DATES, model_ensemble, weights, sd_df, team_features, h2h_history)

    qf_matches= [(r16_winners[h], r16_winners[a]) for h, a in QF_PAIRS]
    qf_winners, sd_df= simulate_knockout_round(qf_matches, QF_DATES, model_ensemble, weights, sd_df, team_features, h2h_history)

    sf_matches= [(qf_winners[h], qf_winners[a]) for h, a in SF_PAIRS]
    sf_winners, sd_df= simulate_knockout_round(sf_matches, SF_DATES, model_ensemble, weights, sd_df, team_features, h2h_history)

    finalist_home, finalist_away = sf_winners[0], sf_winners[1]
    runner, champion= simulate_final(finalist_home, finalist_away, model_ensemble, team_features, weights, sd_df, h2h_history)

    return {
        "group_stage":  advanced,
        "r32_winners":  r32_winners,
        "r16_winners":  r16_winners,
        "qf_winners":   qf_winners,
        "sf_winners":   sf_winners,
        "champion":     champion,
        "runner_up":    runner,
    }
def run_monte_carlo(model_ensemble, weights, clean_df, team_features, h2h_history, n_simulations=100, seed=8):
    np.random.seed(seed)

    all_teams= [t for teams in GROUPS.values() for t in teams]
    counts= {
        team: {
            "group_stage": 0,
            "r32": 0,
            "r16": 0,
            "qf": 0,
            "sf": 0,
            "champion": 0
        }
        for team in all_teams
    }

    sim_df= clean_df.copy()
    for _ in range(n_simulations):
        result = simulate_tournament(model_ensemble, weights, sim_df, team_features, h2h_history)
        for t in result["group_stage"]: counts[t]["group_stage"] += 1
        for t in result["r32_winners"]: counts[t]["r32"] += 1
        for t in result["r16_winners"]: counts[t]["r16"] += 1
        for t in result["qf_winners"]:  counts[t]["qf"] += 1
        for t in result["sf_winners"]:  counts[t]["sf"] += 1
        counts[result["champion"]]["champion"] += 1

    rows = []
    for team, c in counts.items():
        rows.append({
            "team":         team,
            "group_stage%": round(100 * c["group_stage"] / n_simulations, 2),
            "r32%":         round(100 * c["r32"]         / n_simulations, 2),
            "r16%":         round(100 * c["r16"]         / n_simulations, 2),
            "qf%":          round(100 * c["qf"]          / n_simulations, 2),
            "sf%":          round(100 * c["sf"]          / n_simulations, 2),
            "champion%":    round(100 * c["champion"]    / n_simulations, 2),
        })

    return (pd.DataFrame(rows).sort_values("champion%", ascending=False).reset_index(drop=True))
