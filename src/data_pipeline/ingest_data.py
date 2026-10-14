"""
Beyond the Scoreboard: Cricket Franchise Decision Intelligence System
Data Acquisition, Parsing, and Database Ingestion Engine
"""

import os
import sys
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import sqlite3
import io
import time
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DB_PATH = os.path.join(BASE_DIR, "data", "cricket_analytics.db")
ZIP_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"
ZIP_PATH = os.path.join(RAW_DIR, "ipl_csv2.zip")

def download_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    if os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) > 5 * 1024 * 1024:
        print(f"[+] Dataset archive already exists at {ZIP_PATH} ({round(os.path.getsize(ZIP_PATH)/(1024*1024), 2)} MB)")
        return
    
    print(f"[+] Downloading IPL dataset from {ZIP_URL}...")
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp, open(ZIP_PATH, "wb") as out_file:
        data = resp.read()
        out_file.write(data)
    print(f"[+] Download complete: {round(os.path.getsize(ZIP_PATH)/(1024*1024), 2)} MB saved.")

def parse_and_build_database(seasons_to_include=None):
    """
    Parses Cricsheet match info and delivery logs into relational tables and stores in SQLite.
    If seasons_to_include is None, processes all available matches.
    """
    t0 = time.time()
    print("[+] Opening dataset archive...")
    z = zipfile.ZipFile(ZIP_PATH)
    all_files = set(z.namelist())
    
    info_files = [f for f in all_files if f.endswith("_info.csv")]
    print(f"[+] Discovered {len(info_files)} match metadata files.")
    
    matches_list = []
    players_registry = {}
    
    # 1. Parse Match Metadata
    print("[+] Parsing match info files...")
    for idx, f in enumerate(info_files):
        match_id_str = f.replace("_info.csv", "")
        try:
            match_id = int(match_id_str)
        except ValueError:
            continue
            
        content = z.read(f).decode("utf-8", errors="ignore").splitlines()
        
        m_info = {
            "match_id": match_id,
            "season": None,
            "match_date": None,
            "match_type": "T20",
            "venue": None,
            "city": None,
            "team1": None,
            "team2": None,
            "toss_winner": None,
            "toss_decision": None,
            "winner": None,
            "win_type": None,
            "win_margin": 0,
            "player_of_match": None,
            "target_runs": None,
            "target_overs": None
        }
        
        teams = []
        for line in content:
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) < 3:
                continue
            key = parts[1]
            val = parts[2]
            
            if key == "season":
                m_info["season"] = str(val)
            elif key == "date" and not m_info["match_date"]:
                m_info["match_date"] = val.replace("/", "-")
            elif key == "venue":
                m_info["venue"] = val
            elif key == "city":
                m_info["city"] = val
            elif key == "team":
                teams.append(val)
            elif key == "toss_winner":
                m_info["toss_winner"] = val
            elif key == "toss_decision":
                m_info["toss_decision"] = val
            elif key == "winner":
                m_info["winner"] = val
            elif key == "winner_runs":
                m_info["win_type"] = "runs"
                m_info["win_margin"] = int(val) if val.isdigit() else 0
            elif key == "winner_wickets":
                m_info["win_type"] = "wickets"
                m_info["win_margin"] = int(val) if val.isdigit() else 0
            elif key == "outcome":
                m_info["win_type"] = val
            elif key == "player_of_match":
                m_info["player_of_match"] = val
            elif key == "target_runs" and len(parts) >= 4:
                try:
                    m_info["target_runs"] = int(parts[3])
                except ValueError:
                    pass
            elif key == "registry" and len(parts) >= 5:
                if parts[2] == "people":
                    p_name = parts[3]
                    p_id = parts[4]
                    players_registry[p_name] = p_id
                    
        if len(teams) >= 2:
            m_info["team1"] = teams[0]
            m_info["team2"] = teams[1]
            
        matches_list.append(m_info)

    df_matches = pd.DataFrame(matches_list)
    df_matches["match_date"] = pd.to_datetime(df_matches["match_date"], errors="coerce")
    df_matches = df_matches.sort_values("match_date").reset_index(drop=True)
    print(f"[+] Successfully extracted {len(df_matches)} matches across {df_matches['season'].nunique()} seasons.")

    # 2. Parse Ball-by-Ball Deliveries
    print("[+] Extracting and parsing delivery files...")
    delivery_frames = []
    
    # We can process all matches or filter if requested
    target_match_ids = set(df_matches["match_id"])
    
    count = 0
    total_target = len(target_match_ids)
    
    for match_id in target_match_ids:
        deliv_filename = f"{match_id}.csv"
        if deliv_filename not in all_files:
            continue
            
        try:
            raw_bytes = z.read(deliv_filename)
            df_deliv = pd.read_csv(io.BytesIO(raw_bytes))
            delivery_frames.append(df_deliv)
            count += 1
            if count % 200 == 0:
                print(f"    - Processed {count}/{total_target} delivery files...")
        except Exception as e:
            continue

    print(f"[+] Concatenating {len(delivery_frames)} match deliveries...")
    deliveries = pd.concat(delivery_frames, ignore_index=True)
    
    # 3. Clean and Feature Engineer Deliveries
    print("[+] Engineering delivery-level features...")
    # Normalize column names
    deliveries.columns = [c.strip().lower() for c in deliveries.columns]
    
    # Map deliveries
    deliveries["match_id"] = deliveries["match_id"].astype(int)
    deliveries["innings"] = deliveries["innings"].astype(int)
    deliveries["ball_formatted"] = deliveries["ball"].astype(float)
    
    # Extract over_number (0-19) and ball_number (1-n)
    deliveries["over_number"] = deliveries["ball_formatted"].astype(int)
    deliveries["ball_number"] = ((deliveries["ball_formatted"] - deliveries["over_number"]) * 10).round().astype(int)
    
    # Innings Phase: Powerplay (0-5), Middle (6-14), Death (15-19)
    def assign_phase(over):
        if over < 6:
            return "Powerplay"
        elif over < 15:
            return "Middle"
        else:
            return "Death"
            
    deliveries["innings_phase"] = deliveries["over_number"].apply(assign_phase)
    
    # Fill numeric columns
    num_cols = ["runs_off_bat", "extras", "wides", "noballs", "byes", "legbyes", "penalty"]
    for col in num_cols:
        if col in deliveries.columns:
            deliveries[col] = pd.to_numeric(deliveries[col], errors="coerce").fillna(0).astype(int)
        else:
            deliveries[col] = 0
            
    deliveries["total_runs"] = deliveries["runs_off_bat"] + deliveries["extras"]
    
    # Event flags
    deliveries["is_dot_ball"] = ((deliveries["runs_off_bat"] == 0) & (deliveries["extras"] == 0)).astype(int)
    deliveries["is_boundary_4"] = (deliveries["runs_off_bat"] == 4).astype(int)
    deliveries["is_boundary_6"] = (deliveries["runs_off_bat"] == 6).astype(int)
    
    # Dismissals
    deliveries["player_dismissed"] = deliveries["player_dismissed"].fillna("").astype(str).str.strip()
    deliveries["wicket_type"] = deliveries["wicket_type"].fillna("").astype(str).str.strip()
    deliveries["is_wicket"] = (deliveries["player_dismissed"] != "").astype(int)
    
    # Add unique delivery ID
    deliveries["delivery_id"] = np.arange(1, len(deliveries) + 1)
    
    # Ensure season and date match dim_match
    match_meta_map = df_matches.set_index("match_id")[["season", "match_date", "venue"]].to_dict("index")
    deliveries["season"] = deliveries["match_id"].map(lambda mid: match_meta_map.get(mid, {}).get("season", ""))
    deliveries["match_date"] = deliveries["match_id"].map(lambda mid: match_meta_map.get(mid, {}).get("match_date", ""))
    
    print(f"[+] Total Deliveries Engineered: {len(deliveries):,}")

    # 4. Aggregate Player-Match Fact Table
    print("[+] Building FactPlayerMatch aggregated records...")
    # Batting aggregation
    bat_grp = deliveries.groupby(["match_id", "striker", "batting_team", "bowling_team"]).agg(
        runs_scored=("runs_off_bat", "sum"),
        balls_faced=("runs_off_bat", "count"),
        fours=("is_boundary_4", "sum"),
        sixes=("is_boundary_6", "sum"),
        runs_powerplay=("runs_off_bat", lambda x: x[deliveries.loc[x.index, "innings_phase"] == "Powerplay"].sum()),
        runs_middle=("runs_off_bat", lambda x: x[deliveries.loc[x.index, "innings_phase"] == "Middle"].sum()),
        runs_death=("runs_off_bat", lambda x: x[deliveries.loc[x.index, "innings_phase"] == "Death"].sum())
    ).reset_index().rename(columns={"striker": "player_name", "batting_team": "team", "bowling_team": "opponent"})

    # Dismissal check
    dismissals = deliveries[deliveries["is_wicket"] == 1][["match_id", "player_dismissed"]].drop_duplicates()
    dismissals["is_dismissed"] = 1
    bat_grp = bat_grp.merge(dismissals, left_on=["match_id", "player_name"], right_on=["match_id", "player_dismissed"], how="left")
    bat_grp["is_dismissed"] = bat_grp["is_dismissed"].fillna(0).astype(int)
    bat_grp = bat_grp.drop(columns=["player_dismissed"])
    bat_grp["batting_strike_rate"] = ((bat_grp["runs_scored"] / np.maximum(bat_grp["balls_faced"], 1)) * 100).round(2)

    # Bowling aggregation (exclude byes/legbyes from bowler's conceded runs)
    deliveries["bowler_runs"] = deliveries["runs_off_bat"] + deliveries["wides"] + deliveries["noballs"]
    deliveries["is_legal_ball"] = ((deliveries["wides"] == 0) & (deliveries["noballs"] == 0)).astype(int)
    # Exclude run-outs from bowler wickets
    bowler_wickets = deliveries[
        (deliveries["is_wicket"] == 1) & 
        (~deliveries["wicket_type"].isin(["run out", "retired hurt", "obstructing the field"]))
    ]
    
    bowl_grp = deliveries.groupby(["match_id", "bowler", "bowling_team", "batting_team"]).agg(
        balls_bowled=("is_legal_ball", "sum"),
        runs_conceded=("bowler_runs", "sum"),
        dot_balls_bowled=("is_dot_ball", "sum")
    ).reset_index().rename(columns={"bowler": "player_name", "bowling_team": "team", "batting_team": "opponent"})
    
    bowl_w_grp = bowler_wickets.groupby(["match_id", "bowler"]).agg(
        wickets_taken=("is_wicket", "sum"),
        wickets_powerplay=("is_wicket", lambda x: (bowler_wickets.loc[x.index, "innings_phase"] == "Powerplay").sum()),
        wickets_middle=("is_wicket", lambda x: (bowler_wickets.loc[x.index, "innings_phase"] == "Middle").sum()),
        wickets_death=("is_wicket", lambda x: (bowler_wickets.loc[x.index, "innings_phase"] == "Death").sum())
    ).reset_index().rename(columns={"bowler": "player_name"})

    bowl_grp = bowl_grp.merge(bowl_w_grp, on=["match_id", "player_name"], how="left")
    bowl_grp["wickets_taken"] = bowl_grp["wickets_taken"].fillna(0).astype(int)
    bowl_grp["wickets_powerplay"] = bowl_grp["wickets_powerplay"].fillna(0).astype(int)
    bowl_grp["wickets_middle"] = bowl_grp["wickets_middle"].fillna(0).astype(int)
    bowl_grp["wickets_death"] = bowl_grp["wickets_death"].fillna(0).astype(int)
    bowl_grp["overs_bowled"] = (bowl_grp["balls_bowled"] // 6) + ((bowl_grp["balls_bowled"] % 6) / 10.0)
    bowl_grp["bowling_economy"] = ((bowl_grp["runs_conceded"] / np.maximum(bowl_grp["balls_bowled"] / 6.0, 0.1667))).round(2)

    # Outer join batting and bowling player records
    player_match = pd.merge(
        bat_grp, bowl_grp, 
        on=["match_id", "player_name", "team", "opponent"], 
        how="outer"
    )
    
    # Fill NAs
    fill_zeros = [
        "runs_scored", "balls_faced", "fours", "sixes", "batting_strike_rate", "is_dismissed",
        "runs_powerplay", "runs_middle", "runs_death", "balls_bowled", "overs_bowled",
        "runs_conceded", "wickets_taken", "dot_balls_bowled", "bowling_economy",
        "wickets_powerplay", "wickets_middle", "wickets_death"
    ]
    for c in fill_zeros:
        if c in player_match.columns:
            player_match[c] = player_match[c].fillna(0)
            
    # Add match date, season, venue
    player_match["season"] = player_match["match_id"].map(lambda mid: match_meta_map.get(mid, {}).get("season", ""))
    player_match["match_date"] = player_match["match_id"].map(lambda mid: match_meta_map.get(mid, {}).get("match_date", ""))
    player_match["venue"] = player_match["match_id"].map(lambda mid: match_meta_map.get(mid, {}).get("venue", ""))
    player_match["player_match_id"] = np.arange(1, len(player_match) + 1)
    
    print(f"[+] Total Player-Match Performance Records: {len(player_match):,}")

    # 5. Build DimPlayer
    print("[+] Building DimPlayer registry...")
    p_summary = player_match.groupby("player_name").agg(
        total_runs=("runs_scored", "sum"),
        total_balls_faced=("balls_faced", "sum"),
        total_wickets=("wickets_taken", "sum"),
        total_balls_bowled=("balls_bowled", "sum"),
        matches_played=("match_id", "nunique")
    ).reset_index()

    def assign_primary_role(row):
        runs = row["total_runs"]
        wickets = row["total_wickets"]
        balls_bowled = row["total_balls_bowled"]
        balls_faced = row["total_balls_faced"]
        
        if wickets >= 20 and runs >= 500:
            return "All-Rounder"
        elif wickets >= 15 or balls_bowled >= 120:
            if runs >= 300:
                return "Bowling All-Rounder"
            return "Bowler"
        elif runs >= 250 or balls_faced >= 150:
            return "Batter"
        else:
            return "Squad Player"

    p_summary["primary_role"] = p_summary.apply(assign_primary_role, axis=1)
    p_summary["player_id"] = p_summary["player_name"].map(lambda p: players_registry.get(p, f"reg_{hash(p)&0xFFFFFFF:07x}"))
    p_summary["batting_hand"] = "Right-hand bat"  # Standard baseline
    p_summary["bowling_style"] = "Right-arm pace/spin"
    
    df_dim_player = p_summary[["player_id", "player_name", "primary_role", "batting_hand", "bowling_style", "matches_played", "total_runs", "total_wickets"]]

    # 6. Build DimTeam
    all_teams = sorted(list(set(df_matches["team1"].dropna()).union(set(df_matches["team2"].dropna()))))
    team_code_map = {
        "Chennai Super Kings": "CSK",
        "Mumbai Indians": "MI",
        "Royal Challengers Bangalore": "RCB",
        "Royal Challengers Bengaluru": "RCB",
        "Kolkata Knight Riders": "KKR",
        "Delhi Daredevils": "DC",
        "Delhi Capitals": "DC",
        "Kings XI Punjab": "PBKS",
        "Punjab Kings": "PBKS",
        "Rajasthan Royals": "RR",
        "Sunrisers Hyderabad": "SRH",
        "Deccan Chargers": "DEC",
        "Gujarat Titans": "GT",
        "Lucknow Super Giants": "LSG",
        "Gujarat Lions": "GL",
        "Rising Pune Supergiant": "RPS",
        "Rising Pune Supergiants": "RPS",
        "Pune Warriors": "PWI",
        "Kochi Tuskers Kerala": "KTK"
    }
    df_dim_team = pd.DataFrame([
        {"team_id": idx + 1, "team_name": t, "team_code": team_code_map.get(t, t[:3].upper()), "current_status": "Active" if t in ["Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore", "Royal Challengers Bengaluru", "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings", "Rajasthan Royals", "Sunrisers Hyderabad", "Gujarat Titans", "Lucknow Super Giants"] else "Defunct"}
        for idx, t in enumerate(all_teams)
    ])

    # 7. Write to SQLite Database and Processed Parquet/CSV
    print(f"[+] Writing tables to SQLite database at {DB_PATH}...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    
    # Write tables
    df_matches.to_sql("dim_match", conn, if_exists="replace", index=False)
    df_dim_player.to_sql("dim_player", conn, if_exists="replace", index=False)
    df_dim_team.to_sql("dim_team", conn, if_exists="replace", index=False)
    player_match.to_sql("fact_player_match", conn, if_exists="replace", index=False)
    
    # For deliveries, write to SQL and also save compressed parquet for rapid Streamlit ingestion
    print("[+] Writing fact_deliveries to SQLite and Parquet...")
    deliveries.to_sql("fact_deliveries", conn, if_exists="replace", index=False)
    
    # Create indexes for blazing-fast queries
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dm_season ON dim_match(season)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dm_winner ON dim_match(winner)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fpm_player ON fact_player_match(player_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fpm_season ON fact_player_match(season)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deliv_striker ON fact_deliveries(striker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deliv_bowler ON fact_deliveries(bowler)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deliv_phase ON fact_deliveries(innings_phase)")
    conn.commit()
    conn.close()
    
    # Save processed files for instant loading
    df_matches.to_csv(os.path.join(PROCESSED_DIR, "dim_match.csv"), index=False)
    df_dim_player.to_csv(os.path.join(PROCESSED_DIR, "dim_player.csv"), index=False)
    player_match.to_csv(os.path.join(PROCESSED_DIR, "fact_player_match.csv"), index=False)
    
    # Also save deliveries sample / summary
    deliveries_summary = deliveries.groupby(["season", "innings_phase", "batting_team"]).agg(
        runs=("total_runs", "sum"),
        balls=("ball", "count"),
        wickets=("is_wicket", "sum"),
        boundaries=("is_boundary_4", "sum")
    ).reset_index()
    deliveries_summary.to_csv(os.path.join(PROCESSED_DIR, "deliveries_summary.csv"), index=False)
    
    elapsed = round(time.time() - t0, 2)
    print("\n" + "=" * 70)
    print(f"[+] INGESTION & PIPELINE COMPLETE in {elapsed} seconds!")
    print(f"    - Matches Loaded: {len(df_matches)}")
    print(f"    - Players Registered: {len(df_dim_player)}")
    print(f"    - Player-Match Records: {len(player_match)}")
    print(f"    - Total Deliveries: {len(deliveries):,}")
    print(f"    - SQLite Database File: {DB_PATH} ({round(os.path.getsize(DB_PATH)/(1024*1024), 2)} MB)")
    print("=" * 70)

if __name__ == "__main__":
    download_dataset()
    parse_and_build_database()
