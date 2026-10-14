"""
Beyond the Scoreboard: Cricket Franchise Decision Intelligence System
Ultra-Resilient & Universal Data Ingestion Engine
Handles:
1. Multi-file Cricsheet ZIP packages
2. Single delivery CSVs (Kaggle, ESPNcricinfo, custom franchise logs)
3. CSVs with missing mandatory columns (auto-synthesis, heuristic imputation, self-healing)
4. Player/Match summary CSVs (auto-adapts schema without throwing errors)
"""

import os
import sys
import io
import re
import zipfile
import urllib.request
import sqlite3
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "cricket_analytics.db")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "src", "models", "saved_models")

PRESET_DATASETS = {
    "IPL (Indian Premier League)": "https://cricsheet.org/downloads/ipl_csv2.zip",
    "BBL (Big Bash League)": "https://cricsheet.org/downloads/bbl_csv2.zip",
    "PSL (Pakistan Super League)": "https://cricsheet.org/downloads/psl_csv2.zip",
    "CPL (Caribbean Premier League)": "https://cricsheet.org/downloads/cpl_csv2.zip",
    "Men's T20 Internationals (T20Is)": "https://cricsheet.org/downloads/t20s_csv2.zip"
}

def fuzzy_match_column(df_columns, patterns):
    """Finds the best matching column from a list of regex patterns."""
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for col in df_columns:
            if regex.search(str(col).strip()):
                return col
    return None

def heal_and_standardize_dataframe(df):
    """
    Self-healing engine: Inspects any input DataFrame, detects available columns,
    normalizes column names, and synthesizes missing attributes so the downstream
    Descriptive, Diagnostic, Predictive, and Prescriptive pipelines never fail.
    """
    df = df.copy()
    cols = list(df.columns)
    repair_log = []

    # 1. Striker / Batter
    striker_col = fuzzy_match_column(cols, [r"^striker$", r"^batsman$", r"^batter$", r"^player_name$", r"^player$", r"^batter_name$"])
    if striker_col:
        df["striker"] = df[striker_col].astype(str)
        repair_log.append(f"Mapped striker/batter from '{striker_col}'")
    else:
        df["striker"] = "Player " + (df.index % 11 + 1).astype(str)
        repair_log.append("Synthesized missing 'striker' using generic player rotation")

    # 2. Bowler
    bowler_col = fuzzy_match_column(cols, [r"^bowler$", r"^bowling_player$", r"^bowler_name$", r"^bowling$"])
    if bowler_col:
        df["bowler"] = df[bowler_col].astype(str)
        repair_log.append(f"Mapped bowler from '{bowler_col}'")
    else:
        df["bowler"] = "Bowler " + ((df.index // 6) % 5 + 1).astype(str)
        repair_log.append("Synthesized missing 'bowler' using standard 5-bowler rotation")

    # 3. Batting Team & Bowling Team
    bat_team_col = fuzzy_match_column(cols, [r"^batting_team$", r"^bat_team$", r"^team_batting$", r"^team1$", r"^team$"])
    bowl_team_col = fuzzy_match_column(cols, [r"^bowling_team$", r"^bowl_team$", r"^team_bowling$", r"^team2$", r"^opposition$", r"^opponent$"])

    if bat_team_col:
        df["batting_team"] = df[bat_team_col].astype(str)
        repair_log.append(f"Mapped batting team from '{bat_team_col}'")
    else:
        df["batting_team"] = "Franchise A"
        repair_log.append("Defaulted missing 'batting_team' to 'Franchise A'")

    if bowl_team_col:
        df["bowling_team"] = df[bowl_team_col].astype(str)
        repair_log.append(f"Mapped bowling team from '{bowl_team_col}'")
    else:
        df["bowling_team"] = np.where(df["batting_team"] == "Franchise A", "Franchise B", "Franchise A")
        repair_log.append("Synthesized 'bowling_team' as inverse of batting team")

    # 4. Runs Off Bat & Extras
    runs_col = fuzzy_match_column(cols, [r"^runs_off_bat$", r"^batsman_runs$", r"^batter_runs$", r"^runs$", r"^bat_runs$", r"^score$"])
    if runs_col:
        df["runs_off_bat"] = pd.to_numeric(df[runs_col], errors="coerce").fillna(0).astype(int)
        repair_log.append(f"Mapped runs off bat from '{runs_col}'")
    else:
        df["runs_off_bat"] = 0
        repair_log.append("Imputed missing 'runs_off_bat' with zeros")

    extras_col = fuzzy_match_column(cols, [r"^extras$", r"^extra_runs$", r"^total_extras$"])
    if extras_col:
        df["extras"] = pd.to_numeric(df[extras_col], errors="coerce").fillna(0).astype(int)
    else:
        df["extras"] = 0

    df["total_runs"] = df["runs_off_bat"] + df["extras"]

    # 5. Overs & Balls
    ball_col = fuzzy_match_column(cols, [r"^ball$", r"^ball_formatted$", r"^delivery$", r"^actual_delivery$"])
    over_col = fuzzy_match_column(cols, [r"^over$", r"^over_number$", r"^overs$"])

    if ball_col:
        df["ball_formatted"] = pd.to_numeric(df[ball_col], errors="coerce").fillna(0.1)
        df["over_number"] = df["ball_formatted"].astype(int)
        df["ball_number"] = ((df["ball_formatted"] - df["over_number"]) * 10).round().astype(int)
        repair_log.append(f"Mapped deliveries from '{ball_col}'")
    elif over_col:
        df["over_number"] = pd.to_numeric(df[over_col], errors="coerce").fillna(0).astype(int)
        # Sequential ball numbering within overs
        df["ball_number"] = (df.groupby(df["over_number"]).cumcount() % 6) + 1
        df["ball_formatted"] = df["over_number"] + (df["ball_number"] / 10.0)
        repair_log.append(f"Synthesized ball format from over column '{over_col}'")
    else:
        # Generate progressive 20-over sequences (120 balls per innings)
        ball_idx = np.arange(len(df))
        df["over_number"] = (ball_idx // 6) % 20
        df["ball_number"] = (ball_idx % 6) + 1
        df["ball_formatted"] = df["over_number"] + (df["ball_number"] / 10.0)
        repair_log.append("Synthesized 20-over timeline sequentially from row indices")

    # 6. Innings
    innings_col = fuzzy_match_column(cols, [r"^innings$", r"^inning$", r"^inn$"])
    if innings_col:
        df["innings"] = pd.to_numeric(df[innings_col], errors="coerce").fillna(1).astype(int)
    else:
        # Toggle innings every 120 balls or whenever over resets to 0
        over_resets = (df["over_number"] == 0) & (df["over_number"].shift(1) > 0)
        df["innings"] = (over_resets.cumsum() % 2) + 1
        repair_log.append("Synthesized innings split (1 & 2) based on over transitions")

    # 7. Match ID
    mid_col = fuzzy_match_column(cols, [r"^match_id$", r"^id$", r"^matchid$", r"^match_number$"])
    if mid_col:
        df["match_id"] = pd.to_numeric(df[mid_col], errors="coerce").fillna(1).astype(int)
    else:
        # Create a new match every 240 balls (approx 2 innings of 20 overs)
        df["match_id"] = (np.arange(len(df)) // 240) + 1
        repair_log.append("Synthesized match IDs in 240-delivery match chunks")

    # 8. Innings Phase
    def assign_phase(over):
        if over < 6: return "Powerplay"
        elif over < 15: return "Middle"
        return "Death"
    df["innings_phase"] = df["over_number"].apply(assign_phase)

    # 9. Wickets & Dismissals
    w_col = fuzzy_match_column(cols, [r"^is_wicket$", r"^wicket$", r"^out$"])
    dismissed_col = fuzzy_match_column(cols, [r"^player_dismissed$", r"^dismissed_player$", r"^batsman_out$"])
    wicket_type_col = fuzzy_match_column(cols, [r"^wicket_type$", r"^dismissal_kind$", r"^how_out$"])

    if dismissed_col:
        df["player_dismissed"] = df[dismissed_col].fillna("").astype(str).str.strip()
    else:
        df["player_dismissed"] = ""

    if w_col:
        df["is_wicket"] = pd.to_numeric(df[w_col], errors="coerce").fillna(0).astype(int)
    else:
        df["is_wicket"] = (df["player_dismissed"] != "").astype(int)

    if wicket_type_col:
        df["wicket_type"] = df[wicket_type_col].fillna("").astype(str)
    else:
        df["wicket_type"] = np.where(df["is_wicket"] == 1, "caught", "")

    # 10. Boundaries & Dots
    df["is_dot_ball"] = ((df["runs_off_bat"] == 0) & (df["extras"] == 0)).astype(int)
    df["is_boundary_4"] = (df["runs_off_bat"] == 4).astype(int)
    df["is_boundary_6"] = (df["runs_off_bat"] == 6).astype(int)
    df["bowler_runs"] = df["runs_off_bat"] + df.get("wides", 0) + df.get("noballs", 0)
    df["is_legal_ball"] = 1
    df["delivery_id"] = np.arange(1, len(df) + 1)

    # 11. Season, Date & Venue
    season_col = fuzzy_match_column(cols, [r"^season$", r"^year$"])
    df["season"] = df[season_col].astype(str) if season_col else "2026"

    date_col = fuzzy_match_column(cols, [r"^start_date$", r"^match_date$", r"^date$"])
    df["match_date"] = pd.to_datetime(df[date_col], errors="coerce").fillna(pd.Timestamp("2026-01-01")) if date_col else pd.Timestamp("2026-01-01")

    venue_col = fuzzy_match_column(cols, [r"^venue$", r"^stadium$", r"^ground$"])
    df["venue"] = df[venue_col].astype(str) if venue_col else "Standard Stadium"

    return df, repair_log

def synthesize_matches_table(deliveries):
    """Constructs dim_match from delivery records with winners, targets, and margins."""
    match_grp = deliveries.groupby("match_id").agg(
        team1=("batting_team", "first"),
        team2=("bowling_team", "first"),
        venue=("venue", "first"),
        season=("season", "first"),
        match_date=("match_date", "first")
    ).reset_index()

    # Calculate scores to find winners
    scores = deliveries.groupby(["match_id", "batting_team"])["total_runs"].sum().reset_index()
    winners = {}
    targets = {}
    win_margins = {}
    win_types = {}

    for mid, grp in scores.groupby("match_id"):
        if len(grp) >= 2:
            sorted_g = grp.sort_values("total_runs", ascending=False)
            w = sorted_g.iloc[0]["batting_team"]
            margin = int(sorted_g.iloc[0]["total_runs"] - sorted_g.iloc[1]["total_runs"])
            winners[mid] = w
            targets[mid] = int(sorted_g.iloc[1]["total_runs"]) + 1
            win_margins[mid] = margin
            win_types[mid] = "runs"
        elif len(grp) == 1:
            winners[mid] = grp.iloc[0]["batting_team"]
            targets[mid] = int(grp.iloc[0]["total_runs"])
            win_margins[mid] = 10
            win_types[mid] = "runs"

    match_grp["winner"] = match_grp["match_id"].map(winners)
    match_grp["target_runs"] = match_grp["match_id"].map(targets)
    match_grp["win_margin"] = match_grp["match_id"].map(win_margins).fillna(10).astype(int)
    match_grp["win_type"] = match_grp["match_id"].map(win_types).fillna("runs")
    match_grp["toss_winner"] = match_grp["team1"]
    match_grp["toss_decision"] = "bat"
    match_grp["match_type"] = "T20"
    match_grp["city"] = "Tournament City"
    match_grp["player_of_match"] = None
    match_grp["target_overs"] = 20

    return match_grp

def build_downstream_tables_and_models(df_matches, deliveries, progress_callback=None):
    """Builds fact_player_match, dim_player, SQLite database, and retrains models."""
    if progress_callback:
        progress_callback(50, "Aggregating Player-Match performance metrics...")

    # Batting Aggregation
    bat_grp = deliveries.groupby(["match_id", "striker", "batting_team", "bowling_team"]).agg(
        runs_scored=("runs_off_bat", "sum"),
        balls_faced=("runs_off_bat", "count"),
        fours=("is_boundary_4", "sum"),
        sixes=("is_boundary_6", "sum"),
        runs_powerplay=("runs_off_bat", lambda x: x[deliveries.loc[x.index, "innings_phase"] == "Powerplay"].sum()),
        runs_middle=("runs_off_bat", lambda x: x[deliveries.loc[x.index, "innings_phase"] == "Middle"].sum()),
        runs_death=("runs_off_bat", lambda x: x[deliveries.loc[x.index, "innings_phase"] == "Death"].sum())
    ).reset_index().rename(columns={"striker": "player_name", "batting_team": "team", "bowling_team": "opponent"})

    dismissals = deliveries[deliveries["is_wicket"] == 1][["match_id", "player_dismissed"]].drop_duplicates()
    dismissals["is_dismissed"] = 1
    bat_grp = bat_grp.merge(dismissals, left_on=["match_id", "player_name"], right_on=["match_id", "player_dismissed"], how="left").fillna(0)
    bat_grp["batting_strike_rate"] = ((bat_grp["runs_scored"] / np.maximum(bat_grp["balls_faced"], 1)) * 100).round(2)

    # Bowling Aggregation
    bowl_grp = deliveries.groupby(["match_id", "bowler", "bowling_team", "batting_team"]).agg(
        balls_bowled=("is_legal_ball", "sum"),
        runs_conceded=("bowler_runs", "sum"),
        dot_balls_bowled=("is_dot_ball", "sum"),
        wickets_taken=("is_wicket", "sum")
    ).reset_index().rename(columns={"bowler": "player_name", "bowling_team": "team", "batting_team": "opponent"})

    bowl_grp["overs_bowled"] = (bowl_grp["balls_bowled"] // 6) + ((bowl_grp["balls_bowled"] % 6) / 10.0)
    bowl_grp["bowling_economy"] = ((bowl_grp["runs_conceded"] / np.maximum(bowl_grp["balls_bowled"] / 6.0, 0.1667))).round(2)

    player_match = pd.merge(bat_grp, bowl_grp, on=["match_id", "player_name", "team", "opponent"], how="outer").fillna(0)
    match_meta = df_matches.set_index("match_id")[["season", "venue", "match_date"]].to_dict("index")
    player_match["season"] = player_match["match_id"].map(lambda mid: match_meta.get(mid, {}).get("season", "Current"))
    player_match["match_date"] = player_match["match_id"].map(lambda mid: match_meta.get(mid, {}).get("match_date", "2026-01-01"))
    player_match["venue"] = player_match["match_id"].map(lambda mid: match_meta.get(mid, {}).get("venue", "Standard Stadium"))
    player_match["player_match_id"] = np.arange(1, len(player_match) + 1)

    # DimPlayer
    p_summary = player_match.groupby("player_name").agg(
        total_runs=("runs_scored", "sum"),
        total_balls_faced=("balls_faced", "sum"),
        total_wickets=("wickets_taken", "sum"),
        total_balls_bowled=("balls_bowled", "sum"),
        matches_played=("match_id", "nunique")
    ).reset_index()

    def assign_role(row):
        r, w = row["total_runs"], row["total_wickets"]
        if w >= 15 and r >= 300: return "All-Rounder"
        elif w >= 10 or row["total_balls_bowled"] >= 60: return "Bowler"
        elif r >= 150 or row["total_balls_faced"] >= 100: return "Batter"
        return "Squad Player"

    p_summary["primary_role"] = p_summary.apply(assign_role, axis=1)
    p_summary["player_id"] = p_summary["player_name"].map(lambda p: f"reg_{hash(p)&0xFFFFFFF:07x}")
    p_summary["batting_hand"] = "Right-hand bat"
    p_summary["bowling_style"] = "Right-arm pace/spin"

    # DimTeam
    teams = sorted(list(set(df_matches["team1"].dropna()).union(set(df_matches["team2"].dropna()))))
    df_dim_team = pd.DataFrame([{"team_id": i + 1, "team_name": t, "team_code": t[:3].upper(), "current_status": "Active"} for i, t in enumerate(teams)])

    if progress_callback:
        progress_callback(75, "Rebuilding Relational SQLite Database...")

    # Write Database
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df_matches.to_sql("dim_match", conn, if_exists="replace", index=False)
    p_summary.to_sql("dim_player", conn, if_exists="replace", index=False)
    df_dim_team.to_sql("dim_team", conn, if_exists="replace", index=False)
    player_match.to_sql("fact_player_match", conn, if_exists="replace", index=False)
    deliveries.to_sql("fact_deliveries", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    # Save Processed CSVs
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_matches.to_csv(os.path.join(PROCESSED_DIR, "dim_match.csv"), index=False)
    p_summary.to_csv(os.path.join(PROCESSED_DIR, "dim_player.csv"), index=False)
    player_match.to_csv(os.path.join(PROCESSED_DIR, "fact_player_match.csv"), index=False)

    if progress_callback:
        progress_callback(90, "Retraining Predictive Models & Recalculating Role Ratings...")

    # Retrain Models
    try:
        from src.models.train_models import train_player_performance_model, train_match_win_probability_model, build_role_recruitment_system
    except ImportError:
        from models.train_models import train_player_performance_model, train_match_win_probability_model, build_role_recruitment_system
    train_player_performance_model()
    train_match_win_probability_model()
    build_role_recruitment_system()

    if progress_callback:
        progress_callback(100, "All models, databases, and dashboards refreshed!")

    return {
        "matches_count": len(df_matches),
        "deliveries_count": len(deliveries),
        "players_count": len(p_summary),
        "teams_count": len(df_dim_team),
        "seasons_covered": sorted(df_matches["season"].dropna().unique().tolist())
    }

def process_arbitrary_csv_bytes(csv_bytes, progress_callback=None):
    """
    Ingests ANY uploaded CSV file, repairs missing columns, synthesizes matches,
    and deploys everything automatically.
    """
    if progress_callback:
        progress_callback(15, "Inspecting and normalizing uploaded CSV...")
        
    df_raw = pd.read_csv(io.BytesIO(csv_bytes))
    deliveries, repair_log = heal_and_standardize_dataframe(df_raw)
    
    if progress_callback:
        progress_callback(35, f"Repairs completed ({len(repair_log)} adjustments applied). Synthesizing match standings...")
        
    df_matches = synthesize_matches_table(deliveries)
    stats = build_downstream_tables_and_models(df_matches, deliveries, progress_callback)
    stats["repair_log"] = repair_log
    return stats

# Alias for backwards compatibility
process_unified_csv = process_arbitrary_csv_bytes

def process_cricsheet_zip_bytes(zip_bytes, progress_callback=None):
    """Parses Cricsheet ZIP package."""
    if progress_callback:
        progress_callback(10, "Extracting and analyzing ZIP package...")
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    all_files = set(z.namelist())
    info_files = [f for f in all_files if f.endswith("_info.csv")]

    matches_list = []
    players_reg = {}

    for f in info_files:
        mid_str = f.replace("_info.csv", "")
        try: mid = int(mid_str)
        except ValueError: continue
        content = z.read(f).decode("utf-8", errors="ignore").splitlines()
        m_info = {"match_id": mid, "season": "2026", "match_date": "2026-01-01", "venue": "Stadium", "team1": "A", "team2": "B", "toss_winner": "A", "toss_decision": "bat", "winner": "A", "win_type": "runs", "win_margin": 10, "target_runs": 180, "target_overs": 20, "match_type": "T20", "city": "City"}
        teams = []
        for line in content:
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) < 3: continue
            k, v = parts[1], parts[2]
            if k == "season": m_info["season"] = str(v)
            elif k == "date": m_info["match_date"] = v.replace("/", "-")
            elif k == "venue": m_info["venue"] = v
            elif k == "team": teams.append(v)
            elif k == "winner": m_info["winner"] = v
            elif k == "target_runs" and len(parts) >= 4:
                try: m_info["target_runs"] = int(parts[3])
                except ValueError: pass
        if len(teams) >= 2:
            m_info["team1"], m_info["team2"] = teams[0], teams[1]
        matches_list.append(m_info)

    df_matches = pd.DataFrame(matches_list)
    deliv_frames = []
    for mid in df_matches["match_id"]:
        fn = f"{mid}.csv"
        if fn in all_files:
            try:
                b = z.read(fn)
                deliv_frames.append(pd.read_csv(io.BytesIO(b)))
            except Exception: pass

    if not deliv_frames:
        raise ValueError("No delivery CSVs found in zip.")

    deliveries_raw = pd.concat(deliv_frames, ignore_index=True)
    deliveries, repair_log = heal_and_standardize_dataframe(deliveries_raw)
    stats = build_downstream_tables_and_models(df_matches, deliveries, progress_callback)
    stats["repair_log"] = repair_log
    return stats

def download_and_ingest_preset(preset_name, progress_callback=None):
    """Downloads preset zip and triggers ingestion."""
    url = PRESET_DATASETS.get(preset_name)
    if not url:
        raise ValueError(f"Unknown preset: {preset_name}")
    if progress_callback:
        progress_callback(5, f"Downloading official archive for {preset_name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        zip_bytes = resp.read()
    return process_cricsheet_zip_bytes(zip_bytes, progress_callback)
