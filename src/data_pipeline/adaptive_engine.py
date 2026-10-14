"""
Beyond the Scoreboard: Cricket Franchise Decision Intelligence System
Universal Adaptive Dataset Classifier, Automated EDA & Multi-Schema Analytics Engine

Supports ANY Cricket or Tabular Dataset uploaded by the user:
1. Batting Summaries (Runs, Balls, SR, Fours, Sixes, Team, Average, etc. - NO BOWLER REQUIRED)
2. Bowling Summaries (Overs, Wickets, Economy, Maidens, Runs Conceded, Dots, etc. - NO BATTER REQUIRED)
3. All-Rounder Summaries (Both Batting & Bowling Metrics)
4. Match Summaries (Team1, Team2, Winner, Toss, Margin, Venue, etc.)
5. Points Tables / Standings (Team, Played, Won, Lost, Points, NRR)
6. Ball-by-Ball Granular Logs (Standard multi-match deliveries)
7. Custom / Generic Tabular Datasets (Arbitrary columns, Automated EDA, AutoML & Segmentation)
"""

import os
import io
import json
import re
import sqlite3
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "src", "models", "saved_models")
DB_PATH = os.path.join(BASE_DIR, "data", "cricket_analytics.db")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

ACTIVE_DATASET_PATH = os.path.join(PROCESSED_DIR, "active_dataset.csv")
ACTIVE_META_PATH = os.path.join(PROCESSED_DIR, "active_metadata.json")


def detect_dataset_type(df):
    """
    Intelligently classifies ANY cricket or tabular dataset without requiring missing columns.
    Never throws an error if columns are missing.
    """
    cols_lower = [str(c).strip().lower() for c in df.columns]

    # 1. Check for Ball-by-Ball
    has_ball = any(c in cols_lower for c in ["ball", "delivery", "actual_delivery"])
    has_over = any(c in cols_lower for c in ["over", "over_number"])
    has_striker = any(c in cols_lower for c in ["striker", "batsman", "batter"])
    has_bowler = any(c in cols_lower for c in ["bowler", "bowling_player"])
    if (has_ball or has_over) and (has_striker or has_bowler) and len(df) > 100:
        return "BALL_BY_BALL"

    # 2. Check for Points Table / Standings
    has_pts = any(c in cols_lower for c in ["points", "pts", "point"])
    has_nrr = any(c in cols_lower for c in ["nrr", "net_run_rate", "net_rr"])
    has_won = any(c in cols_lower for c in ["won", "win", "wins", "w"])
    if has_pts and (has_nrr or has_won) and len(df) <= 30:
        return "POINTS_TABLE"

    # 3. Check for Match Summary
    match_keywords = ["winner", "team1", "team2", "toss_winner", "margin", "win_by", "venue", "city", "player_of_match", "pom"]
    match_matches = sum(1 for k in match_keywords if any(k in c for c in cols_lower))
    if match_matches >= 2 and not any(c in cols_lower for c in ["strike_rate", "economy", "overs"]):
        return "MATCH_SUMMARY"

    # 4. Check for Batting & Bowling Keywords
    pure_bat_cols = [c for c in cols_lower if not any(b in c for b in ["bowl", "conceded", "wkt", "wicket", "overs", "econ"])]
    pure_bowl_cols = [c for c in cols_lower if not any(b in c for b in ["bat", "faced", "fours", "sixes", "boundary", "highest_score"])]

    bat_keywords = ["runs", "balls", "sr", "strike_rate", "fours", "sixes", "avg", "average", "highest_score", "hs", "50", "100", "not_out"]
    bat_matches = sum(1 for k in bat_keywords if any(k in c for c in pure_bat_cols))

    bowl_keywords = ["wickets", "overs", "economy", "econ", "maidens", "dots", "bowling_avg", "runs_conceded", "wkt", "wkts", "bowling_sr"]
    bowl_matches = sum(1 for k in bowl_keywords if any(k in c for c in pure_bowl_cols))

    # All-Rounder summary check: must have distinct batting volume AND bowling volume columns
    has_bat_vol = any(c in pure_bat_cols for c in ["runs", "balls", "fours", "sixes", "strike_rate"])
    has_bowl_vol = any(c in pure_bowl_cols for c in ["wickets", "overs", "economy", "maidens"])

    if has_bat_vol and has_bowl_vol and bat_matches >= 2 and bowl_matches >= 2:
        return "ALL_ROUNDER_SUMMARY"
    elif bat_matches >= 2 and bat_matches > bowl_matches:
        return "BATTING_SUMMARY"
    elif bowl_matches >= 2 and bowl_matches > bat_matches:
        return "BOWLING_SUMMARY"

    # Fallback heuristics
    if any("run" in c for c in pure_bat_cols) and any("ball" in c for c in pure_bat_cols):
        return "BATTING_SUMMARY"
    elif any("wkt" in c or "wicket" in c or "overs" in c for c in pure_bowl_cols):
        return "BOWLING_SUMMARY"
    elif any("win" in c for c in cols_lower) and any("team" in c for c in cols_lower):
        return "MATCH_SUMMARY"

    return "CUSTOM_DATASET"


def perform_eda_and_clean_dataset(df, dataset_type):
    """
    Performs comprehensive automated Exploratory Data Analysis (EDA),
    handles missing values via intelligent imputation,
    engineers domain features, trains predictive models,
    and registers the dataset in SQLite.
    """
    df = df.copy()
    initial_shape = df.shape

    # 1. Audit Missing Values
    missing_counts = df.isnull().sum()
    missing_report = {}
    for col, count in missing_counts.items():
        if count > 0:
            missing_report[col] = {
                "missing_count": int(count),
                "missing_pct": round(float(count) / len(df) * 100, 2)
            }

    # 2. Impute Missing Values without complaining
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Numeric columns: fill NaNs with median or 0
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)

    # Categorical columns: fill NaNs with 'Unknown'
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
            df[col] = df[col].fillna(fill_val).astype(str).str.strip()

    # 3. Compute Statistical Summary for EDA
    summary_stats = {}
    for col in numeric_cols:
        summary_stats[col] = {
            "mean": round(float(df[col].mean()), 2),
            "std": round(float(df[col].std()), 2) if len(df) > 1 else 0.0,
            "median": round(float(df[col].median()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2)
        }

    # Compute correlation matrix if multiple numeric columns exist
    corr_matrix = {}
    if len(numeric_cols) >= 2:
        cm = df[numeric_cols].corr().fillna(0).round(3)
        corr_matrix = cm.to_dict()

    ml_report = {}

    # Helper for bulletproof numeric extraction
    def to_num(c, default=0.0):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(default)
        if isinstance(default, pd.Series):
            return default
        return pd.Series([default] * len(df), index=df.index)

    # =========================================================================
    # 4. TYPE-SPECIFIC NORMALIZATION, FEATURE ENGINEERING & PREDICTIVE MODELS
    # =========================================================================

    # --- CASE A: BATTING SUMMARY ---
    if dataset_type == "BATTING_SUMMARY":
        col_map = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if any(cl == k for k in ["player", "player_name", "batsman", "batter", "name"]): col_map[c] = "player_name"
            elif any(cl == k for k in ["team", "franchise", "squad"]): col_map[c] = "team"
            elif any(cl == k for k in ["runs", "total_runs", "score", "r"]): col_map[c] = "runs"
            elif any(cl == k for k in ["balls", "balls_faced", "bf", "b"]): col_map[c] = "balls"
            elif any(cl == k for k in ["sr", "strike_rate", "strikerate"]): col_map[c] = "strike_rate"
            elif any(cl == k for k in ["avg", "average", "batting_avg"]): col_map[c] = "average"
            elif any(cl == k for k in ["fours", "4s", "4"]): col_map[c] = "fours"
            elif any(cl == k for k in ["sixes", "6s", "6"]): col_map[c] = "sixes"
            elif any(cl == k for k in ["matches", "m", "mat"]): col_map[c] = "matches"
            elif any(cl == k for k in ["innings", "inns", "inn"]): col_map[c] = "innings"
        df = df.rename(columns=col_map)

        if "player_name" not in df.columns: df["player_name"] = "Batter " + (df.index + 1).astype(str)
        if "team" not in df.columns: df["team"] = "Franchise"
        df["runs"] = to_num("runs", 0).astype(int)
        df["balls"] = to_num("balls", np.maximum(1, (df["runs"] * 0.75).astype(int))).astype(int)

        if "strike_rate" in df.columns and not df["strike_rate"].isnull().all():
            df["strike_rate"] = to_num("strike_rate", 120.0).round(2)
        else:
            df["strike_rate"] = ((df["runs"] / np.maximum(df["balls"], 1)) * 100).round(2)

        df["fours"] = to_num("fours", 0).astype(int)
        df["sixes"] = to_num("sixes", 0).astype(int)

        df["boundary_runs"] = (df["fours"] * 4) + (df["sixes"] * 6)
        df["boundary_pct"] = np.where(df["runs"] > 0, ((df["boundary_runs"] / df["runs"]) * 100).round(1), 0.0)

        def assign_bat_role(row):
            sr = row["strike_rate"]
            b_pct = row["boundary_pct"]
            runs = row["runs"]
            if sr >= 165 and b_pct >= 60: return "Death-Overs Finisher / Power Hitter"
            elif sr >= 140 and runs >= 200: return "Aggressive Powerplay Opener"
            elif sr >= 120 and runs >= 150: return "Middle-Order Anchor / Rotator"
            else: return "Supporting Batting Depth"
        df["tactical_role"] = df.apply(assign_bat_role, axis=1)

        # Train ML Model
        if len(df) >= 6 and "runs" in df.columns and "balls" in df.columns:
            feat_cols = [c for c in ["balls", "fours", "sixes", "strike_rate"] if c in df.columns]
            X = df[feat_cols]
            y = df["runs"]
            lr = LinearRegression()
            lr.fit(X, y)
            preds = lr.predict(X)
            ml_report = {
                "model_type": "Batting Contribution Regression",
                "target": "runs",
                "features": feat_cols,
                "mae": round(float(mean_absolute_error(y, preds)), 2),
                "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 2),
                "r2_score": round(float(r2_score(y, preds)), 4),
                "sample_size": len(df),
                "coefficients": dict(zip(feat_cols, [round(float(c), 3) for c in lr.coef_]))
            }
            joblib.dump({"model": lr, "features": feat_cols, "metrics": ml_report},
                        os.path.join(MODELS_DIR, "adaptive_batting_model.joblib"))

    # --- CASE B: BOWLING SUMMARY ---
    elif dataset_type == "BOWLING_SUMMARY":
        col_map = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if any(cl == k for k in ["player", "player_name", "bowler", "name"]): col_map[c] = "player_name"
            elif any(cl == k for k in ["team", "franchise", "squad"]): col_map[c] = "team"
            elif any(cl == k for k in ["wickets", "wkts", "w", "wkt"]): col_map[c] = "wickets"
            elif any(cl == k for k in ["overs", "ov", "o"]): col_map[c] = "overs"
            elif any(cl == k for k in ["economy", "econ", "er"]): col_map[c] = "economy"
            elif any(cl == k for k in ["runs", "runs_conceded", "rc", "r"]): col_map[c] = "runs_conceded"
            elif any(cl == k for k in ["maidens", "m", "mdn"]): col_map[c] = "maidens"
            elif any(cl == k for k in ["dots", "dot_balls", "dot"]): col_map[c] = "dots"
            elif any(cl == k for k in ["avg", "bowling_avg", "average"]): col_map[c] = "bowling_avg"
            elif any(cl == k for k in ["sr", "strike_rate", "bowling_sr"]): col_map[c] = "bowling_sr"
        df = df.rename(columns=col_map)

        if "player_name" not in df.columns: df["player_name"] = "Bowler " + (df.index + 1).astype(str)
        if "team" not in df.columns: df["team"] = "Franchise"
        df["wickets"] = to_num("wickets", 0).astype(int)
        df["overs"] = to_num("overs", 1.0).astype(float)
        df["runs_conceded"] = to_num("runs_conceded", df["overs"] * 8.5).astype(float)
        df["maidens"] = to_num("maidens", 0).astype(int)

        if "economy" in df.columns and not df["economy"].isnull().all():
            df["economy"] = to_num("economy", 8.5).round(2)
        else:
            df["economy"] = np.where(df["overs"] > 0, (df["runs_conceded"] / df["overs"]).round(2), 8.5)

        if "dots" in df.columns:
            df["dots"] = to_num("dots", 0).astype(int)
            df["dot_ball_pct"] = np.where(df["overs"] > 0, ((df["dots"] / (df["overs"] * 6)) * 100).round(1), 35.0)
        else:
            df["dot_ball_pct"] = np.where(df["economy"] < 8.0, 45.0, 32.0)

        def assign_bowl_role(row):
            econ = row["economy"]
            wkts = row["wickets"]
            if econ <= 7.2 and wkts >= 10: return "Elite All-Phase Spearhead"
            elif econ <= 7.5: return "Powerplay / Lockdown Specialist"
            elif wkts >= 12 and econ >= 8.2: return "Attacking Death-Overs Wicket-Taker"
            elif econ <= 8.5 and wkts >= 6: return "Middle-Overs Control Enforcer"
            else: return "Containment / Support Bowler"
        df["tactical_role"] = df.apply(assign_bowl_role, axis=1)

        # Train ML Model for Bowlers (Predict Wickets or Economy)
        if len(df) >= 6:
            feat_cols = [c for c in ["overs", "runs_conceded", "economy", "maidens"] if c in df.columns]
            X = df[feat_cols]
            y = df["wickets"]
            lr = LinearRegression()
            lr.fit(X, y)
            preds = lr.predict(X)
            ml_report = {
                "model_type": "Bowling Wicket-Taking Efficiency Regression",
                "target": "wickets",
                "features": feat_cols,
                "mae": round(float(mean_absolute_error(y, preds)), 2),
                "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 2),
                "r2_score": round(float(r2_score(y, preds)), 4),
                "sample_size": len(df),
                "coefficients": dict(zip(feat_cols, [round(float(c), 3) for c in lr.coef_]))
            }
            joblib.dump({"model": lr, "features": feat_cols, "metrics": ml_report},
                        os.path.join(MODELS_DIR, "adaptive_bowling_model.joblib"))

    # --- CASE C: ALL-ROUNDER SUMMARY ---
    elif dataset_type == "ALL_ROUNDER_SUMMARY":
        col_map = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if any(cl == k for k in ["player", "player_name", "all_rounder", "name"]): col_map[c] = "player_name"
            elif any(cl == k for k in ["team", "franchise"]): col_map[c] = "team"
            elif any(cl == k for k in ["runs", "r"]): col_map[c] = "runs"
            elif any(cl == k for k in ["wickets", "wkts", "w"]): col_map[c] = "wickets"
            elif any(cl == k for k in ["sr", "strike_rate"]): col_map[c] = "strike_rate"
            elif any(cl == k for k in ["economy", "econ"]): col_map[c] = "economy"
        df = df.rename(columns=col_map)

        if "player_name" not in df.columns: df["player_name"] = "Player " + (df.index + 1).astype(str)
        if "team" not in df.columns: df["team"] = "Franchise"
        df["runs"] = to_num("runs", 0).astype(int)
        df["wickets"] = to_num("wickets", 0).astype(int)
        df["strike_rate"] = to_num("strike_rate", 125.0).round(2)
        df["economy"] = to_num("economy", 8.2).round(2)

        # Composite All-Rounder Index (AR Index)
        df["all_rounder_index"] = (
            (df["runs"] * (df["strike_rate"] / 100.0) * 0.15) +
            (df["wickets"] * 25.0) -
            (np.maximum(0, df["economy"] - 7.0) * 8.0)
        ).round(1)

        def assign_ar_role(row):
            idx = row["all_rounder_index"]
            r = row["runs"]
            w = row["wickets"]
            if idx >= 300 or (r >= 250 and w >= 10): return "Marquee Core All-Rounder"
            elif w >= 12: return "Bowling All-Rounder"
            elif r >= 250: return "Batting All-Rounder"
            else: return "Utility Balance Player"
        df["tactical_role"] = df.apply(assign_ar_role, axis=1)

        # Train ML Model
        if len(df) >= 6:
            X = df[["runs", "wickets", "strike_rate", "economy"]]
            y = df["all_rounder_index"]
            lr = LinearRegression()
            lr.fit(X, y)
            preds = lr.predict(X)
            ml_report = {
                "model_type": "Composite All-Rounder Utility Model",
                "target": "all_rounder_index",
                "features": ["runs", "wickets", "strike_rate", "economy"],
                "mae": round(float(mean_absolute_error(y, preds)), 2),
                "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 2),
                "r2_score": round(float(r2_score(y, preds)), 4),
                "sample_size": len(df)
            }
            joblib.dump({"model": lr, "features": ["runs", "wickets", "strike_rate", "economy"], "metrics": ml_report},
                        os.path.join(MODELS_DIR, "adaptive_allrounder_model.joblib"))

    # --- CASE D: MATCH SUMMARY ---
    elif dataset_type == "MATCH_SUMMARY":
        col_map = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if any(cl == k for k in ["winner", "winning_team"]): col_map[c] = "winner"
            elif any(cl == k for k in ["team1", "team_1"]): col_map[c] = "team1"
            elif any(cl == k for k in ["team2", "team_2"]): col_map[c] = "team2"
            elif any(cl == k for k in ["toss_winner"]): col_map[c] = "toss_winner"
            elif any(cl == k for k in ["toss_decision"]): col_map[c] = "toss_decision"
            elif any(cl == k for k in ["margin", "win_by_runs", "win_margin"]): col_map[c] = "margin"
            elif any(cl == k for k in ["venue", "stadium"]): col_map[c] = "venue"
            elif any(cl == k for k in ["city"]): col_map[c] = "city"
        df = df.rename(columns=col_map)

        if "winner" not in df.columns and "team1" in df.columns:
            df["winner"] = df["team1"]
        if "toss_winner" in df.columns and "winner" in df.columns:
            df["toss_match_win"] = (df["toss_winner"] == df["winner"]).astype(int)

        ml_report = {
            "model_type": "Match Diagnostic Profiler",
            "sample_size": len(df),
            "total_matches": len(df),
            "unique_winners": df["winner"].nunique() if "winner" in df.columns else 0
        }

    # --- CASE E: POINTS TABLE ---
    elif dataset_type == "POINTS_TABLE":
        col_map = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if any(cl == k for k in ["team", "squad", "franchise"]): col_map[c] = "team"
            elif any(cl == k for k in ["matches", "played", "p", "mat"]): col_map[c] = "played"
            elif any(cl == k for k in ["won", "w", "wins"]): col_map[c] = "won"
            elif any(cl == k for k in ["lost", "l", "losses"]): col_map[c] = "lost"
            elif any(cl == k for k in ["points", "pts", "point"]): col_map[c] = "points"
            elif any(cl == k for k in ["nrr", "net_run_rate"]): col_map[c] = "nrr"
        df = df.rename(columns=col_map)

        if "team" not in df.columns: df["team"] = "Team " + (df.index + 1).astype(str)
        df["won"] = to_num("won", 0).astype(int)
        df["played"] = to_num("played", df["won"]).astype(int)
        df["points"] = to_num("points", df["won"] * 2).astype(int)
        df["nrr"] = to_num("nrr", 0.0).astype(float)
        df["win_pct"] = np.where(df["played"] > 0, ((df["won"] / df["played"]) * 100).round(1), 0.0)

        # Tactical qualification tier
        df["playoff_tier"] = np.where(df.index < 4, "Top-4 Playoff Contender", "Elimination Zone")

        ml_report = {
            "model_type": "Championship Qualification Profiler",
            "teams_count": len(df),
            "leader": df.sort_values("points", ascending=False).iloc[0]["team"] if not df.empty else "None"
        }

    # --- CASE F: CUSTOM / GENERIC TABULAR DATASET ---
    else:
        # Determine primary numeric column (highest variance)
        if len(numeric_cols) > 0:
            var_series = df[numeric_cols].var()
            primary_col = var_series.idxmax() if not var_series.empty else numeric_cols[0]
            feature_cols = [c for c in numeric_cols if c != primary_col]

            # Assign Performance Tiers based on primary numeric column
            q75 = df[primary_col].quantile(0.75)
            q25 = df[primary_col].quantile(0.25)
            def assign_tier(val):
                if val >= q75: return "Tier 1: High-Performance / Elite"
                elif val >= q25: return "Tier 2: Core Consistent Contributor"
                else: return "Tier 3: Developmental / Support Depth"
            df["tactical_role"] = df[primary_col].apply(assign_tier)

            # Train AutoML Regression
            if len(feature_cols) >= 1 and len(df) >= 6:
                X = df[feature_cols]
                y = df[primary_col]
                lr = Ridge(alpha=1.0)
                lr.fit(X, y)
                preds = lr.predict(X)
                ml_report = {
                    "model_type": f"AutoML Linear Ridge Regression (Target: {primary_col})",
                    "target": primary_col,
                    "features": feature_cols,
                    "mae": round(float(mean_absolute_error(y, preds)), 2),
                    "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 2),
                    "r2_score": round(float(r2_score(y, preds)), 4),
                    "sample_size": len(df),
                    "coefficients": dict(zip(feature_cols, [round(float(c), 3) for c in lr.coef_]))
                }
                joblib.dump({"model": lr, "features": feature_cols, "metrics": ml_report},
                            os.path.join(MODELS_DIR, "adaptive_custom_model.joblib"))

    # =========================================================================
    # 5. REGISTER UPLOADED DATASET IN SQLITE FOR SQL STUDIO QUERYING
    # =========================================================================
    try:
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("uploaded_dataset", conn, if_exists="replace", index=False)
        conn.close()
    except Exception as e:
        print(f"Notice: SQLite registration: {e}")

    # =========================================================================
    # 6. PERSIST PROCESSED DATASET & METADATA
    # =========================================================================
    df.to_csv(ACTIVE_DATASET_PATH, index=False)

    meta = {
        "dataset_type": dataset_type,
        "rows_count": len(df),
        "columns_count": len(df.columns),
        "columns_list": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude=[np.number]).columns.tolist(),
        "missing_values_report": missing_report,
        "summary_stats": summary_stats,
        "correlation_matrix": corr_matrix,
        "ml_report": ml_report,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(ACTIVE_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    return df, meta


def load_active_dataset():
    """Loads current active dataset and metadata if present."""
    if os.path.exists(ACTIVE_DATASET_PATH) and os.path.exists(ACTIVE_META_PATH):
        try:
            df = pd.read_csv(ACTIVE_DATASET_PATH)
            with open(ACTIVE_META_PATH, "r") as f:
                meta = json.load(f)
            return df, meta
        except Exception:
            return None, None
    return None, None
