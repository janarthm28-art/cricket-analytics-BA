"""
Beyond the Scoreboard: Cricket Franchise Decision Intelligence System
Predictive Analytics and Machine Learning Training Engine
1. Model 1: Player Performance Forecaster (Next Match Batting Runs)
2. Model 2: In-Game Match Win Probability Classifier
3. Model 3: Role-Based Recruitment Decision Engine
"""

import os
import sys
import sqlite3
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "cricket_analytics.db")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "src", "models", "saved_models")

os.makedirs(MODELS_DIR, exist_ok=True)

def train_player_performance_model():
    print("\n" + "=" * 70)
    print("TRAINING MODEL 1: PLAYER PERFORMANCE FORECASTING (BATTING RUNS)")
    print("=" * 70)
    
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        player_match_id, match_id, season, match_date, player_name, team, opponent, venue,
        runs_scored, balls_faced, fours, sixes, batting_strike_rate, is_dismissed
    FROM fact_player_match
    WHERE balls_faced > 0
    ORDER BY match_date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df["match_date"] = pd.to_datetime(df["match_date"])
    print(f"[+] Loaded {len(df):,} batting innings records.")

    # Engineer Lagged Rolling Features (STRICTLY NO DATA LEAKAGE: shift(1))
    print("[+] Engineering lagged rolling features per player...")
    records = []
    
    for player, group in df.groupby("player_name"):
        if len(group) < 5:
            continue
            
        group = group.sort_values("match_date").copy()
        
        # Lagged rolling features
        group["rolling_runs_3"] = group["runs_scored"].shift(1).rolling(3, min_periods=1).mean()
        group["rolling_runs_5"] = group["runs_scored"].shift(1).rolling(5, min_periods=2).mean()
        group["rolling_runs_10"] = group["runs_scored"].shift(1).rolling(10, min_periods=3).mean()
        group["rolling_sr_5"] = group["batting_strike_rate"].shift(1).rolling(5, min_periods=2).mean()
        group["rolling_dismissal_rate_5"] = group["is_dismissed"].shift(1).rolling(5, min_periods=2).mean()
        group["career_innings_count"] = np.arange(len(group))
        
        # Target: actual runs scored in this match
        records.append(group)

    feat_df = pd.concat(records, ignore_index=True)
    feat_df = feat_df.dropna(subset=["rolling_runs_5", "rolling_sr_5"]).reset_index(drop=True)
    print(f"[+] Formed {len(feat_df):,} validated instances for player forecasting.")

    # Time-based Train / Test Split (Chronological)
    # Train: Seasons prior to 2023, Test: 2023-2026
    train_mask = feat_df["season"].astype(str) < "2023"
    test_mask = feat_df["season"].astype(str) >= "2023"
    
    feature_cols = [
        "rolling_runs_3", "rolling_runs_5", "rolling_runs_10", 
        "rolling_sr_5", "rolling_dismissal_rate_5", "career_innings_count"
    ]
    target_col = "runs_scored"

    X_train, y_train = feat_df.loc[train_mask, feature_cols], feat_df.loc[train_mask, target_col]
    X_test, y_test = feat_df.loc[test_mask, feature_cols], feat_df.loc[test_mask, target_col]

    print(f"[+] Train split: {len(X_train):,} instances | Test split: {len(X_test):,} instances.")

    # Handle any remaining NaNs in rolling features with median imputation
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

    # Baseline 1: Historical 5-match moving average
    baseline_pred = X_test_imp["rolling_runs_5"]
    base_mae = mean_absolute_error(y_test, baseline_pred)
    base_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    print(f"[+] BASELINE (Rolling 5-Match MA) -> MAE: {base_mae:.2f} | RMSE: {base_rmse:.2f}")

    # Model 1A: Ridge Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
    print(f"[+] MODEL 1A: Ridge Regression -> MAE: {ridge_mae:.2f} | RMSE: {ridge_rmse:.2f}")

    # Model 1B: Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    gbr.fit(X_train_imp, y_train)
    gbr_pred = gbr.predict(X_test_imp)
    gbr_mae = mean_absolute_error(y_test, gbr_pred)
    gbr_rmse = np.sqrt(mean_squared_error(y_test, gbr_pred))
    print(f"[+] MODEL 1B: Gradient Boosting Regressor -> MAE: {gbr_mae:.2f} | RMSE: {gbr_rmse:.2f}")

    # Save artifacts
    model_payload = {
        "model": gbr,
        "scaler": scaler,
        "imputer": imputer,
        "feature_cols": feature_cols,
        "metrics": {
            "baseline_mae": round(base_mae, 2),
            "baseline_rmse": round(base_rmse, 2),
            "ridge_mae": round(ridge_mae, 2),
            "ridge_rmse": round(ridge_rmse, 2),
            "gbr_mae": round(gbr_mae, 2),
            "gbr_rmse": round(gbr_rmse, 2),
        },
        "feature_importances": dict(zip(feature_cols, gbr.feature_importances_))
    }
    joblib.dump(model_payload, os.path.join(MODELS_DIR, "player_forecast_model.joblib"))
    print("[+] Model 1 saved to disk.")
    return model_payload

def train_match_win_probability_model():
    print("\n" + "=" * 70)
    print("TRAINING MODEL 2: IN-GAME MATCH WIN PROBABILITY CLASSIFIER")
    print("=" * 70)
    
    conn = sqlite3.connect(DB_PATH)
    # Query over-by-over progression in 2nd innings (chase)
    query = """
    SELECT 
        d.match_id, d.season, d.innings, d.over_number, d.batting_team, d.bowling_team,
        d.total_runs, d.is_wicket, m.winner, m.target_runs
    FROM fact_deliveries d
    JOIN dim_match m ON d.match_id = m.match_id
    WHERE d.innings = 2 AND m.winner IS NOT NULL AND m.winner != '' AND m.target_runs IS NOT NULL
    ORDER BY d.match_id, d.over_number, d.ball_number
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"[+] Loaded {len(df):,} chase delivery events.")

    # Group by match and over to create over-state snapshot
    over_states = df.groupby(["match_id", "season", "over_number", "batting_team", "bowling_team", "winner", "target_runs"]).agg(
        runs_in_over=("total_runs", "sum"),
        wickets_in_over=("is_wicket", "sum")
    ).reset_index()

    # Calculate cumulative runs and cumulative wickets
    over_states["cum_runs"] = over_states.groupby("match_id")["runs_in_over"].cumsum()
    over_states["cum_wickets"] = over_states.groupby("match_id")["wickets_in_over"].cumsum()
    over_states["overs_completed"] = over_states["over_number"] + 1
    over_states["overs_remaining"] = 20 - over_states["overs_completed"]
    over_states["balls_remaining"] = over_states["overs_remaining"] * 6
    over_states["runs_remaining"] = np.maximum(over_states["target_runs"] - over_states["cum_runs"], 0)
    over_states["current_rr"] = (over_states["cum_runs"] / over_states["overs_completed"]).round(2)
    over_states["required_rr"] = np.where(
        over_states["overs_remaining"] > 0,
        (over_states["runs_remaining"] / over_states["overs_remaining"]).round(2),
        over_states["runs_remaining"]
    )
    over_states["wickets_in_hand"] = 10 - np.minimum(over_states["cum_wickets"], 10)
    
    # Target: 1 if batting_team won, 0 otherwise
    over_states["target_win"] = (over_states["batting_team"] == over_states["winner"]).astype(int)

    feature_cols = [
        "overs_completed", "runs_remaining", "balls_remaining", 
        "wickets_in_hand", "current_rr", "required_rr"
    ]
    
    # Filter out extreme anomalies
    over_states = over_states[(over_states["required_rr"] <= 36) & (over_states["required_rr"] >= 0)]

    # Chronological Split
    train_mask = over_states["season"].astype(str) < "2023"
    test_mask = over_states["season"].astype(str) >= "2023"

    X_train = over_states.loc[train_mask, feature_cols]
    y_train = over_states.loc[train_mask, "target_win"]
    X_test = over_states.loc[test_mask, feature_cols]
    y_test = over_states.loc[test_mask, "target_win"]

    print(f"[+] Train states: {len(X_train):,} | Test states: {len(X_test):,}")

    # Baseline: Majority Class Classifier
    majority_prob = y_train.mean()
    base_brier = brier_score_loss(y_test, np.full_like(y_test, majority_prob, dtype=float))
    print(f"[+] BASELINE (Pre-match win frequency {majority_prob:.1%}) -> Brier Score: {base_brier:.4f}")

    # Model 2A: Calibrated Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression()
    log_reg.fit(X_train_scaled, y_train)
    lr_probs = log_reg.predict_proba(X_test_scaled)[:, 1]
    
    lr_brier = brier_score_loss(y_test, lr_probs)
    lr_logloss = log_loss(y_test, lr_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    print(f"[+] MODEL 2A: Calibrated Logistic Regression -> Brier: {lr_brier:.4f} | LogLoss: {lr_logloss:.4f} | ROC-AUC: {lr_auc:.4f}")

    # Model 2B: Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    rf_brier = brier_score_loss(y_test, rf_probs)
    rf_logloss = log_loss(y_test, rf_probs)
    rf_auc = roc_auc_score(y_test, rf_probs)
    print(f"[+] MODEL 2B: Random Forest Classifier -> Brier: {rf_brier:.4f} | LogLoss: {rf_logloss:.4f} | ROC-AUC: {rf_auc:.4f}")

    model_payload = {
        "model": rf,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics": {
            "baseline_brier": round(base_brier, 4),
            "lr_brier": round(lr_brier, 4),
            "lr_logloss": round(lr_logloss, 4),
            "lr_auc": round(lr_auc, 4),
            "rf_brier": round(rf_brier, 4),
            "rf_logloss": round(rf_logloss, 4),
            "rf_auc": round(rf_auc, 4),
        },
        "feature_importances": dict(zip(feature_cols, rf.feature_importances_))
    }
    joblib.dump(model_payload, os.path.join(MODELS_DIR, "match_win_model.joblib"))
    print("[+] Model 2 saved to disk.")
    return model_payload

def build_role_recruitment_system():
    print("\n" + "=" * 70)
    print("BUILDING MODEL 3: ROLE-BASED RECRUITMENT DECISION SUPPORT ENGINE")
    print("=" * 70)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Batting by Phase
    q_bat = """
    SELECT 
        striker AS player_name,
        innings_phase,
        SUM(runs_off_bat) AS phase_runs,
        COUNT(*) AS phase_balls,
        SUM(is_dot_ball) AS phase_dots,
        SUM(is_boundary_4 + is_boundary_6) AS phase_boundaries,
        SUM(is_wicket) AS phase_dismissals
    FROM fact_deliveries
    GROUP BY striker, innings_phase
    """
    df_bat_phase = pd.read_sql_query(q_bat, conn)

    # 2. Bowling by Phase
    q_bowl = """
    SELECT 
        bowler AS player_name,
        innings_phase,
        SUM(bowler_runs) AS phase_runs_conceded,
        SUM(is_legal_ball) AS phase_balls_bowled,
        SUM(is_dot_ball) AS phase_dots_bowled,
        SUM(CASE WHEN is_wicket = 1 AND wicket_type NOT IN ('run out', 'retired hurt') THEN 1 ELSE 0 END) AS phase_wickets
    FROM fact_deliveries
    GROUP BY bowler, innings_phase
    """
    df_bowl_phase = pd.read_sql_query(q_bowl, conn)

    # 3. Overall player aggregates
    q_player = """
    SELECT 
        p.player_name, p.primary_role,
        SUM(fpm.runs_scored) AS total_runs,
        SUM(fpm.balls_faced) AS total_balls,
        SUM(fpm.is_dismissed) AS total_outs,
        SUM(fpm.wickets_taken) AS total_wickets,
        SUM(fpm.balls_bowled) AS total_balls_bowled,
        SUM(fpm.runs_conceded) AS total_runs_conceded,
        COUNT(DISTINCT fpm.match_id) AS matches
    FROM dim_player p
    JOIN fact_player_match fpm ON p.player_name = fpm.player_name
    GROUP BY p.player_name
    """
    df_player = pd.read_sql_query(q_player, conn)
    conn.close()

    # Pivot phase batting
    pivot_bat = df_bat_phase.pivot_table(
        index="player_name", 
        columns="innings_phase", 
        values=["phase_runs", "phase_balls", "phase_dots", "phase_boundaries", "phase_dismissals"],
        fill_value=0
    )
    pivot_bat.columns = [f"{col[0]}_{col[1]}" for col in pivot_bat.columns]
    pivot_bat = pivot_bat.reset_index()

    # Pivot phase bowling
    pivot_bowl = df_bowl_phase.pivot_table(
        index="player_name",
        columns="innings_phase",
        values=["phase_runs_conceded", "phase_balls_bowled", "phase_dots_bowled", "phase_wickets"],
        fill_value=0
    )
    pivot_bowl.columns = [f"{col[0]}_{col[1]}" for col in pivot_bowl.columns]
    pivot_bowl = pivot_bowl.reset_index()

    merged = df_player.merge(pivot_bat, on="player_name", how="left").merge(pivot_bowl, on="player_name", how="left").fillna(0)

    # Derived Phase Rates
    # Powerplay Batter
    pp_balls = merged["phase_balls_Powerplay"]
    merged["pp_sr"] = np.where(pp_balls >= 60, (merged["phase_runs_Powerplay"] / pp_balls) * 100, np.nan)
    merged["pp_dot_pct"] = np.where(pp_balls >= 60, (merged["phase_dots_Powerplay"] / pp_balls) * 100, np.nan)
    merged["pp_boundary_pct"] = np.where(pp_balls >= 60, (merged["phase_boundaries_Powerplay"] / pp_balls) * 100, np.nan)

    # Middle-Order Batter
    mid_balls = merged["phase_balls_Middle"]
    merged["mid_sr"] = np.where(mid_balls >= 60, (merged["phase_runs_Middle"] / mid_balls) * 100, np.nan)
    merged["mid_avg"] = np.where(mid_balls >= 60, merged["phase_runs_Middle"] / np.maximum(merged["phase_dismissals_Middle"], 1), np.nan)

    # Finisher
    death_balls = merged["phase_balls_Death"]
    merged["death_sr"] = np.where(death_balls >= 40, (merged["phase_runs_Death"] / death_balls) * 100, np.nan)
    merged["death_boundary_pct"] = np.where(death_balls >= 40, (merged["phase_boundaries_Death"] / death_balls) * 100, np.nan)

    # Powerplay Bowler
    pp_bowl_balls = merged["phase_balls_bowled_Powerplay"]
    merged["pp_economy"] = np.where(pp_bowl_balls >= 60, (merged["phase_runs_conceded_Powerplay"] / (pp_bowl_balls / 6.0)), np.nan)
    merged["pp_dot_bowl_pct"] = np.where(pp_bowl_balls >= 60, (merged["phase_dots_bowled_Powerplay"] / pp_bowl_balls) * 100, np.nan)

    # Middle-Over Wicket-Taker
    mid_bowl_balls = merged["phase_balls_bowled_Middle"]
    merged["mid_sr_bowl"] = np.where((mid_bowl_balls >= 60) & (merged["phase_wickets_Middle"] > 0), mid_bowl_balls / merged["phase_wickets_Middle"], np.nan)
    merged["mid_economy"] = np.where(mid_bowl_balls >= 60, (merged["phase_runs_conceded_Middle"] / (mid_bowl_balls / 6.0)), np.nan)

    # Death Bowler
    death_bowl_balls = merged["phase_balls_bowled_Death"]
    merged["death_economy"] = np.where(death_bowl_balls >= 40, (merged["phase_runs_conceded_Death"] / (death_bowl_balls / 6.0)), np.nan)
    merged["death_wickets"] = merged["phase_wickets_Death"]

    # All-Rounder Composite
    merged["all_rounder_index"] = np.where(
        (merged["total_runs"] >= 300) & (merged["total_wickets"] >= 15),
        (merged["total_runs"] / np.maximum(merged["total_outs"], 1)) * 0.4 + (merged["total_wickets"] * 2.5),
        np.nan
    )

    # Standardize Role Scores (Z-score normalizer into 0-100 index)
    def z_to_100(series, ascending=True):
        valid = series.dropna()
        if len(valid) == 0:
            return pd.Series(index=series.index)
        mean, std = valid.mean(), valid.std()
        if std == 0:
            std = 1.0
        z = (series - mean) / std
        if not ascending:
            z = -z  # For metrics where lower is better, like economy
        # Min-max scale the z score to 40 - 99
        score = 70 + (z * 10)
        return score.clip(30, 99).round(1)

    merged["score_powerplay_batter"] = (z_to_100(merged["pp_sr"], True) * 0.5 + z_to_100(merged["pp_dot_pct"], False) * 0.5).round(1)
    merged["score_middle_batter"] = (z_to_100(merged["mid_avg"], True) * 0.5 + z_to_100(merged["mid_sr"], True) * 0.5).round(1)
    merged["score_finisher"] = (z_to_100(merged["death_sr"], True) * 0.6 + z_to_100(merged["death_boundary_pct"], True) * 0.4).round(1)
    merged["score_powerplay_bowler"] = (z_to_100(merged["pp_economy"], False) * 0.6 + z_to_100(merged["pp_dot_bowl_pct"], True) * 0.4).round(1)
    merged["score_middle_bowler"] = (z_to_100(merged["mid_sr_bowl"], False) * 0.6 + z_to_100(merged["mid_economy"], False) * 0.4).round(1)
    merged["score_death_bowler"] = (z_to_100(merged["death_economy"], False) * 0.6 + z_to_100(merged["death_wickets"], True) * 0.4).round(1)
    merged["score_all_rounder"] = z_to_100(merged["all_rounder_index"], True).round(1)

    # Save rankings
    recruitment_file = os.path.join(PROCESSED_DIR, "recruitment_rankings.csv")
    merged.to_csv(recruitment_file, index=False)
    print(f"[+] Role recruitment engine generated and saved to {recruitment_file} ({len(merged)} players).")

    # Save summary top candidates
    for role, col in [
        ("Powerplay Batter", "score_powerplay_batter"),
        ("Middle-Order Anchor", "score_middle_batter"),
        ("Death-Overs Finisher", "score_finisher"),
        ("Powerplay Bowler", "score_powerplay_bowler"),
        ("Middle-Overs Wicket Taker", "score_middle_bowler"),
        ("Death Specialist Bowler", "score_death_bowler"),
        ("Dual All-Rounder", "score_all_rounder")
    ]:
        top = merged.dropna(subset=[col]).sort_values(col, ascending=False).head(3)
        print(f"    - Top {role}: {', '.join(top['player_name'] + ' (' + top[col].astype(str) + ')')}")

    # Write predictions into fact_predictions table
    conn = sqlite3.connect(DB_PATH)
    records = []
    pid = 1
    for role, col in [
        ("Powerplay Batter", "score_powerplay_batter"),
        ("Finisher", "score_finisher"),
        ("Death Bowler", "score_death_bowler"),
        ("All-Rounder", "score_all_rounder")
    ]:
        top = merged.dropna(subset=[col]).sort_values(col, ascending=False).head(50)
        for _, row in top.iterrows():
            records.append({
                "prediction_id": pid,
                "model_name": "RoleRecruitmentScorer_v1",
                "model_type": "Ranking",
                "prediction_target": f"recruitment_score_{role.lower().replace(' ', '_')}",
                "entity_id": row["player_name"],
                "entity_type": "PLAYER",
                "season": "Historical",
                "predicted_value": float(row[col]),
                "prediction_prob": None,
                "actual_value": None,
                "residual_error": None,
                "evaluation_split": "ALL",
                "model_version": "1.0.0"
            })
            pid += 1

    df_preds = pd.DataFrame(records)
    df_preds.to_sql("fact_predictions", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"[+] Injected {len(df_preds)} recruitment predictions into fact_predictions table in SQLite.")

if __name__ == "__main__":
    train_player_performance_model()
    train_match_win_probability_model()
    build_role_recruitment_system()
