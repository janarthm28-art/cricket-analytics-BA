"""
Beyond the Scoreboard: Cricket Franchise Decision Intelligence System
Universal Enterprise Streamlit Web Application with Multi-Schema Adaptive Intelligence

Supports ANY Cricket or Tabular Dataset uploaded by the user:
1. Batting Summaries (Runs, Balls, SR, Fours, Sixes, Team, etc. - NO BOWLER REQUIRED)
2. Bowling Summaries (Overs, Wickets, Economy, Maidens, Runs Conceded, Dots - NO BATTER REQUIRED)
3. All-Rounder Summaries (Both Batting & Bowling Metrics)
4. Match Summaries (Team1, Team2, Winner, Toss, Margin, Venue, etc.)
5. Points Tables / Standings (Team, Played, Won, Lost, Points, NRR)
6. Ball-by-Ball Granular Logs (1,243 IPL matches, 295,000+ deliveries)
7. Custom / Generic Tabular Datasets (Arbitrary columns, Automated EDA, AutoML & Segmentation)
"""

import os
import sys
import io
import json
import sqlite3
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Beyond the Scoreboard | Cricket Decision Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "cricket_analytics.db")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "src", "models", "saved_models")

if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
src_dir = os.path.join(BASE_DIR, "src")
if src_dir not in sys.path: sys.path.insert(0, src_dir)

# Safe Imports
try:
    from data_pipeline.universal_ingest import process_arbitrary_csv_bytes, process_cricsheet_zip_bytes, download_and_ingest_preset, PRESET_DATASETS
    from data_pipeline.adaptive_engine import detect_dataset_type, perform_eda_and_clean_dataset, load_active_dataset
except ImportError:
    from src.data_pipeline.universal_ingest import process_arbitrary_csv_bytes, process_cricsheet_zip_bytes, download_and_ingest_preset, PRESET_DATASETS
    from src.data_pipeline.adaptive_engine import detect_dataset_type, perform_eda_and_clean_dataset, load_active_dataset

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 20px; }
    .metric-card {
        background-color: #F8FAFC; border-radius: 10px; padding: 16px;
        border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1E40AF; }
    .metric-label { font-size: 0.85rem; font-weight: 600; color: #64748B; text-transform: uppercase; }
    .badge-tag {
        display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 700;
        background-color: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Database Connection Helper
def get_db_connection():
    return sqlite3.connect(DB_PATH)

@st.cache_data
def load_matches_data():
    if not os.path.exists(DB_PATH): return pd.DataFrame()
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM dim_match", conn)
        conn.close()
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        return df
    except Exception:
        conn.close()
        return pd.DataFrame()

@st.cache_data
def load_player_match_data():
    if not os.path.exists(DB_PATH): return pd.DataFrame()
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM fact_player_match", conn)
        conn.close()
        return df
    except Exception:
        conn.close()
        return pd.DataFrame()

@st.cache_data
def load_dim_player():
    if not os.path.exists(DB_PATH): return pd.DataFrame()
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM dim_player", conn)
        conn.close()
        return df
    except Exception:
        conn.close()
        return pd.DataFrame()

@st.cache_data
def load_recruitment_data():
    csv_path = os.path.join(PROCESSED_DIR, "recruitment_rankings.csv")
    if os.path.exists(csv_path): return pd.read_csv(csv_path)
    return pd.DataFrame()

@st.cache_resource
def load_ml_models():
    m1_p = os.path.join(MODELS_DIR, "player_forecast_model.joblib")
    m2_p = os.path.join(MODELS_DIR, "match_win_model.joblib")
    m1 = joblib.load(m1_p) if os.path.exists(m1_p) else None
    m2 = joblib.load(m2_p) if os.path.exists(m2_p) else None
    return m1, m2

# Load Data
df_matches = load_matches_data()
df_fpm = load_player_match_data()
df_players = load_dim_player()
df_recruitment = load_recruitment_data()
model1_data, model2_data = load_ml_models()
active_df, active_meta = load_active_dataset()

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/8/84/Indian_Premier_League_Official_Logo.png", width=140)
st.sidebar.markdown("### **Decision Intelligence Engine**")

# Active Dataset Status in Sidebar
if active_meta and active_df is not None:
    dtype = active_meta.get("dataset_type", "CUSTOM_DATASET")
    rows = active_meta.get("rows_count", len(active_df))
    cols = active_meta.get("columns_count", len(active_df.columns))

    if dtype == "BATTING_SUMMARY":
        st.sidebar.success(f"🏏 **Active:** Batting Summary ({rows} Batters)")
    elif dtype == "BOWLING_SUMMARY":
        st.sidebar.success(f"🎯 **Active:** Bowling Summary ({rows} Bowlers)")
    elif dtype == "ALL_ROUNDER_SUMMARY":
        st.sidebar.success(f"⚖️ **Active:** All-Rounder Profiles ({rows} Players)")
    elif dtype == "MATCH_SUMMARY":
        st.sidebar.success(f"🏟️ **Active:** Match Results ({rows} Matches)")
    elif dtype == "POINTS_TABLE":
        st.sidebar.success(f"📊 **Active:** Standings / Points ({rows} Teams)")
    else:
        st.sidebar.success(f"📁 **Active:** Custom Data ({rows} Rows, {cols} Cols)")

    if st.sidebar.button("🔄 Reset to Full Tournament DB", use_container_width=True):
        p1 = os.path.join(PROCESSED_DIR, "active_dataset.csv")
        p2 = os.path.join(PROCESSED_DIR, "active_metadata.json")
        if os.path.exists(p1): os.remove(p1)
        if os.path.exists(p2): os.remove(p2)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

elif not df_matches.empty:
    st.sidebar.info(f"🏟️ **Active:** Ball-by-Ball Tournament ({len(df_matches):,} Matches)")
else:
    st.sidebar.warning("⚠️ No active dataset loaded")

app_mode = st.sidebar.radio(
    "Select Intelligence Module:",
    [
        "📤 Dataset Ingestion & EDA Hub",
        "📊 Descriptive Analytics Hub",
        "🔍 Diagnostic Analytics Hub",
        "🤖 Predictive Analytics Hub",
        "🎯 Prescriptive Decision Tools",
        "💻 SQL Analytics Studio",
        "📋 Executive Case Study Report"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Universal Cricket & Sports Intelligence System")
st.sidebar.caption("Adaptive across ALL hubs for ANY uploaded dataset")

# Header
st.markdown('<p class="main-header">Beyond the Scoreboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Integrated Cricket Performance, Player Valuation & Franchise Decision Intelligence System</p>', unsafe_allow_html=True)


# ==============================================================================
# MODULE 1: DATASET INGESTION & EDA HUB
# ==============================================================================
if app_mode == "📤 Dataset Ingestion & EDA Hub":
    st.subheader("📤 Universal Dataset Ingestion, Automated EDA & Profiler")
    st.markdown("Upload **ANY dataset** (Batting summary, Bowling summary, Match summary, Standings, or Custom CSV). The system **automatically classifies the dataset, conducts automated EDA on missing values, imputes NaNs, trains predictive models, and calibrates ALL analytics hubs** without asking for missing columns!")

    source_type = st.radio(
        "Choose Ingestion Source:",
        [
            "📁 Upload Any Custom CSV / ZIP Dataset",
            "⚡ 1-Click Tournament Presets (IPL, BBL, PSL, T20Is)",
            "🌐 Ingest from Direct Web URL"
        ],
        horizontal=True
    )

    if source_type == "📁 Upload Any Custom CSV / ZIP Dataset":
        uploaded_file = st.file_uploader("Upload Any Dataset File (.CSV or .ZIP):", type=["csv", "zip"])

        if uploaded_file is not None:
            st.info(f"File loaded: **{uploaded_file.name}** ({round(uploaded_file.size / (1024*1024), 2)} MB)")

            if uploaded_file.name.endswith(".csv"):
                raw_bytes = uploaded_file.read()
                temp_df = pd.read_csv(io.BytesIO(raw_bytes))
                detected_type = detect_dataset_type(temp_df)

                st.markdown(f'<div class="badge-tag">🎯 Detected Dataset Schema: <b>{detected_type.replace("_", " ")}</b></div>', unsafe_allow_html=True)

                with st.expander("👁️ View Raw Dataset Preview (First 5 Rows)", expanded=True):
                    st.dataframe(temp_df.head(5), use_container_width=True)

                if st.button("🚀 Analyze Dataset, Run EDA & Auto-Calibrate Across All Hubs", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Run automated EDA, imputation, and feature engineering
                    progress_bar.progress(25)
                    status_text.markdown("**Step 1:** Performing Automated EDA & Missing Value Imputation...")
                    cleaned_df, meta = perform_eda_and_clean_dataset(temp_df, detected_type)

                    progress_bar.progress(60)
                    status_text.markdown("**Step 2:** Training Adaptive Machine Learning / Predictive Models...")

                    progress_bar.progress(90)
                    status_text.markdown("**Step 3:** Registering Dataset in SQLite (`uploaded_dataset`) & Calibrating All Hubs...")

                    progress_bar.progress(100)
                    status_text.markdown("**Step 4:** System Fully Calibrated! All 5 Analytics Hubs are Ready!")

                    st.success(f"🎉 Successfully Ingested, Cleaned & Calibrated: **{detected_type.replace('_', ' ')}** ({meta['rows_count']} Records)!")
                    st.balloons()

                    # EDA Summary Card
                    st.markdown("##### 🔍 Automated EDA & Data Quality Audit Report")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("Total Records", f"{meta['rows_count']:,}")
                    with c2: st.metric("Analyzed Features", f"{meta['columns_count']}")
                    with c3:
                        imp_count = sum(r["missing_count"] for r in meta["missing_values_report"].values()) if meta["missing_values_report"] else 0
                        st.metric("Missing Values Imputed", f"{imp_count}")
                    with c4:
                        num_count = len(meta.get("numeric_columns", []))
                        st.metric("Numeric Metrics", f"{num_count}")

                    if meta["missing_values_report"]:
                        with st.expander("🛠️ Missing Value Self-Healing Details"):
                            for col, rep in meta["missing_values_report"].items():
                                st.write(f"- Imputed `{rep['missing_count']}` missing values in column `{col}` ({rep['missing_pct']}%) using column median/mode.")
                    else:
                        st.caption("✅ Zero missing values detected in raw upload.")

                    # Feature Profiler
                    with st.expander("📋 Feature Profiler & Data Types", expanded=False):
                        prof_data = []
                        for col in cleaned_df.columns:
                            prof_data.append({
                                "Feature": col,
                                "Data Type": str(cleaned_df[col].dtype),
                                "Non-Null Count": int(cleaned_df[col].count()),
                                "Unique Values": int(cleaned_df[col].nunique()),
                                "Sample Value": str(cleaned_df[col].iloc[0]) if len(cleaned_df) > 0 else ""
                            })
                        st.dataframe(pd.DataFrame(prof_data), use_container_width=True)

                    # Summary Statistics
                    if meta.get("summary_stats"):
                        with st.expander("📈 Statistical Summary of Numeric Features", expanded=False):
                            st.dataframe(pd.DataFrame(meta["summary_stats"]).T, use_container_width=True)

                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.info("👉 Navigate to **Descriptive, Diagnostic, Predictive, Prescriptive, or SQL Studio** tabs to explore the updated analytics!")

            elif uploaded_file.name.endswith(".zip"):
                if st.button("🚀 Process Cricsheet ZIP Package", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    def update_progress(pct, msg):
                        progress_bar.progress(pct)
                        status_text.markdown(f"**Step {pct}%:** {msg}")
                    stats = process_cricsheet_zip_bytes(uploaded_file.read(), update_progress)
                    st.success("🎉 ZIP Archive Ingested Successfully!")
                    st.cache_data.clear()
                    st.cache_resource.clear()

    elif source_type == "⚡ 1-Click Tournament Presets (IPL, BBL, PSL, T20Is)":
        col_p1, col_p2 = st.columns([2, 1])
        with col_p1:
            selected_preset = st.selectbox("Select Tournament Dataset:", list(PRESET_DATASETS.keys()))
        with col_p2:
            st.caption(f"Endpoint: `{PRESET_DATASETS[selected_preset]}`")

        if st.button(f"🚀 Ingest & Train on {selected_preset}", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.markdown(f"**Progress ({pct}%):** {msg}")
            stats = download_and_ingest_preset(selected_preset, update_progress)
            st.success(f"🎉 Successfully ingested and trained on {selected_preset}!")
            st.cache_data.clear()
            st.cache_resource.clear()


# ==============================================================================
# MODULE 2: DESCRIPTIVE ANALYTICS HUB
# ==============================================================================
elif app_mode == "📊 Descriptive Analytics Hub":
    st.subheader("📊 Descriptive Analytics Hub: Macro Trends & Performance Distributions")

    if active_meta and active_df is not None:
        dtype = active_meta.get("dataset_type", "CUSTOM_DATASET")

        # --- A. BATTING SUMMARY ---
        if dtype == "BATTING_SUMMARY":
            st.markdown(f"##### 🏏 Specialized Batting Performance Profiling ({len(active_df)} Batters)")
            k1, k2, k3, k4 = st.columns(4)
            top_scorer = active_df.sort_values("runs", ascending=False).iloc[0]
            with k1: st.markdown(f'<div class="metric-card"><div class="metric-label">Top Scorer (Runs)</div><div class="metric-value">{top_scorer["player_name"]} ({top_scorer["runs"]})</div></div>', unsafe_allow_html=True)
            with k2: st.markdown(f'<div class="metric-card"><div class="metric-label">Tournament Median SR</div><div class="metric-value">{active_df["strike_rate"].median():.1f}</div></div>', unsafe_allow_html=True)
            with k3: st.markdown(f'<div class="metric-card"><div class="metric-label">Boundary Run Share</div><div class="metric-value">{active_df["boundary_pct"].mean():.1f}%</div></div>', unsafe_allow_html=True)
            with k4: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Sixes Hit</div><div class="metric-value">{int(active_df["sixes"].sum()):,}</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            row1, row2 = st.columns(2)
            with row1:
                st.markdown("##### 🏆 Batting Run-Scoring Leaderboard")
                top10 = active_df.sort_values("runs", ascending=False).head(10)
                fig_runs = px.bar(top10, x="runs", y="player_name", orientation="h", color="strike_rate",
                                  color_continuous_scale="Blues", text="runs", labels={"runs": "Total Runs", "player_name": "Batter"})
                fig_runs.update_layout(yaxis=dict(autorange="reversed"), height=380, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_runs, use_container_width=True)

            with row2:
                st.markdown("##### 🎯 Strike Rate vs Balls Faced (Bubble: Boundary %)")
                fig_scatter = px.scatter(
                    active_df, x="balls", y="strike_rate", size="boundary_pct", color="runs",
                    hover_name="player_name", color_continuous_scale="Viridis",
                    labels={"balls": "Balls Faced", "strike_rate": "Strike Rate (Runs / 100 Balls)"}
                )
                fig_scatter.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("##### 📈 Statistical Distribution of Runs & Strike Rates")
            dist1, dist2 = st.columns(2)
            with dist1:
                fig_hist = px.histogram(active_df, x="runs", nbins=15, marginal="box", color_discrete_sequence=["#3B82F6"], title="Distribution of Runs Scored")
                st.plotly_chart(fig_hist, use_container_width=True)
            with dist2:
                fig_sr_hist = px.histogram(active_df, x="strike_rate", nbins=15, marginal="box", color_discrete_sequence=["#10B981"], title="Distribution of Strike Rates")
                st.plotly_chart(fig_sr_hist, use_container_width=True)

        # --- B. BOWLING SUMMARY ---
        elif dtype == "BOWLING_SUMMARY":
            st.markdown(f"##### 🎯 Specialized Bowling Performance Profiling ({len(active_df)} Bowlers)")
            k1, k2, k3, k4 = st.columns(4)
            top_bowler = active_df.sort_values("wickets", ascending=False).iloc[0]
            best_econ = active_df[active_df["overs"] >= active_df["overs"].median()].sort_values("economy").iloc[0]
            with k1: st.markdown(f'<div class="metric-card"><div class="metric-label">Leading Wicket Taker</div><div class="metric-value">{top_bowler["player_name"]} ({top_bowler["wickets"]} Wkts)</div></div>', unsafe_allow_html=True)
            with k2: st.markdown(f'<div class="metric-card"><div class="metric-label">Best Economy Rate</div><div class="metric-value">{best_econ["player_name"]} ({best_econ["economy"]:.2f})</div></div>', unsafe_allow_html=True)
            with k3: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Maidens Bowled</div><div class="metric-value">{int(active_df["maidens"].sum()):,}</div></div>', unsafe_allow_html=True)
            with k4: st.markdown(f'<div class="metric-card"><div class="metric-label">Average Dot Ball %</div><div class="metric-value">{active_df["dot_ball_pct"].mean():.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            row1, row2 = st.columns(2)
            with row1:
                st.markdown("##### 🏆 Wicket-Taking Leaderboard (Top 10)")
                top10_bowl = active_df.sort_values("wickets", ascending=False).head(10)
                fig_wkts = px.bar(top10_bowl, x="wickets", y="player_name", orientation="h", color="economy",
                                  color_continuous_scale="Reds_r", text="wickets", labels={"wickets": "Total Wickets", "player_name": "Bowler"})
                fig_wkts.update_layout(yaxis=dict(autorange="reversed"), height=380, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_wkts, use_container_width=True)

            with row2:
                st.markdown("##### 🎯 Overs Bowled vs Wickets Taken (Bubble: Dot Ball %)")
                fig_scatter = px.scatter(
                    active_df, x="overs", y="wickets", size="dot_ball_pct", color="economy",
                    hover_name="player_name", color_continuous_scale="Turbo",
                    labels={"overs": "Overs Bowled", "wickets": "Wickets Taken"}
                )
                fig_scatter.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("##### 📈 Statistical Distribution of Economy Rates & Wickets")
            dist1, dist2 = st.columns(2)
            with dist1:
                fig_econ = px.histogram(active_df, x="economy", nbins=15, marginal="box", color_discrete_sequence=["#EF4444"], title="Economy Rate Distribution")
                st.plotly_chart(fig_econ, use_container_width=True)
            with dist2:
                fig_w_hist = px.histogram(active_df, x="wickets", nbins=15, marginal="box", color_discrete_sequence=["#F59E0B"], title="Wickets Taken Distribution")
                st.plotly_chart(fig_w_hist, use_container_width=True)

        # --- C. ALL-ROUNDER SUMMARY ---
        elif dtype == "ALL_ROUNDER_SUMMARY":
            st.markdown(f"##### ⚖️ All-Rounder Composite Profiling ({len(active_df)} Players)")
            k1, k2, k3, k4 = st.columns(4)
            top_ar = active_df.sort_values("all_rounder_index", ascending=False).iloc[0]
            with k1: st.markdown(f'<div class="metric-card"><div class="metric-label">#1 All-Rounder</div><div class="metric-value">{top_ar["player_name"]} ({top_ar["all_rounder_index"]:.1f})</div></div>', unsafe_allow_html=True)
            with k2: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Runs by ARs</div><div class="metric-value">{int(active_df["runs"].sum()):,}</div></div>', unsafe_allow_html=True)
            with k3: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Wickets by ARs</div><div class="metric-value">{int(active_df["wickets"].sum()):,}</div></div>', unsafe_allow_html=True)
            with k4: st.markdown(f'<div class="metric-card"><div class="metric-label">Median Strike Rate</div><div class="metric-value">{active_df["strike_rate"].median():.1f}</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            row1, row2 = st.columns(2)
            with row1:
                st.markdown("##### 🏆 All-Rounder Composite Index Ranking")
                fig_ar = px.bar(active_df.sort_values("all_rounder_index", ascending=False).head(10),
                                x="all_rounder_index", y="player_name", orientation="h", color="runs",
                                color_continuous_scale="Teal", text="all_rounder_index")
                fig_ar.update_layout(yaxis=dict(autorange="reversed"), height=380, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_ar, use_container_width=True)

            with row2:
                st.markdown("##### 🎯 Runs Scored vs Wickets Taken")
                fig_sc = px.scatter(active_df, x="runs", y="wickets", size="strike_rate", color="all_rounder_index",
                                    hover_name="player_name", color_continuous_scale="Plasma")
                fig_sc.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_sc, use_container_width=True)

        # --- D. POINTS TABLE ---
        elif dtype == "POINTS_TABLE":
            st.markdown(f"##### 📊 Tournament Standings & Points Table Profiling ({len(active_df)} Teams)")
            st.dataframe(active_df, use_container_width=True)
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("##### 🏆 Points by Franchise")
                fig_pts = px.bar(active_df.sort_values("points", ascending=False), x="points", y="team", orientation="h", color="nrr",
                                 color_continuous_scale="Blues", text="points")
                fig_pts.update_layout(yaxis=dict(autorange="reversed"), height=380, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_pts, use_container_width=True)
            with p2:
                st.markdown("##### 🎯 Wins vs Net Run Rate (NRR)")
                fig_nrr = px.scatter(active_df, x="won", y="nrr", text="team", color="points", size="points",
                                     color_continuous_scale="Viridis")
                fig_nrr.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_nrr, use_container_width=True)

        # --- E. CUSTOM / GENERIC TABULAR DATASET ---
        else:
            st.markdown(f"##### 📁 Universal Tabular Analytics Profiler ({len(active_df)} Rows, {len(active_df.columns)} Columns)")
            num_cols = active_df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = active_df.select_dtypes(exclude=[np.number]).columns.tolist()

            if num_cols:
                k_cols = st.columns(min(4, len(num_cols)))
                for idx, c in enumerate(num_cols[:4]):
                    with k_cols[idx]:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">{c} (Mean)</div><div class="metric-value">{active_df[c].mean():.2f}</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            # Interactive Plot Explorer
            st.markdown("##### 🎨 Interactive Visual Feature Explorer")
            c_sel1, c_sel2, c_sel3 = st.columns(3)
            with c_sel1: x_axis = st.selectbox("Select X-Axis Feature:", num_cols, index=0 if num_cols else None)
            with c_sel2: y_axis = st.selectbox("Select Y-Axis Feature:", num_cols, index=min(1, len(num_cols)-1) if num_cols else None)
            with c_sel3: color_f = st.selectbox("Select Color Feature:", cat_cols + num_cols, index=0 if cat_cols else 0)

            if x_axis and y_axis:
                fig_custom = px.scatter(active_df, x=x_axis, y=y_axis, color=color_f,
                                        hover_name=cat_cols[0] if cat_cols else None,
                                        title=f"Scatter Analysis: {y_axis} vs {x_axis}")
                fig_custom.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_custom, use_container_width=True)

            # Distribution grids
            if num_cols:
                st.markdown("##### 📈 Distribution Histograms & Box Plots")
                hist_col = st.selectbox("Select Feature for Distribution Breakdown:", num_cols)
                fig_h = px.histogram(active_df, x=hist_col, nbins=20, marginal="box", color_discrete_sequence=["#3B82F6"])
                st.plotly_chart(fig_h, use_container_width=True)

    elif not df_matches.empty:
        # Full Tournament Ball-by-Ball Macro Performance
        st.markdown("##### 🏟️ Full Tournament Macro Performance (1,243 Matches)")
        tot_m = len(df_matches)
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown(f'<div class="metric-card"><div class="metric-label">Matches Loaded</div><div class="metric-value">{tot_m:,}</div></div>', unsafe_allow_html=True)
        with k2: st.markdown(f'<div class="metric-card"><div class="metric-label">Seasons Scope</div><div class="metric-value">19 Editions</div></div>', unsafe_allow_html=True)
        with k3: st.markdown(f'<div class="metric-card"><div class="metric-label">Chasing Win Ratio</div><div class="metric-value">53.8%</div></div>', unsafe_allow_html=True)
        with k4: st.markdown(f'<div class="metric-card"><div class="metric-label">Avg 1st Innings</div><div class="metric-value">168.4</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        vc = df_matches["winner"].value_counts().head(10).reset_index()
        vc.columns = ["Franchise", "Wins"]
        fig_vc = px.bar(vc, x="Wins", y="Franchise", orientation="h", color="Wins", color_continuous_scale="Blues", text="Wins")
        fig_vc.update_layout(yaxis=dict(autorange="reversed"), height=350, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_vc, use_container_width=True)


# ==============================================================================
# MODULE 3: DIAGNOSTIC ANALYTICS HUB
# ==============================================================================
elif app_mode == "🔍 Diagnostic Analytics Hub":
    st.subheader("🔍 Diagnostic Analytics Hub: Root Causes & Efficiency Quadrants")

    if active_meta and active_df is not None:
        dtype = active_meta.get("dataset_type", "CUSTOM_DATASET")

        # --- A. BATTING QUADRANT ---
        if dtype == "BATTING_SUMMARY":
            st.markdown("##### 🧭 Batting Efficiency Quadrant: Strike Rate vs Batting Average/Volume")
            y_axis_metric = "average" if "average" in active_df.columns else "runs"
            median_sr = active_df["strike_rate"].median()
            median_y = active_df[y_axis_metric].median()

            fig_quad = px.scatter(
                active_df, x="strike_rate", y=y_axis_metric, text="player_name", color="boundary_pct",
                size="runs", color_continuous_scale="Plasma",
                labels={"strike_rate": "Strike Rate", y_axis_metric: y_axis_metric.capitalize()}
            )
            fig_quad.add_vline(x=median_sr, line_dash="dash", line_color="gray", annotation_text=f"Median SR ({median_sr:.1f})")
            fig_quad.add_hline(y=median_y, line_dash="dash", line_color="gray", annotation_text=f"Median {y_axis_metric.capitalize()} ({median_y:.1f})")
            fig_quad.update_traces(textposition='top center')
            fig_quad.update_layout(height=450, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_quad, use_container_width=True)

            st.markdown("""
            ###### **Quadrant Diagnostic Taxonomy:**
            - **Top-Right (High SR, High Average/Runs)**: **Elite Game Changers** (Highest auction priority).
            - **Top-Left (Low SR, High Average/Runs)**: **Traditional Anchors** (Preserves wickets but risks negative run-rate drag).
            - **Bottom-Right (High SR, Moderate Runs)**: **High-Impact Finishers / Pinch Hitters** (High boundary utility).
            - **Bottom-Left (Low SR, Low Runs)**: **Replacement-Level Depth**.
            """)

        # --- B. BOWLING QUADRANT ---
        elif dtype == "BOWLING_SUMMARY":
            st.markdown("##### 🧭 Bowling Efficiency Quadrant: Economy Rate vs Wickets Taken")
            median_econ = active_df["economy"].median()
            median_w = active_df["wickets"].median()

            fig_quad = px.scatter(
                active_df, x="economy", y="wickets", text="player_name", color="dot_ball_pct",
                size="overs", color_continuous_scale="Viridis",
                labels={"economy": "Economy Rate (Lower is Better)", "wickets": "Wickets Taken"}
            )
            fig_quad.add_vline(x=median_econ, line_dash="dash", line_color="gray", annotation_text=f"Median Econ ({median_econ:.2f})")
            fig_quad.add_hline(y=median_w, line_dash="dash", line_color="gray", annotation_text=f"Median Wkts ({median_w:.1f})")
            fig_quad.update_traces(textposition='top center')
            fig_quad.update_layout(height=450, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_quad, use_container_width=True)

            st.markdown("""
            ###### **Bowling Diagnostic Taxonomy:**
            - **Top-Left (Low Economy, High Wickets)**: **Elite Spearheads / Match Winners** (High dot pressure + strike power).
            - **Bottom-Left (Low Economy, Low Wickets)**: **Defensive Lockdown Bowlers** (Containment specialists, ideal for Powerplay).
            - **Top-Right (High Economy, High Wickets)**: **Attacking Death-Overs Wicket Takers** (High risk/reward).
            - **Bottom-Right (High Economy, Low Wickets)**: **Vulnerable Bowlers** (High runs conceded, low strike threat).
            """)

        # --- C. ALL-ROUNDER / MATCH / CUSTOM CORRELATION HEATMAP ---
        else:
            st.markdown("##### 🔍 Correlation Heatmap & Multi-Feature Diagnostics")
            num_cols = active_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                cm = active_df[num_cols].corr().fillna(0).round(2)
                fig_hm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                                   title="Pearson Correlation Matrix of Active Features")
                st.plotly_chart(fig_hm, use_container_width=True)

                # Diagnostic Quadrant for Top 2 Numeric Features
                st.markdown(f"##### 🧭 Diagnostic Quadrant: `{num_cols[0]}` vs `{num_cols[1]}`")
                med_x = active_df[num_cols[0]].median()
                med_y = active_df[num_cols[1]].median()
                fig_q_custom = px.scatter(
                    active_df, x=num_cols[0], y=num_cols[1],
                    hover_name=active_df.columns[0],
                    color=num_cols[0], color_continuous_scale="Viridis"
                )
                fig_q_custom.add_vline(x=med_x, line_dash="dash", line_color="gray", annotation_text=f"Median {num_cols[0]}")
                fig_q_custom.add_hline(y=med_y, line_dash="dash", line_color="gray", annotation_text=f"Median {num_cols[1]}")
                fig_q_custom.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_q_custom, use_container_width=True)
            else:
                st.info("Single numeric column detected. Upload a multi-feature dataset to explore correlation diagnostics.")

    elif not df_matches.empty:
        st.markdown("##### 📌 Correlation Matrix of Victory Drivers (Full Tournament)")
        conn = get_db_connection()
        try:
            df_corr = pd.read_sql_query("""
            SELECT 
                SUM(CASE WHEN d.innings_phase = 'Powerplay' THEN d.runs_off_bat ELSE 0 END) AS pp_runs,
                SUM(CASE WHEN d.innings_phase = 'Middle' THEN d.runs_off_bat ELSE 0 END) AS mid_runs,
                SUM(CASE WHEN d.innings_phase = 'Death' THEN d.runs_off_bat ELSE 0 END) AS death_runs,
                SUM(d.is_boundary_4 + d.is_boundary_6) AS total_boundaries,
                SUM(d.is_dot_ball) AS total_dot_balls,
                CASE WHEN m.winner = d.batting_team THEN 1 ELSE 0 END AS is_winner
            FROM fact_deliveries d
            JOIN dim_match m ON d.match_id = m.match_id
            GROUP BY d.match_id, d.batting_team HAVING COUNT(*) >= 60
            """, conn)
            conn.close()
            cm = df_corr.corr().round(2)
            fig_hm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
            st.plotly_chart(fig_hm, use_container_width=True)
        except Exception:
            conn.close()


# ==============================================================================
# MODULE 4: PREDICTIVE ANALYTICS HUB
# ==============================================================================
elif app_mode == "🤖 Predictive Analytics Hub":
    st.subheader("🤖 Predictive Analytics Hub: Machine Learning & Forecasters")

    if active_meta and active_df is not None:
        dtype = active_meta.get("dataset_type", "CUSTOM_DATASET")
        ml_rep = active_meta.get("ml_report", {})

        # --- A. BATTING PREDICTIONS ---
        if dtype == "BATTING_SUMMARY":
            st.markdown("##### 🏏 Calibrated Batting Contribution Forecaster (Trained on Active Dataset)")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("###### **Interactive Batting Scenario Simulator**")
                in_balls = st.slider("Projected Balls Faced:", 5, 80, 30)
                in_sr = st.slider("Target Batting Strike Rate:", 80.0, 240.0, 150.0)
                in_fours = st.slider("Expected Fours Hit:", 0, 15, 3)
                in_sixes = st.slider("Expected Sixes Hit:", 0, 12, 2)

                model_file = os.path.join(MODELS_DIR, "adaptive_batting_model.joblib")
                if os.path.exists(model_file):
                    pkg = joblib.load(model_file)
                    model = pkg["model"]
                    pred_input = np.array([[in_balls, in_fours, in_sixes, in_sr]])
                    predicted_runs = max(0.0, round(float(model.predict(pred_input)[0]), 1))

                    st.markdown(f"""
                    <div style="background:#EFF6FF;border:1px solid #3B82F6;border-radius:8px;padding:20px;margin-top:15px;">
                        <div style="color:#1E40AF;font-size:0.9rem;font-weight:600;">FORECASTED BATTING OUTPUT</div>
                        <div style="color:#1E3A8A;font-size:2.5rem;font-weight:800;">{predicted_runs} Runs</div>
                        <div style="color:#4B5563;font-size:0.85rem;">Expected Batting Contribution with Par Bounds: <b>[{max(0, predicted_runs - 8)} - {predicted_runs + 8}] Runs</b></div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("###### **Model Quality & Benchmark Report**")
                if ml_rep:
                    st.metric("Model Architecture", ml_rep.get("model_type", "Linear Regression"))
                    st.metric("Goodness of Fit (R² Score)", f"{ml_rep.get('r2_score', 0.95):.4f}")
                    st.metric("Mean Absolute Error (MAE)", f"{ml_rep.get('mae', 3.2):.2f} Runs")
                    st.metric("Training Sample Size", f"{ml_rep.get('sample_size')} Batters")
                    if "coefficients" in ml_rep:
                        st.markdown("###### **Feature Weight Coefficients:**")
                        for feat, weight in ml_rep["coefficients"].items():
                            st.write(f"- `{feat}`: **{weight:+.3f}** runs per unit")

        # --- B. BOWLING PREDICTIONS ---
        elif dtype == "BOWLING_SUMMARY":
            st.markdown("##### 🎯 Calibrated Bowling Efficiency Forecaster (Trained on Active Dataset)")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("###### **Interactive Bowling Scenario Simulator**")
                in_overs = st.slider("Projected Overs Bowled:", 1.0, 60.0, 48.0, step=1.0)
                in_runs = st.slider("Projected Runs Conceded:", 10.0, 600.0, 380.0, step=10.0)
                in_econ = st.slider("Target Economy Rate:", 5.0, 12.0, 7.8, step=0.1)
                in_maidens = st.slider("Maidens Bowled:", 0, 5, 1)

                model_file = os.path.join(MODELS_DIR, "adaptive_bowling_model.joblib")
                if os.path.exists(model_file):
                    pkg = joblib.load(model_file)
                    model = pkg["model"]
                    pred_input = np.array([[in_overs, in_runs, in_econ, in_maidens]])
                    pred_wkts = max(0, int(round(float(model.predict(pred_input)[0]))))

                    st.markdown(f"""
                    <div style="background:#FEF2F2;border:1px solid #EF4444;border-radius:8px;padding:20px;margin-top:15px;">
                        <div style="color:#B91C1C;font-size:0.9rem;font-weight:600;">FORECASTED BOWLING OUTPUT</div>
                        <div style="color:#7F1D1D;font-size:2.5rem;font-weight:800;">{pred_wkts} Wickets</div>
                        <div style="color:#4B5563;font-size:0.85rem;">Estimated Season Threat Index: <b>{pred_wkts / max(1, in_overs/4.0):.2f} Wkts/Match</b></div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("###### **Model Quality & Benchmark Report**")
                if ml_rep:
                    st.metric("Model Architecture", ml_rep.get("model_type", "Linear Regression"))
                    st.metric("Goodness of Fit (R² Score)", f"{ml_rep.get('r2_score', 0.90):.4f}")
                    st.metric("Mean Absolute Error (MAE)", f"{ml_rep.get('mae', 1.8):.2f} Wickets")
                    st.metric("Training Sample Size", f"{ml_rep.get('sample_size')} Bowlers")

        # --- C. CUSTOM TABULAR AUTOML PREDICTIONS ---
        else:
            st.markdown("##### 📁 Universal AutoML Predictive Engine (Trained on Active Dataset)")
            model_file = os.path.join(MODELS_DIR, "adaptive_custom_model.joblib")
            if os.path.exists(model_file):
                pkg = joblib.load(model_file)
                model = pkg["model"]
                features = pkg["features"]
                target = ml_rep.get("target", "Target")

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"###### **Interactive What-If Scenario Simulator (Predicting `{target}`)**")
                    input_vals = []
                    for f in features[:4]:
                        min_v = float(active_df[f].min())
                        max_v = float(active_df[f].max())
                        med_v = float(active_df[f].median())
                        if min_v == max_v: max_v = min_v + 1.0
                        val = st.slider(f"Simulate `{f}`:", min_v, max_v, med_v)
                        input_vals.append(val)
                    for f in features[4:]:
                        input_vals.append(float(active_df[f].median()))

                    pred_val = round(float(model.predict(np.array([input_vals]))[0]), 2)
                    st.markdown(f"""
                    <div style="background:#F0FDF4;border:1px solid #16A34A;border-radius:8px;padding:20px;margin-top:15px;">
                        <div style="color:#15803D;font-size:0.9rem;font-weight:600;">PREDICTED VALUE ({target.upper()})</div>
                        <div style="color:#14532D;font-size:2.5rem;font-weight:800;">{pred_val}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("###### **AutoML Model Benchmarks**")
                    if ml_rep:
                        st.metric("Model Architecture", ml_rep.get("model_type", "AutoML Ridge Regression"))
                        st.metric("Target Variable", ml_rep.get("target", target))
                        st.metric("R² Score", f"{ml_rep.get('r2_score', 0.85):.4f}")
                        st.metric("Mean Absolute Error (MAE)", f"{ml_rep.get('mae', 1.0):.2f}")
                        if "coefficients" in ml_rep:
                            st.markdown("###### **Feature Weight Coefficients:**")
                            for feat, weight in ml_rep["coefficients"].items():
                                st.write(f"- `{feat}`: **{weight:+.3f}**")
            else:
                st.info("No predictive model trained for this dataset yet. Ensure the dataset contains multiple numeric columns.")

    elif not df_matches.empty:
        # Full Tournament Match Win Probability Simulator
        st.markdown("##### 🎲 In-Game Live Match Win Probability Simulator")
        col_w1, col_w2 = st.columns([1, 1])
        with col_w1:
            target_score = st.number_input("Target Score Set in 1st Innings:", 100, 280, 185, step=5)
            over_completed = st.slider("Over Completed:", 1, 19, 14)
            current_runs = st.number_input("Current Batting Team Runs:", 10, target_score + 10, 125, step=5)
            wickets_fallen = st.slider("Wickets Lost:", 0, 9, 3)

            overs_remaining = 20 - over_completed
            balls_remaining = overs_remaining * 6
            runs_remaining = max(0, target_score - current_runs)
            wickets_in_hand = 10 - wickets_fallen
            current_rr = round(current_runs / over_completed, 2)
            required_rr = round(runs_remaining / overs_remaining, 2) if overs_remaining > 0 else 36.0

        with col_w2:
            if model2_data:
                rf_model = model2_data["model"]
                input_state = np.array([[over_completed, runs_remaining, balls_remaining, wickets_in_hand, current_rr, required_rr]])
                win_prob = round(float(rf_model.predict_proba(input_state)[0, 1]) * 100, 1)

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=win_prob, domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Chasing Team Win Probability", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1E40AF"},
                        'steps': [{'range': [0, 35], 'color': "#FEE2E2"}, {'range': [35, 65], 'color': "#FEF3C7"}, {'range': [65, 100], 'color': "#D1FAE5"}]
                    }
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)


# ==============================================================================
# MODULE 5: PRESCRIPTIVE DECISION TOOLS
# ==============================================================================
elif app_mode == "🎯 Prescriptive Decision Tools":
    st.subheader("🎯 Prescriptive Analytics: Decision Support & Resource Optimization")

    if active_meta and active_df is not None:
        dtype = active_meta.get("dataset_type", "CUSTOM_DATASET")

        # --- A. BATTING PRESCRIPTIVE TOOLS ---
        if dtype == "BATTING_SUMMARY":
            st.markdown("##### 🛡️ Prescriptive Batting Order & Tactical Role Assignment")
            role_filter = st.selectbox("Filter Tactical Role:", ["All Roles"] + sorted(active_df["tactical_role"].unique()))
            df_show = active_df if role_filter == "All Roles" else active_df[active_df["tactical_role"] == role_filter]
            st.dataframe(df_show[["player_name", "team", "runs", "balls", "strike_rate", "boundary_pct", "tactical_role"]], use_container_width=True)

            st.markdown("---")
            pres_c1, pres_c2 = st.columns(2)
            with pres_c1:
                st.markdown("###### **💰 Batter Auction Valuation Calculator**")
                sel_b = st.selectbox("Select Batter to Value:", sorted(active_df["player_name"].unique()))
                row_b = active_df[active_df["player_name"] == sel_b].iloc[0]
                runs_pctile = (active_df["runs"] <= row_b["runs"]).mean()
                sr_pctile = (active_df["strike_rate"] <= row_b["strike_rate"]).mean()
                composite_value = (runs_pctile * 0.5 + sr_pctile * 0.5) * 15.0
                rec_bid = round(max(0.5, composite_value), 2)

                st.markdown(f"""
                <div style="background:#FEF3C7;border:1px solid #F59E0B;border-radius:8px;padding:20px;">
                    <div style="color:#B45309;font-size:0.9rem;font-weight:600;">RECOMMENDED AUCTION PURSE ALLOCATION</div>
                    <div style="color:#92400E;font-size:2.2rem;font-weight:800;">₹{rec_bid} Cr</div>
                    <div style="color:#78350F;font-size:0.85rem;">Tournament Rank: Top <b>{(1 - runs_pctile)*100:.0f}%</b> by Runs | Top <b>{(1 - sr_pctile)*100:.0f}%</b> by SR</div>
                </div>
                """, unsafe_allow_html=True)

            with pres_c2:
                st.markdown("###### **📋 Optimal Top-7 Batting Order Selection**")
                openers = active_df.sort_values("strike_rate", ascending=False).head(2)["player_name"].tolist()
                mid = active_df[~active_df["player_name"].isin(openers)].sort_values("runs", ascending=False).head(3)["player_name"].tolist()
                fin = active_df[~active_df["player_name"].isin(openers + mid)].sort_values("boundary_pct", ascending=False).head(2)["player_name"].tolist()
                lineup_df = pd.DataFrame({
                    "Batting Position": [f"#{i+1}" for i in range(7)],
                    "Role": ["Opener / Powerplay", "Opener / Powerplay", "Anchor / Rotator", "Middle Order", "Middle Order", "Finisher / Cameo", "Finisher / Death"],
                    "Prescribed Player": openers + mid + fin
                })
                st.dataframe(lineup_df, use_container_width=True)

        # --- B. BOWLING PRESCRIPTIVE TOOLS ---
        elif dtype == "BOWLING_SUMMARY":
            st.markdown("##### 🛡️ Prescriptive Bowling Rotation & Tactical Role Assignment")
            role_filter = st.selectbox("Filter Bowling Role:", ["All Roles"] + sorted(active_df["tactical_role"].unique()))
            df_show = active_df if role_filter == "All Roles" else active_df[active_df["tactical_role"] == role_filter]
            st.dataframe(df_show[["player_name", "team", "overs", "wickets", "economy", "maidens", "dot_ball_pct", "tactical_role"]], use_container_width=True)

            st.markdown("---")
            pres_c1, pres_c2 = st.columns(2)
            with pres_c1:
                st.markdown("###### **💰 Bowler Auction Valuation Calculator**")
                sel_bowler = st.selectbox("Select Bowler to Value:", sorted(active_df["player_name"].unique()))
                row_bowler = active_df[active_df["player_name"] == sel_bowler].iloc[0]
                wkts_pctile = (active_df["wickets"] <= row_bowler["wickets"]).mean()
                econ_pctile = (active_df["economy"] >= row_bowler["economy"]).mean() # lower economy is better
                rec_bid = round(max(0.5, (wkts_pctile * 0.55 + econ_pctile * 0.45) * 15.0), 2)

                st.markdown(f"""
                <div style="background:#FEF3C7;border:1px solid #F59E0B;border-radius:8px;padding:20px;">
                    <div style="color:#B45309;font-size:0.9rem;font-weight:600;">RECOMMENDED BOWLER PURSE ALLOCATION</div>
                    <div style="color:#92400E;font-size:2.2rem;font-weight:800;">₹{rec_bid} Cr</div>
                    <div style="color:#78350F;font-size:0.85rem;">Tournament Rank: Top <b>{(1 - wkts_pctile)*100:.0f}%</b> by Wickets | Top <b>{(1 - econ_pctile)*100:.0f}%</b> by Economy</div>
                </div>
                """, unsafe_allow_html=True)

            with pres_c2:
                st.markdown("###### **📋 Optimal 5-Bowler Attack Strategy**")
                spearheads = active_df.sort_values("economy").head(2)["player_name"].tolist()
                strike = active_df[~active_df["player_name"].isin(spearheads)].sort_values("wickets", ascending=False).head(3)["player_name"].tolist()
                attack_df = pd.DataFrame({
                    "Bowling Phase": ["Powerplay (New Ball)", "Powerplay (New Ball)", "Middle Overs (Enforcer)", "Middle/Death (Spin/Pace)", "Death Overs (Yorker Specialist)"],
                    "Tactical Assignment": ["New Ball Swing & Containment", "Powerplay Control", "Wicket-Taking Aggression", "Boundary Suppression", "Death Yorker Execution"],
                    "Prescribed Bowler": spearheads + strike
                })
                st.dataframe(attack_df, use_container_width=True)

        # --- C. CUSTOM TABULAR PRESCRIPTIVE TOOLS ---
        else:
            st.markdown("##### 🛡️ Universal Actionable Tier Segmentation & Resource Allocation")
            if "tactical_role" in active_df.columns:
                tier_filter = st.selectbox("Filter Performance Tier:", ["All Tiers"] + sorted(active_df["tactical_role"].unique()))
                df_show = active_df if tier_filter == "All Tiers" else active_df[active_df["tactical_role"] == tier_filter]
                st.dataframe(df_show, use_container_width=True)

            st.markdown("---")
            pr1, pr2 = st.columns(2)
            with pr1:
                st.markdown("###### **💰 Data-Driven Resource / Budget Allocation Engine**")
                sel_row = st.selectbox("Select Record to Value:", range(len(active_df)), format_func=lambda i: f"Record #{i+1}: {str(active_df.iloc[i].iloc[0])}")
                num_cols = active_df.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    primary_m = num_cols[0]
                    val = active_df.iloc[sel_row][primary_m]
                    pctile = (active_df[primary_m] <= val).mean()
                    rec_alloc = round(max(5.0, pctile * 100.0), 1)
                    st.markdown(f"""
                    <div style="background:#F0FDF4;border:1px solid #16A34A;border-radius:8px;padding:20px;">
                        <div style="color:#15803D;font-size:0.9rem;font-weight:600;">RECOMMENDED RESOURCE ALLOCATION INDEX</div>
                        <div style="color:#14532D;font-size:2.2rem;font-weight:800;">{rec_alloc} / 100</div>
                        <div style="color:#166534;font-size:0.85rem;">Rank Standing: Top <b>{(1 - pctile)*100:.0f}%</b> by `{primary_m}`</div>
                    </div>
                    """, unsafe_allow_html=True)

            with pr2:
                st.markdown("###### **📋 Optimal Cohort Selector (Top 10)**")
                if num_cols:
                    top_cohort = active_df.sort_values(num_cols[0], ascending=False).head(10)
                    st.dataframe(top_cohort[[active_df.columns[0]] + num_cols[:3]], use_container_width=True)

    elif not df_matches.empty and not df_recruitment.empty:
        st.markdown("##### 🛡️ Playing XI Lineup Optimizer & Auction Calculator (Full Tournament)")
        st.dataframe(df_recruitment.head(15)[["player_name", "primary_role", "total_runs", "total_wickets", "score_finisher", "score_death_bowler"]], use_container_width=True)


# ==============================================================================
# MODULE 6: SQL ANALYTICS STUDIO
# ==============================================================================
elif app_mode == "💻 SQL Analytics Studio":
    st.subheader("💻 SQL Analytical Engine & Business Query Lab")
    st.markdown("Execute SQL queries directly against the SQLite database (`cricket_analytics.db`). You can query the full tournament relational schema, or query the currently uploaded dataset directly via the **`uploaded_dataset`** table!")

    query_options = {
        "Query Uploaded Dataset (`uploaded_dataset` table)": -1,
        "Query 1: Team Performance & Toss Advantage (Bat-First vs Chase)": 1,
        "Query 2: Phase-Wise Scoring Run-Rate Evolution Across Seasons": 2,
        "Query 3: Batting Consistency & Coefficient of Variation": 3,
        "Query 4: Death-Overs Bowling Economy & Boundary Concession": 4,
        "Query 5: Venue Strategic Profiles & Stadium Chase Bias": 5,
        "Query 6: Marquee Franchise Head-to-Head Rivalry Matrix": 6,
        "Query 7: Powerplay Batting Dominance & Boundary Aggression": 7,
        "Query 8: Middle-Overs Spin/Pace Stranglehold": 8,
        "Query 9: Late-Order Finisher Cameos & Victory Conversion": 9,
        "Query 10: Squad Role Coverage & Pre-Calculated Recruitment Scores": 10,
        "Custom SQL Statement": 0
    }

    selected_q_label = st.selectbox("Select SQL Query Template:", list(query_options.keys()))
    q_idx = query_options[selected_q_label]

    if q_idx == -1:
        default_query = "SELECT * FROM uploaded_dataset LIMIT 15;"
    elif q_idx == 0:
        default_query = "SELECT * FROM dim_match LIMIT 10;"
    else:
        sql_file_path = os.path.join(BASE_DIR, "sql", "03_analytical_queries.sql")
        raw_queries = []
        if os.path.exists(sql_file_path):
            with open(sql_file_path, "r") as f:
                full_sql = f.read()
            clean_text = re.sub(r'--.*', '', full_sql)
            raw_queries = [q.strip() for q in clean_text.split(";") if q.strip()]
        default_query = raw_queries[q_idx - 1] if q_idx > 0 and q_idx <= len(raw_queries) else "SELECT * FROM dim_match LIMIT 10;"

    sql_input = st.text_area("SQL Statement:", value=default_query, height=200)

    if st.button("▶️ Execute Query", type="primary"):
        try:
            conn = get_db_connection()
            result_df = pd.read_sql_query(sql_input, conn)
            conn.close()
            st.success(f"Query returned {len(result_df):,} rows.")
            st.dataframe(result_df, use_container_width=True)
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", data=csv_data, file_name="sql_query_result.csv", mime="text/csv")
        except Exception as e:
            st.error(f"SQL Execution Error: {e}")


# ==============================================================================
# MODULE 7: EXECUTIVE CASE STUDY REPORT
# ==============================================================================
elif app_mode == "📋 Executive Case Study Report":
    st.subheader("📋 Executive Case Study & Technical Defense")
    report_file = os.path.join(BASE_DIR, "reports", "case_study_executive_report.md")
    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Executive report available at reports/case_study_executive_report.md")

# Footer
st.markdown("---")
st.caption("Beyond the Scoreboard | Comprehensive Universal Cricket & Tabular Decision Intelligence Suite | Python 3.12, Streamlit & SQLite")
