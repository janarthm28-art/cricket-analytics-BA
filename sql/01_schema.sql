-- ==============================================================================
-- PROJECT: Beyond the Scoreboard: Cricket Franchise Decision Intelligence System
-- SCRIPT: 01_schema.sql - Relational Schema Architecture (PostgreSQL / MySQL / SQLite)
-- ==============================================================================

-- Drop tables in reverse foreign key order if resetting
DROP TABLE IF EXISTS fact_predictions;
DROP TABLE IF EXISTS fact_player_match;
DROP TABLE IF EXISTS fact_deliveries;
DROP TABLE IF EXISTS dim_player;
DROP TABLE IF EXISTS dim_match;
DROP TABLE IF EXISTS dim_team;

-- ------------------------------------------------------------------------------
-- 1. DIMENSION: dim_team
-- Master table for IPL franchises, abbreviations, and status
-- ------------------------------------------------------------------------------
CREATE TABLE dim_team (
    team_id INT PRIMARY KEY,
    team_name VARCHAR(100) NOT NULL UNIQUE,
    team_code VARCHAR(10) NOT NULL,
    current_status VARCHAR(20) DEFAULT 'Active'
);

-- ------------------------------------------------------------------------------
-- 2. DIMENSION: dim_match
-- Metadata for every match: teams, venues, toss decisions, winners, margins
-- ------------------------------------------------------------------------------
CREATE TABLE dim_match (
    match_id INT PRIMARY KEY,
    season VARCHAR(15) NOT NULL,
    match_date DATE NOT NULL,
    match_type VARCHAR(20) DEFAULT 'T20',
    venue VARCHAR(255) NOT NULL,
    city VARCHAR(100),
    team1 VARCHAR(100) NOT NULL,
    team2 VARCHAR(100) NOT NULL,
    toss_winner VARCHAR(100) NOT NULL,
    toss_decision VARCHAR(20) NOT NULL,
    winner VARCHAR(100),
    win_type VARCHAR(20), -- 'runs', 'wickets', 'tie', 'no result'
    win_margin INT DEFAULT 0,
    player_of_match VARCHAR(100),
    target_runs INT,
    target_overs INT
);

CREATE INDEX idx_match_season ON dim_match(season);
CREATE INDEX idx_match_date ON dim_match(match_date);
CREATE INDEX idx_match_winner ON dim_match(winner);
CREATE INDEX idx_match_venue ON dim_match(venue);

-- ------------------------------------------------------------------------------
-- 3. DIMENSION: dim_player
-- Registry of players with unique Cricsheet identifiers and playing roles
-- ------------------------------------------------------------------------------
CREATE TABLE dim_player (
    player_id VARCHAR(50) PRIMARY KEY, -- Cricsheet Global Identifier
    player_name VARCHAR(100) NOT NULL,
    primary_role VARCHAR(50) DEFAULT 'All-Rounder', -- Batter, Bowler, All-Rounder, Wicketkeeper
    batting_hand VARCHAR(30), -- Right-hand bat, Left-hand bat
    bowling_style VARCHAR(60) -- Right-arm fast, Slow left-arm orthodox, etc.
);

CREATE INDEX idx_player_name ON dim_player(player_name);
CREATE INDEX idx_player_role ON dim_player(primary_role);

-- ------------------------------------------------------------------------------
-- 4. FACT TABLE: fact_deliveries
-- Granular ball-by-ball delivery log enriched with innings phase and flags
-- ------------------------------------------------------------------------------
CREATE TABLE fact_deliveries (
    delivery_id BIGINT PRIMARY KEY,
    match_id INT NOT NULL,
    season VARCHAR(15) NOT NULL,
    innings INT NOT NULL,
    over_number INT NOT NULL, -- 0-indexed over (0 = Over 1)
    ball_number INT NOT NULL, -- 1-indexed ball within over
    ball_formatted DECIMAL(4,1) NOT NULL, -- 0.1, 0.2 ... 19.6
    innings_phase VARCHAR(20) NOT NULL, -- 'Powerplay' (0-5), 'Middle' (6-14), 'Death' (15-19)
    batting_team VARCHAR(100) NOT NULL,
    bowling_team VARCHAR(100) NOT NULL,
    striker VARCHAR(100) NOT NULL,
    non_striker VARCHAR(100) NOT NULL,
    bowler VARCHAR(100) NOT NULL,
    runs_off_bat INT NOT NULL DEFAULT 0,
    extras INT NOT NULL DEFAULT 0,
    wides INT NOT NULL DEFAULT 0,
    noballs INT NOT NULL DEFAULT 0,
    byes INT NOT NULL DEFAULT 0,
    legbyes INT NOT NULL DEFAULT 0,
    penalty INT NOT NULL DEFAULT 0,
    total_runs INT NOT NULL DEFAULT 0,
    is_dot_ball SMALLINT NOT NULL DEFAULT 0, -- 1 if total_runs = 0 or 0 runs off bat with 0 extras
    is_boundary_4 SMALLINT NOT NULL DEFAULT 0,
    is_boundary_6 SMALLINT NOT NULL DEFAULT 0,
    is_wicket SMALLINT NOT NULL DEFAULT 0,
    wicket_type VARCHAR(50),
    player_dismissed VARCHAR(100),
    fielder_1 VARCHAR(100),
    fielder_2 VARCHAR(100),
    FOREIGN KEY (match_id) REFERENCES dim_match(match_id)
);

CREATE INDEX idx_deliv_match_inn ON fact_deliveries(match_id, innings);
CREATE INDEX idx_deliv_striker ON fact_deliveries(striker);
CREATE INDEX idx_deliv_bowler ON fact_deliveries(bowler);
CREATE INDEX idx_deliv_phase ON fact_deliveries(innings_phase);
CREATE INDEX idx_deliv_batteam ON fact_deliveries(batting_team);
CREATE INDEX idx_deliv_bowlteam ON fact_deliveries(bowling_team);

-- ------------------------------------------------------------------------------
-- 5. FACT TABLE: fact_player_match
-- Aggregated individual player performance record per match
-- ------------------------------------------------------------------------------
CREATE TABLE fact_player_match (
    player_match_id BIGINT PRIMARY KEY,
    match_id INT NOT NULL,
    season VARCHAR(15) NOT NULL,
    match_date DATE NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    team VARCHAR(100) NOT NULL,
    opponent VARCHAR(100) NOT NULL,
    venue VARCHAR(255) NOT NULL,
    
    -- Batting metrics
    runs_scored INT DEFAULT 0,
    balls_faced INT DEFAULT 0,
    fours INT DEFAULT 0,
    sixes INT DEFAULT 0,
    batting_strike_rate DECIMAL(6,2) DEFAULT 0.00,
    is_dismissed SMALLINT DEFAULT 0,
    runs_powerplay INT DEFAULT 0,
    runs_middle INT DEFAULT 0,
    runs_death INT DEFAULT 0,
    
    -- Bowling metrics
    balls_bowled INT DEFAULT 0,
    overs_bowled DECIMAL(4,1) DEFAULT 0.0,
    runs_conceded INT DEFAULT 0,
    wickets_taken INT DEFAULT 0,
    dot_balls_bowled INT DEFAULT 0,
    bowling_economy DECIMAL(5,2) DEFAULT 0.00,
    wickets_powerplay INT DEFAULT 0,
    wickets_middle INT DEFAULT 0,
    wickets_death INT DEFAULT 0,
    
    -- Fielding metrics
    catches INT DEFAULT 0,
    run_outs INT DEFAULT 0,
    
    FOREIGN KEY (match_id) REFERENCES dim_match(match_id)
);

CREATE INDEX idx_fpm_player ON fact_player_match(player_name);
CREATE INDEX idx_fpm_season ON fact_player_match(season);
CREATE INDEX idx_fpm_match ON fact_player_match(match_id);

-- ------------------------------------------------------------------------------
-- 6. FACT TABLE: fact_predictions
-- Model inference outcomes, win probabilities, and evaluation metrics
-- ------------------------------------------------------------------------------
CREATE TABLE fact_predictions (
    prediction_id BIGINT PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- Regression, Classification, Ranking
    prediction_target VARCHAR(100) NOT NULL, -- 'batting_runs', 'match_winner', 'recruitment_score'
    entity_id VARCHAR(100) NOT NULL, -- match_id or player_name
    entity_type VARCHAR(20) NOT NULL, -- 'MATCH', 'PLAYER'
    season VARCHAR(15),
    predicted_value DECIMAL(10,4),
    prediction_prob DECIMAL(6,4),
    actual_value DECIMAL(10,4),
    residual_error DECIMAL(10,4),
    evaluation_split VARCHAR(20) NOT NULL, -- 'TRAIN', 'VAL', 'TEST'
    model_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pred_model ON fact_predictions(model_name);
CREATE INDEX idx_pred_target ON fact_predictions(prediction_target);
CREATE INDEX idx_pred_split ON fact_predictions(evaluation_split);
