-- ==============================================================================
-- PROJECT: Beyond the Scoreboard: Cricket Franchise Decision Intelligence System
-- SCRIPT: 03_analytical_queries.sql - 10 Core Franchise Business Queries
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- QUERY 1: Team Performance & Toss Advantage: Batting-First vs Chasing Win Rates
-- Business Purpose: Identify franchise tactical bias and whether winning toss dictates match outcome
-- ------------------------------------------------------------------------------
WITH MatchTossCTE AS (
    SELECT 
        match_id, season, team1, team2, toss_winner, toss_decision, winner,
        CASE 
            WHEN (toss_winner = team1 AND toss_decision = 'bat') OR (toss_winner = team2 AND toss_decision = 'field') THEN team1
            ELSE team2
        END AS bat_first_team
    FROM dim_match
    WHERE winner IS NOT NULL AND winner != ''
)
SELECT 
    winner AS franchise,
    COUNT(*) AS total_wins,
    SUM(CASE WHEN winner = bat_first_team THEN 1 ELSE 0 END) AS wins_batting_first,
    SUM(CASE WHEN winner != bat_first_team THEN 1 ELSE 0 END) AS wins_chasing,
    ROUND(CAST(SUM(CASE WHEN winner != bat_first_team THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) AS chase_win_pct
FROM MatchTossCTE
GROUP BY winner
HAVING COUNT(*) >= 20
ORDER BY total_wins DESC;

-- ------------------------------------------------------------------------------
-- QUERY 2: Phase-Wise Scoring Progression Across Seasons (Powerplay, Middle, Death)
-- Business Purpose: Track meta-evolution of T20 cricket scoring aggression over 19 seasons
-- ------------------------------------------------------------------------------
SELECT 
    season,
    innings_phase,
    COUNT(*) AS total_balls,
    SUM(total_runs) AS phase_runs,
    SUM(is_wicket) AS phase_wickets,
    ROUND(CAST(SUM(total_runs) AS FLOAT) / (COUNT(*) / 6.0), 2) AS run_rate_per_over,
    ROUND(CAST(SUM(is_boundary_4 + is_boundary_6) AS FLOAT) / COUNT(*) * 100, 2) AS boundary_pct,
    ROUND(CAST(SUM(is_dot_ball) AS FLOAT) / COUNT(*) * 100, 2) AS dot_ball_pct
FROM fact_deliveries
WHERE innings IN (1, 2)
GROUP BY season, innings_phase
ORDER BY season DESC, 
    CASE innings_phase 
        WHEN 'Powerplay' THEN 1 
        WHEN 'Middle' THEN 2 
        WHEN 'Death' THEN 3 
    END;

-- ------------------------------------------------------------------------------
-- QUERY 3: Elite Batting Consistency & Boundary Contribution (Min 1000 Runs)
-- Business Purpose: Balance high-ceiling explosive batters against reliable strike rotators
-- ------------------------------------------------------------------------------
WITH PlayerBatCTE AS (
    SELECT 
        player_name,
        COUNT(DISTINCT match_id) AS innings_batted,
        SUM(runs_scored) AS total_runs,
        SUM(balls_faced) AS total_balls,
        SUM(fours) AS total_fours,
        SUM(sixes) AS total_sixes,
        SUM(is_dismissed) AS times_out,
        ROUND(AVG(runs_scored), 2) AS avg_runs_per_innings,
        ROUND(
            SQRT(AVG(runs_scored * runs_scored) - (AVG(runs_scored) * AVG(runs_scored))), 
            2
        ) AS runs_std_dev
    FROM fact_player_match
    WHERE balls_faced > 0
    GROUP BY player_name
)
SELECT 
    player_name,
    innings_batted,
    total_runs,
    ROUND(CAST(total_runs AS FLOAT) / NULLIF(times_out, 0), 2) AS batting_average,
    ROUND(CAST(total_runs AS FLOAT) / total_balls * 100, 2) AS career_strike_rate,
    ROUND(CAST((total_fours * 4 + total_sixes * 6) AS FLOAT) / total_runs * 100, 2) AS boundary_run_pct,
    avg_runs_per_innings,
    runs_std_dev,
    ROUND(runs_std_dev / NULLIF(avg_runs_per_innings, 0), 2) AS coefficient_of_variation
FROM PlayerBatCTE
WHERE total_runs >= 1000
ORDER BY total_runs DESC
LIMIT 15;

-- ------------------------------------------------------------------------------
-- QUERY 4: Death-Overs Bowling Specialists (Overs 15-20, Min 150 Balls Bowled)
-- Business Purpose: Identify elite pace/spin bowlers who suppress runs at the death
-- ------------------------------------------------------------------------------
SELECT 
    bowler AS player_name,
    COUNT(*) AS death_balls_bowled,
    ROUND(COUNT(*) / 6.0, 1) AS death_overs,
    SUM(bowler_runs) AS death_runs_conceded,
    SUM(CASE WHEN is_wicket = 1 AND wicket_type NOT IN ('run out', 'retired hurt') THEN 1 ELSE 0 END) AS death_wickets,
    ROUND(CAST(SUM(bowler_runs) AS FLOAT) / (COUNT(*) / 6.0), 2) AS death_economy_rate,
    ROUND(CAST(SUM(is_dot_ball) AS FLOAT) / COUNT(*) * 100, 2) AS dot_ball_pct,
    ROUND(CAST(SUM(is_boundary_4 + is_boundary_6) AS FLOAT) / COUNT(*) * 100, 2) AS boundary_concession_pct
FROM fact_deliveries
WHERE innings_phase = 'Death' AND innings IN (1, 2)
GROUP BY bowler
HAVING COUNT(*) >= 150
ORDER BY death_economy_rate ASC
LIMIT 15;

-- ------------------------------------------------------------------------------
-- QUERY 5: Venue Strategic Profiles & Chase Bias (Min 25 Matches)
-- Business Purpose: Guide toss decisions and squad selection tailored to stadium dimensions and pitch pace
-- ------------------------------------------------------------------------------
WITH VenueCTE AS (
    SELECT 
        m.venue,
        m.match_id,
        m.winner,
        CASE 
            WHEN (m.toss_winner = m.team1 AND m.toss_decision = 'bat') OR (m.toss_winner = m.team2 AND m.toss_decision = 'field') THEN m.team1
            ELSE m.team2
        END AS bat_first_team,
        (SELECT SUM(d.total_runs) FROM fact_deliveries d WHERE d.match_id = m.match_id AND d.innings = 1) AS inn1_total
    FROM dim_match m
    WHERE m.winner IS NOT NULL AND m.winner != ''
)
SELECT 
    venue,
    COUNT(*) AS matches_played,
    ROUND(AVG(inn1_total), 1) AS avg_1st_innings_score,
    SUM(CASE WHEN winner = bat_first_team THEN 1 ELSE 0 END) AS bat_first_wins,
    SUM(CASE WHEN winner != bat_first_team THEN 1 ELSE 0 END) AS chase_wins,
    ROUND(CAST(SUM(CASE WHEN winner != bat_first_team THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 1) AS chase_win_pct
FROM VenueCTE
WHERE inn1_total IS NOT NULL
GROUP BY venue
HAVING COUNT(*) >= 25
ORDER BY matches_played DESC;

-- ------------------------------------------------------------------------------
-- QUERY 6: Head-to-Head Franchise Dominance Matrix (CSK vs MI, KKR, RCB)
-- Business Purpose: Provide opposition-specific tactical preparation for marquee rivalries
-- ------------------------------------------------------------------------------
SELECT 
    CASE WHEN team1 < team2 THEN team1 ELSE team2 END AS franchise_a,
    CASE WHEN team1 < team2 THEN team2 ELSE team1 END AS franchise_b,
    COUNT(*) AS matches_played,
    SUM(CASE WHEN winner = CASE WHEN team1 < team2 THEN team1 ELSE team2 END THEN 1 ELSE 0 END) AS wins_a,
    SUM(CASE WHEN winner = CASE WHEN team1 < team2 THEN team2 ELSE team1 END THEN 1 ELSE 0 END) AS wins_b,
    SUM(CASE WHEN winner IS NULL OR winner NOT IN (team1, team2) THEN 1 ELSE 0 END) AS no_results
FROM dim_match
GROUP BY franchise_a, franchise_b
HAVING COUNT(*) >= 15
ORDER BY matches_played DESC
LIMIT 10;

-- ------------------------------------------------------------------------------
-- QUERY 7: Powerplay Batting Dominance: Boundary Aggression vs Wicket Preservation
-- Business Purpose: Identify openers who maximize the 6-over fielding restrictions without exposing middle order
-- ------------------------------------------------------------------------------
SELECT 
    striker AS player_name,
    COUNT(*) AS powerplay_balls_faced,
    SUM(runs_off_bat) AS powerplay_runs,
    SUM(is_wicket) AS powerplay_outs,
    ROUND(CAST(SUM(runs_off_bat) AS FLOAT) / COUNT(*) * 100, 2) AS powerplay_strike_rate,
    ROUND(CAST(SUM(is_dot_ball) AS FLOAT) / COUNT(*) * 100, 2) AS powerplay_dot_pct,
    ROUND(CAST(SUM(is_boundary_4 + is_boundary_6) AS FLOAT) / COUNT(*) * 100, 2) AS powerplay_boundary_pct
FROM fact_deliveries
WHERE innings_phase = 'Powerplay' AND innings IN (1, 2)
GROUP BY striker
HAVING COUNT(*) >= 300
ORDER BY powerplay_strike_rate DESC
LIMIT 15;

-- ------------------------------------------------------------------------------
-- QUERY 8: Middle-Overs Spin/Pace Stranglehold (Overs 7-14)
-- Business Purpose: Measure ability to break partnerships and restrict run flow during building phase
-- ------------------------------------------------------------------------------
SELECT 
    bowler AS player_name,
    COUNT(*) AS middle_balls_bowled,
    SUM(bowler_runs) AS middle_runs_conceded,
    SUM(CASE WHEN is_wicket = 1 AND wicket_type NOT IN ('run out', 'retired hurt') THEN 1 ELSE 0 END) AS middle_wickets,
    ROUND(CAST(SUM(bowler_runs) AS FLOAT) / (COUNT(*) / 6.0), 2) AS middle_economy,
    ROUND(CAST(COUNT(*) AS FLOAT) / NULLIF(SUM(CASE WHEN is_wicket = 1 AND wicket_type NOT IN ('run out', 'retired hurt') THEN 1 ELSE 0 END), 0), 1) AS bowling_strike_rate,
    ROUND(CAST(SUM(is_dot_ball) AS FLOAT) / COUNT(*) * 100, 2) AS dot_ball_pct
FROM fact_deliveries
WHERE innings_phase = 'Middle' AND innings IN (1, 2)
GROUP BY bowler
HAVING COUNT(*) >= 300
ORDER BY middle_economy ASC
LIMIT 15;

-- ------------------------------------------------------------------------------
-- QUERY 9: Finisher Leverage: High-Impact Cameos in Victories
-- Business Purpose: Quantify late-order batting that flips close match outcomes
-- ------------------------------------------------------------------------------
WITH DeathBatCTE AS (
    SELECT 
        d.match_id,
        d.striker AS player_name,
        d.batting_team,
        SUM(d.runs_off_bat) AS death_runs,
        COUNT(*) AS death_balls,
        ROUND(CAST(SUM(d.runs_off_bat) AS FLOAT) / COUNT(*) * 100, 1) AS death_sr
    FROM fact_deliveries d
    WHERE d.innings_phase = 'Death'
    GROUP BY d.match_id, d.striker, d.batting_team
    HAVING COUNT(*) >= 10
)
SELECT 
    b.player_name,
    COUNT(*) AS high_leverage_innings,
    ROUND(AVG(b.death_runs), 1) AS avg_death_runs,
    ROUND(AVG(b.death_sr), 1) AS avg_death_sr,
    SUM(CASE WHEN m.winner = b.batting_team THEN 1 ELSE 0 END) AS matches_won,
    ROUND(CAST(SUM(CASE WHEN m.winner = b.batting_team THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 1) AS win_conversion_pct
FROM DeathBatCTE b
JOIN dim_match m ON b.match_id = m.match_id
GROUP BY b.player_name
HAVING COUNT(*) >= 15
ORDER BY avg_death_sr DESC
LIMIT 12;

-- ------------------------------------------------------------------------------
-- QUERY 10: Squad Role Coverage & Top Recruitment Candidates (From FactPredictions)
-- Business Purpose: Feed recruitment shortlists directly to franchise auction war-room
-- ------------------------------------------------------------------------------
SELECT 
    prediction_target AS tactical_role,
    entity_id AS player_name,
    predicted_value AS performance_recruitment_score,
    DENSE_RANK() OVER (PARTITION BY prediction_target ORDER BY predicted_value DESC) AS role_rank
FROM fact_predictions
WHERE model_name = 'RoleRecruitmentScorer_v1'
ORDER BY prediction_target, role_rank ASC
LIMIT 40;
