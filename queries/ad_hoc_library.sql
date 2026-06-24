-- =============================================================================
-- queries/ad_hoc_library.sql
-- PatientPoint Campaign Intelligence — Ad-Hoc Query Library
--
-- 8 business queries covering the core analytics use cases a PatientPoint
-- Data Analyst handles daily. Each query is documented with:
--   BUSINESS QUESTION — what a stakeholder would actually ask
--   WHEN TO USE      — the trigger scenario (Slack message, weekly review, etc.)
--   INTERVIEW NOTE   — what this query proves to the hiring panel
--
-- All queries run against campaign_data.db (SQLite).
-- To run interactively: sqlite3 data/campaign_data.db < queries/ad_hoc_library.sql
-- To run one query:     copy/paste into the sqlite3 shell or Agent 1's prompt
-- =============================================================================


-- =============================================================================
-- QUERY 1: Campaign Performance vs Weekly Benchmark
-- =============================================================================
-- BUSINESS QUESTION: Which active campaigns are currently underperforming their
--   KPI targets, and by how much?
-- WHEN TO USE: Monday morning weekly review. First query run before the team
--   meeting to know which campaigns need a client call.
-- INTERVIEW NOTE: Demonstrates LEFT JOIN + HAVING + calculated delta column —
--   the exact pattern for "identify trends and patterns to drive decisions."
-- =============================================================================

SELECT
    c.campaign_id,
    c.name                                                AS campaign_name,
    c.health_condition,
    wb.week_start,
    wb.target_completion_rate                             AS kpi_target,
    wb.national_avg,
    ROUND(AVG(dm.completion_rate), 4)                     AS actual_completion,
    ROUND(AVG(dm.completion_rate) - wb.target_completion_rate, 4) AS delta_vs_target,
    CASE
        WHEN AVG(dm.completion_rate) < wb.target_completion_rate * 0.85 THEN 'Critical'
        WHEN AVG(dm.completion_rate) < wb.target_completion_rate * 0.95 THEN 'At Risk'
        ELSE 'On Track'
    END                                                   AS status
FROM daily_metrics dm
JOIN campaign_placements cp  ON dm.placement_id  = cp.placement_id
JOIN campaigns c             ON cp.campaign_id   = c.campaign_id
JOIN weekly_benchmarks wb    ON c.campaign_id    = wb.campaign_id
WHERE wb.week_start = (
    SELECT MAX(week_start) FROM weekly_benchmarks
)
  AND dm.date BETWEEN wb.week_start AND wb.week_end
  AND dm.impressions > 0
GROUP BY c.campaign_id, wb.week_start
HAVING AVG(dm.completion_rate) < wb.target_completion_rate
ORDER BY delta_vs_target ASC
LIMIT 20;


-- =============================================================================
-- QUERY 2: Office-Level Underperformers by Region and Specialty
-- =============================================================================
-- BUSINESS QUESTION: Which cardiology offices in the Midwest had the lowest
--   completion rate last month, and how far are they from the national average?
-- WHEN TO USE: When a client success manager pings: "Something looks off in
--   the Midwest for our cardiology campaign — can you pull the offices?"
-- INTERVIEW NOTE: This is the exact demo query for Tab 1 of the Streamlit app.
--   Shows multi-table JOIN + regional filter + benchmark comparison.
-- =============================================================================

SELECT
    po.office_id,
    po.name                                               AS office_name,
    po.state,
    po.region,
    po.specialty,
    po.tier,
    ROUND(AVG(dm.completion_rate), 4)                     AS avg_completion_rate,
    ROUND(AVG(wb.national_avg), 4)                        AS national_avg,
    ROUND(AVG(dm.completion_rate) - AVG(wb.national_avg), 4) AS delta_vs_national,
    SUM(dm.impressions)                                   AS total_impressions
FROM daily_metrics dm
JOIN campaign_placements cp  ON dm.placement_id = cp.placement_id
JOIN physician_offices po    ON cp.office_id    = po.office_id
JOIN campaigns c             ON cp.campaign_id  = c.campaign_id
JOIN weekly_benchmarks wb    ON c.campaign_id   = wb.campaign_id
WHERE po.specialty  = 'cardiology'
  AND po.region     = 'Midwest'
  AND dm.date      >= DATE('now', '-30 days')
  AND dm.impressions > 0
GROUP BY po.office_id
ORDER BY avg_completion_rate ASC
LIMIT 20;


-- =============================================================================
-- QUERY 3: Week-over-Week Completion Rate Trend (Window Function)
-- =============================================================================
-- BUSINESS QUESTION: How has Campaign X trended week-over-week over the last
--   8 weeks? Is the drop this week a one-time blip or a consistent decline?
-- WHEN TO USE: Before a client call when a pharma brand asks "is our campaign
--   getting worse over time or was last week just a bad week?"
-- INTERVIEW NOTE: Uses LAG() window function — the strongest SQL signal for
--   a Data Analyst role. Most candidates can write GROUP BY; fewer use LAG().
-- =============================================================================

WITH weekly_performance AS (
    SELECT
        c.campaign_id,
        c.name                                            AS campaign_name,
        strftime('%Y-W%W', dm.date)                       AS iso_week,
        MIN(dm.date)                                      AS week_start,
        ROUND(AVG(dm.completion_rate), 4)                 AS avg_completion,
        SUM(dm.impressions)                               AS weekly_impressions
    FROM daily_metrics dm
    JOIN campaign_placements cp ON dm.placement_id = cp.placement_id
    JOIN campaigns c            ON cp.campaign_id  = c.campaign_id
    WHERE c.campaign_id = 'CAMP_035'          -- swap campaign ID as needed
      AND dm.impressions > 0
    GROUP BY c.campaign_id, iso_week
)
SELECT
    campaign_id,
    campaign_name,
    iso_week,
    avg_completion,
    weekly_impressions,
    LAG(avg_completion) OVER (
        PARTITION BY campaign_id ORDER BY iso_week
    )                                                     AS prev_week_completion,
    ROUND(
        avg_completion - LAG(avg_completion) OVER (
            PARTITION BY campaign_id ORDER BY iso_week
        ), 4
    )                                                     AS wow_delta,
    CASE
        WHEN avg_completion < LAG(avg_completion) OVER (
            PARTITION BY campaign_id ORDER BY iso_week
        ) * 0.85 THEN 'Significant Drop'
        WHEN avg_completion < LAG(avg_completion) OVER (
            PARTITION BY campaign_id ORDER BY iso_week
        ) * 0.95 THEN 'Slight Decline'
        WHEN avg_completion > LAG(avg_completion) OVER (
            PARTITION BY campaign_id ORDER BY iso_week
        ) * 1.05 THEN 'Improving'
        ELSE 'Stable'
    END                                                   AS trend_signal
FROM weekly_performance
ORDER BY iso_week DESC
LIMIT 8;


-- =============================================================================
-- QUERY 4: Completion Rate Breakdown by Physician Specialty
-- =============================================================================
-- BUSINESS QUESTION: Which physician specialties respond best to our
--   cardiovascular campaign? Where should we prioritize future placements?
-- WHEN TO USE: Quarterly business review prep. Helps the client understand
--   which specialties are worth expanding into.
-- INTERVIEW NOTE: Demonstrates aggregation + ranking + relative comparison
--   (completion vs national avg) — directly maps to "pricing proposals" and
--   "campaign targeting structures" in the JD.
-- =============================================================================

SELECT
    po.specialty,
    COUNT(DISTINCT cp.office_id)                          AS offices_reached,
    COUNT(DISTINCT cp.placement_id)                       AS total_placements,
    SUM(dm.impressions)                                   AS total_impressions,
    ROUND(AVG(dm.completion_rate), 4)                     AS avg_completion_rate,
    ROUND(AVG(dm.ctr), 4)                                 AS avg_ctr,
    ROUND(SUM(dm.completions) * 1.0 / NULLIF(SUM(dm.impressions), 0), 4)
                                                          AS overall_completion_rate,
    RANK() OVER (
        ORDER BY AVG(dm.completion_rate) DESC
    )                                                     AS performance_rank
FROM daily_metrics dm
JOIN campaign_placements cp  ON dm.placement_id = cp.placement_id
JOIN physician_offices po    ON cp.office_id    = po.office_id
JOIN campaigns c             ON cp.campaign_id  = c.campaign_id
WHERE c.health_condition = 'cardiovascular'
  AND dm.impressions > 0
GROUP BY po.specialty
ORDER BY avg_completion_rate DESC;


-- =============================================================================
-- QUERY 5: Regional Campaign Performance Heatmap Data
-- =============================================================================
-- BUSINESS QUESTION: How does the same campaign perform across regions?
--   Are there geographic patterns we should act on?
-- WHEN TO USE: Building the Power BI regional heatmap. Also the basis for
--   telling a pharma client "your campaign is overperforming in the South
--   but lagging in the Northeast."
-- INTERVIEW NOTE: Cross-tab style aggregation (campaign × region) that feeds
--   directly into the Power BI dashboard — shows you think in reporting layers.
-- =============================================================================

SELECT
    c.health_condition,
    po.region,
    po.tier,
    COUNT(DISTINCT po.office_id)                          AS offices,
    SUM(dm.impressions)                                   AS total_impressions,
    SUM(dm.completions)                                   AS total_completions,
    ROUND(AVG(dm.completion_rate), 4)                     AS avg_completion_rate,
    ROUND(AVG(dm.ctr), 4)                                 AS avg_ctr,
    ROUND(
        SUM(dm.completions) * 1.0 / NULLIF(SUM(dm.impressions), 0) -
        AVG(wb.national_avg), 4
    )                                                     AS delta_vs_national
FROM daily_metrics dm
JOIN campaign_placements cp  ON dm.placement_id = cp.placement_id
JOIN physician_offices po    ON cp.office_id    = po.office_id
JOIN campaigns c             ON cp.campaign_id  = c.campaign_id
JOIN weekly_benchmarks wb    ON c.campaign_id   = wb.campaign_id
WHERE dm.impressions > 0
GROUP BY c.health_condition, po.region, po.tier
ORDER BY c.health_condition, po.region, po.tier;


-- =============================================================================
-- QUERY 6: Top and Bottom Offices for a Given Campaign (Client Report Core)
-- =============================================================================
-- BUSINESS QUESTION: For Campaign X, show me the top 10 and bottom 10 offices
--   by completion rate. I need this for the weekly client report.
-- WHEN TO USE: Agent 3 (Report Writer) calls this pattern to populate the
--   "Office-level highlights" section of the weekly markdown report.
-- INTERVIEW NOTE: UNION ALL with ranked subqueries — intermediate SQL that
--   shows you think in reporting layers, not just single-dimension pulls.
-- =============================================================================

WITH office_performance AS (
    SELECT
        po.office_id,
        po.name                                           AS office_name,
        po.state,
        po.region,
        po.specialty,
        po.tier,
        ROUND(AVG(dm.completion_rate), 4)                 AS avg_completion_rate,
        SUM(dm.impressions)                               AS total_impressions,
        COUNT(DISTINCT dm.date)                           AS days_active
    FROM daily_metrics dm
    JOIN campaign_placements cp ON dm.placement_id = cp.placement_id
    JOIN physician_offices po   ON cp.office_id    = po.office_id
    WHERE cp.campaign_id = 'CAMP_035'         -- swap campaign ID as needed
      AND dm.impressions > 0
    GROUP BY po.office_id
    HAVING days_active >= 7                   -- only offices with meaningful data
)
SELECT 'Top 10'    AS bucket, * FROM office_performance ORDER BY avg_completion_rate DESC LIMIT 10
UNION ALL
SELECT 'Bottom 10' AS bucket, * FROM office_performance ORDER BY avg_completion_rate ASC  LIMIT 10;


-- =============================================================================
-- QUERY 7: 90-Day Rolling Baseline per Placement (Agent 2 Foundation Query)
-- =============================================================================
-- BUSINESS QUESTION: What is the normal performance baseline for each
--   placement so we can detect when something is statistically abnormal?
-- WHEN TO USE: Agent 2 (Anomaly Detector) runs this pattern to establish
--   the rolling 90-day baseline before computing Z-scores.
-- INTERVIEW NOTE: Rolling window aggregate — this is the SQL behind the
--   anomaly detection system. Shows you understand how to build a detection
--   layer, not just pull a snapshot.
-- =============================================================================

WITH rolling_baselines AS (
    SELECT
        placement_id,
        date,
        completion_rate,
        AVG(completion_rate) OVER (
            PARTITION BY placement_id
            ORDER BY date
            ROWS BETWEEN 89 PRECEDING AND 1 PRECEDING
        )                                                 AS rolling_90d_avg,
        COUNT(*) OVER (
            PARTITION BY placement_id
            ORDER BY date
            ROWS BETWEEN 89 PRECEDING AND 1 PRECEDING
        )                                                 AS days_in_window
    FROM daily_metrics
    WHERE impressions > 0
)
SELECT
    rb.placement_id,
    rb.date,
    rb.completion_rate                                    AS current_value,
    ROUND(rb.rolling_90d_avg, 4)                          AS baseline,
    ROUND(rb.completion_rate - rb.rolling_90d_avg, 4)     AS deviation,
    rb.days_in_window
FROM rolling_baselines rb
WHERE rb.days_in_window >= 14                 -- need at least 14 days for a valid baseline
  AND rb.rolling_90d_avg IS NOT NULL
ORDER BY ABS(rb.completion_rate - rb.rolling_90d_avg) DESC
LIMIT 50;


-- =============================================================================
-- QUERY 8: Campaign Executive Summary (Full KPI Snapshot)
-- =============================================================================
-- BUSINESS QUESTION: Give me the complete performance summary for a campaign
--   from launch to today — the one-pager I send to the VP before the client call.
-- WHEN TO USE: Pre-client-call prep. Also the data source for Agent 3's
--   executive summary section in the weekly report.
-- INTERVIEW NOTE: Most comprehensive query in the library — single CTE that
--   produces a full campaign scorecard. This is the query you walk through
--   last in the interview: "This is what I'd run before a pharma client call."
-- =============================================================================

WITH campaign_summary AS (
    SELECT
        c.campaign_id,
        c.name                                            AS campaign_name,
        c.health_condition,
        c.specialty_target,
        c.start_date,
        c.end_date,
        c.budget,
        c.target_completion,
        COUNT(DISTINCT cp.office_id)                      AS offices_reached,
        COUNT(DISTINCT cp.placement_id)                   AS total_placements,
        COUNT(DISTINCT po.state)                          AS states_covered,
        SUM(dm.impressions)                               AS total_impressions,
        SUM(dm.completions)                               AS total_completions,
        ROUND(SUM(dm.completions) * 1.0 /
              NULLIF(SUM(dm.impressions), 0), 4)          AS overall_completion_rate,
        ROUND(AVG(dm.ctr), 4)                             AS avg_ctr,
        MIN(dm.date)                                      AS first_data_date,
        MAX(dm.date)                                      AS last_data_date
    FROM campaigns c
    JOIN campaign_placements cp ON c.campaign_id   = cp.campaign_id
    JOIN physician_offices po   ON cp.office_id    = po.office_id
    JOIN daily_metrics dm       ON cp.placement_id = dm.placement_id
    WHERE c.campaign_id = 'CAMP_035'          -- swap campaign ID as needed
      AND dm.impressions > 0
    GROUP BY c.campaign_id
)
SELECT
    campaign_id,
    campaign_name,
    health_condition,
    specialty_target,
    start_date,
    end_date,
    ROUND(budget, 2)                                      AS budget_usd,
    target_completion                                     AS kpi_target,
    offices_reached,
    total_placements,
    states_covered,
    total_impressions,
    total_completions,
    overall_completion_rate,
    ROUND(overall_completion_rate - target_completion, 4) AS delta_vs_kpi,
    CASE
        WHEN overall_completion_rate >= target_completion        THEN 'Exceeding KPI'
        WHEN overall_completion_rate >= target_completion * 0.95 THEN 'Near Target'
        WHEN overall_completion_rate >= target_completion * 0.85 THEN 'Below Target'
        ELSE 'Critical Underperformance'
    END                                                   AS kpi_status,
    avg_ctr,
    first_data_date,
    last_data_date
FROM campaign_summary;
