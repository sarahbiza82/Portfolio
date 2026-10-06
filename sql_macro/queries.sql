-- Macro & markets analytics in SQL (SQLite dialect; window functions and CTEs need SQLite >= 3.25).
-- Each block starts with "-- Q<n>:" so run_queries.py can execute them one by one.

-- Q1: Year-on-year US CPI inflation and its 12-month rolling mean (LAG and window frame)
WITH infl AS (
    SELECT month, cpi,
           100.0 * (cpi / LAG(cpi, 12) OVER (ORDER BY month) - 1) AS infl_yoy
    FROM us_monthly
)
SELECT month, ROUND(infl_yoy, 2) AS infl_yoy,
       ROUND(AVG(infl_yoy) OVER (ORDER BY month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW), 2) AS infl_12m_mean
FROM infl
WHERE infl_yoy IS NOT NULL
ORDER BY month DESC
LIMIT 8;

-- Q2: Yield-curve inversions (10y - 3m < 0 at month end) and recessions in the following 12 months
WITH month_end AS (                       -- last trading day of each month
    SELECT MAX(date) AS date FROM us_yields_daily GROUP BY substr(date, 1, 7)
), curve AS (
    SELECT substr(y.date, 1, 7) AS ym, y.y_10y - y.y_3m AS slope
    FROM us_yields_daily y JOIN month_end m ON y.date = m.date
), rec AS (
    SELECT substr(month, 1, 7) AS ym, recession,
           MAX(recession) OVER (ORDER BY month ROWS BETWEEN 1 FOLLOWING AND 12 FOLLOWING) AS rec_next_12m
    FROM us_monthly
)
SELECT CASE WHEN slope < 0 THEN 'inverted' ELSE 'normal' END AS curve_state,
       COUNT(*) AS months,
       SUM(rec_next_12m) AS followed_by_recession,
       ROUND(100.0 * SUM(rec_next_12m) / COUNT(*), 1) AS pct_followed
FROM curve JOIN rec USING (ym)
WHERE rec_next_12m IS NOT NULL
GROUP BY curve_state;

-- Q3: SQL version of the event study: spread move from the day before to the day after each political date
WITH ordered AS (
    SELECT date, spread_bp,
           LAG(spread_bp, 1)  OVER (ORDER BY date) AS prev_day,
           LEAD(spread_bp, 1) OVER (ORDER BY date) AS next_day
    FROM fr_de_daily
)
SELECT e.date, e.label, ROUND(o.next_day - o.prev_day, 1) AS move_bp_2d
FROM political_events e JOIN ordered o ON o.date = e.date
ORDER BY e.date;

-- Q4: Average level and daily volatility of the spread by year (SQLite has no STDDEV, so variance = E[x^2] - E[x]^2; SQRT is registered by run_queries.py)
WITH d AS (
    SELECT substr(date, 1, 4) AS yr, spread_bp,
           spread_bp - LAG(spread_bp) OVER (ORDER BY date) AS d_spread
    FROM fr_de_daily
)
SELECT yr, ROUND(AVG(spread_bp), 1) AS mean_spread_bp,
       ROUND(SQRT(AVG(d_spread * d_spread) - AVG(d_spread) * AVG(d_spread)), 2) AS daily_vol_bp,
       RANK() OVER (ORDER BY AVG(d_spread * d_spread) - AVG(d_spread) * AVG(d_spread) DESC) AS vol_rank
FROM d
WHERE d_spread IS NOT NULL
GROUP BY yr
ORDER BY vol_rank
LIMIT 6;

-- Q5: The 8 largest two-day spread widenings and whether a political event falls inside the window
WITH moves AS (
    SELECT date, spread_bp - LAG(spread_bp, 2) OVER (ORDER BY date) AS move_2d,
           LAG(date, 2) OVER (ORDER BY date) AS window_start
    FROM fr_de_daily
)
SELECT date, ROUND(move_2d, 1) AS move_2d_bp,
       CASE WHEN EXISTS (SELECT 1 FROM political_events e WHERE e.date > m.window_start AND e.date <= m.date)
            THEN 'political event in window' ELSE '-' END AS context
FROM moves m
WHERE move_2d IS NOT NULL
ORDER BY move_2d DESC
LIMIT 8;

-- Q6: Fed funds regime table: average unemployment and industrial production by 2-point fed-funds bucket
SELECT CAST(fedfunds / 2 AS INT) * 2 AS ffr_bucket_from,
       COUNT(*) AS months,
       ROUND(AVG(unrate), 2) AS avg_unrate,
       ROUND(AVG(indpro), 1) AS avg_ip_index
FROM us_monthly
GROUP BY ffr_bucket_from
ORDER BY ffr_bucket_from;
