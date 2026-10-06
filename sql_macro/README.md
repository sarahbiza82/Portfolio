# SQL for macro and market data

A SQLite database built from public macro and market data, and six analytical queries. Everything runs locally.

| Query | SQL techniques | Question |
|---|---|---|
| Q1 | `LAG`, rolling window frame | Year-on-year US CPI inflation and its 12-month mean |
| Q2 | CTEs, `JOIN ... USING`, forward-looking window (`1 FOLLOWING AND 12 FOLLOWING`) | How often does a month-end inverted yield curve precede a recession within a year? |
| Q3 | `LAG`/`LEAD`, join to an events table | Spread move around each political date (matches the Python event study in notebook 04) |
| Q4 | Aggregates, derived variance, `RANK` | Level and daily volatility of the OAT-Bund spread by year |
| Q5 | `EXISTS` subquery, ordering | Largest two-day spread widenings and whether a political event falls in the window |
| Q6 | Bucketing with `CAST`, `GROUP BY` | Unemployment and industrial production by fed funds regime |

Results (queries and output tables): [results.md](results.md).

## Findings

- Of 56 month-ends with an inverted 10y-3m curve, 32.1% were followed by a recession within 12 months, against 12.9% of the 334 other months. The sample covers three recessions.
- Q3 reproduces, in SQL, the two-day spread moves computed in Python (for example +15.0 bp after the 2024 dissolution, -16.8 bp after the 2017 first round). The two implementations give the same numbers.
- The eight largest two-day widenings of the spread all fall between October 2011 and March 2012, during the euro crisis. None of the political dates in the table is among them.

## Run

```bash
pip install pandas requests
python build_db.py       # creates macro.db from the data cached by ../macro_econometrie
python run_queries.py    # runs queries.sql and writes results.md
```

SQLite 3.25 or later is needed for window functions. `SQRT` is registered from Python because some SQLite builds omit it.

## Limits

The French and German 10-year series are not the same instrument (see the macro README), so the level of the spread is approximate and one-day changes contain measurement noise. In 2005 the daily changes of the spread have a first-order autocorrelation of -0.46, close to the -0.5 that pure measurement noise gives, so the volatility that Q4 reports for that year is mostly noise. One day of the French series, 28 June 2022, is dropped as a recording error when the database is built.
