# Sarah Biza: Portfolio

Site: https://sarahbiza82.github.io/Portfolio/

Contact: sarahbiza82@gmail.com

## About me

I am in the second year of the master's degree in Econometrics and Statistics, applied econometrics track (M2 ECAP), at IAE Nantes.

The programme lasts two years and covers economics, econometrics, statistics and computing. Courses include time series and forecasting, macro and financial econometrics, panel and spatial econometrics, public policy evaluation, credit-risk scoring, biostatistics, machine learning and text mining. Software: R, Python, SQL, Power BI, R Shiny, Stata and QGIS.

I am looking for a six-month end-of-studies internship starting in March 2027 (three months minimum), in econometrics, statistics, data analysis or finance.

## My approach

Every project follows the same four rules.

1. **Out of sample.** A forecast is judged on data the model has not seen, with expanding or rolling windows.
2. **Simple benchmarks.** Each model is compared with a random walk, a mean or an autoregression, and the difference is tested.
3. **Negative results stay.** When a model does not beat its benchmark, the page says so.
4. **Reproducible.** Data downloads are scripted, seeds are fixed, and notebooks keep their outputs.

Each project has its own page with the question, the data, the method, the results and the limits.

## Analyses

Twelve studies in Python, R and SQL, from the master's thesis to course projects.

### Master's thesis

[Impact of Fed and ECB monetary policy on financial returns](thesis_monetary_policy/) (Python, [synthesis note in French](thesis_monetary_policy/NOTE_DE_SYNTHESE.docx)). To what extent do ECB and Fed decisions influence the returns of French and US equity markets? The data cover 297 months from April 2000 to December 2024: the CAC 40, S&P 500 and Nasdaq 100, nine CAC 40 sectors and four stocks.

- A one-point rise in the ECB rate, two months earlier, lowers the monthly return by 1.4 points for the CAC 40, 0.8 for the S&P 500 and 0.9 for the Nasdaq 100.
- The Fed rate in level has no negative effect.
- Long rates act in opposite directions: rising French yields lower the CAC 40 and rising US yields raise it.
- The VIX is the most stable driver.
- The ECB rate is significantly negative for 11 of 13 sectors and stocks, strongest for autos.
- In rolling windows the ECB effect is negative and significant in 95% of the windows ending between 2007 and 2015. Later it can only be estimated when the rate moves.
- Announcement surprises: each market reacts first to its own central bank. The Fed shock also lowers the CAC 40, and the ECB shock does not significantly move US indices.

Several OLS assumptions are rejected at 10%, so all results use robust errors.

### Other analyses

| Project | Field | Tool | Question | Result |
|---|---|---|---|---|
| [US inflation forecasting](macro_econometrie/01_inflation_forecasting_oos.ipynb) | Forecasting | Python | Do ML models beat an AR for US inflation, out of sample? | Before 2020 a random forest has a 9 to 19% lower RMSE than the AR at 3 to 12 months, but most of that gain comes from 2008-2009. After 2020 no model clearly beats the AR. |
| [Monetary policy and information shocks](macro_econometrie/02_monetary_shocks_local_projections.ipynb) | Monetary policy | Python | What do Fed and ECB policy and information shocks do to output, prices and stocks? | Policy shocks lower stocks (Fed) and industrial production (ECB) on impact. Price responses are not distinguishable from zero. |
| [Nelson-Siegel and the yield curve](macro_econometrie/03_nelson_siegel_forecasting.ipynb) | Yield curve | Python | Do yield-curve factors forecast yields and recessions? | Close fit, but no gain over a random walk on the full test sample. The slope signal has an out-of-sample AUC of about 0.5. |
| [OAT-Bund spread](macro_econometrie/04_oat_bund_spread.ipynb) | Sovereign spreads | Python | Fundamentals or politics? | No cointegration found. The 2024 dissolution moved the spread +15 bp in two days (z = 8.8). The later censure and confidence votes were already priced in. |
| [Nowcasting French GDP](macro_econometrie/05_gdp_nowcasting_france.ipynb) | Nowcasting | Python | Do survey data nowcast French GDP? | Better than an AR(1), but only 0 to 5% better than the historical mean. |
| [Factor models: Food and Insurance](eaf_factor_models/) | Asset pricing | Python | Which risk factors are two US industries exposed to, and are the exposures stable? | No abnormal return in any model. Both industries have a structural break, at different dates. |
| [Forecast evaluation: returns and volatility](r_forecast_evaluation/) | Forecast evaluation | R | Can an AR predict wheat returns, and which HAR model forecasts KOSPI 200 volatility best? | No model beats a zero forecast for returns. The HAR in logs beats the random walk for volatility. |
| [Forecasting French inflation](r_inflation_france/) | Forecasting | R | Do oil, energy, the exchange rate and the yield improve on an AR? | Not with past information only. Known future predictors would gain 6 to 29%. |
| [Imputing missing data](r_missing_data/) (group project) | Statistics | R | Which imputation method recovers missing values best on a small dataset? | With 40 firms no method beats mean imputation. Multiple imputation restores the standard errors that listwise deletion inflates. |
| [PCR, PLS and OLS for air quality](r_pls_air_quality/) (group project) | Statistics | R | Do PCR and PLS predict fine particles better than OLS? | PLS matches OLS with half the components but is not more accurate. A random split of hours flatters every model. |
| [SQL for macro and market data](sql_macro/) | Databases | SQL | How often does an inverted yield curve precede a recession? | Six queries on a SQLite database. 32.1% of inverted month-ends were followed by a recession within a year, against 12.9% of the others. |

The five notebooks on public data have a common page: [macro_econometrie/](macro_econometrie/).

## Dashboards

| Tool | Piece | What it shows |
|---|---|---|
| Dash, Plotly | [Thesis dashboard](thesis_monetary_policy/dashboard.py) | Choose an index and a model, see the effects with their 95% intervals and the 84-month rolling coefficients. It runs locally. |
| Power BI | [SkyPort airport dashboard](powerbi_skyport/) (group project) | Five-page report on a fictitious dataset generated with AI: activity and commercial revenue, delays and costs, passenger experience, and a risk index by airline. |

## Reproducibility

The macro notebooks download their data with scripts and cache it. The other projects include their data files. Notebooks keep their outputs, and scripts save their tables and figures in the project folders, so results can be read without running anything.
