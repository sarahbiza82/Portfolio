# Macro-econometrics & market forecasting

Five Python notebooks on public data. Each one is executed, with its outputs visible on GitHub, and states its limits. Forecasts are evaluated out of sample. Some results are negative: a forecast that does not beat a simple benchmark is reported as such.

| # | Project | Methods | Data |
|---|---------|---------|------|
| 01 | [US inflation forecasting, out of sample](01_inflation_forecasting_oos.ipynb) | Direct expanding-window forecasts, AR / Phillips / ridge / lasso / random forest, forecast combination, Diebold-Mariano (HLN) | FRED |
| 02 | [Monetary policy vs central-bank information shocks](02_monetary_shocks_local_projections.ipynb) | High-frequency shocks (Jarocinski-Karadi), local projections, HAC bands, ZLB robustness | Fed / ECB shocks, FRED, ECB, Yahoo |
| 03 | [Dynamic Nelson-Siegel and the yield curve](03_nelson_siegel_forecasting.ipynb) | Diebold-Li factors, direct AR/VAR factor forecasts, Estrella-Mishkin logit, out-of-sample AUC | FRED |
| 04 | [OAT-Bund spread](04_oat_bund_spread.ipynb) | ARDL / UECM bounds test, event study with empirical percentiles, local projections | Banque de France, Bundesbank, Eurostat, ECB, EPU |
| 05 | [Nowcasting French GDP](05_gdp_nowcasting_france.ipynb) | Bridge equations, ridge, PCA factor, ragged edge, pseudo real-time evaluation | Eurostat |

## What the results say

- **01 Inflation.** Before 2020 a random forest has a lower RMSE than the AR benchmark by 9 to 19% at horizons of 3 to 12 months (significant at 5% for 3 and 6 months, at 10% for 12 months). Ridge and lasso gain 3% at most and the Phillips curve gains nothing. Two facts limit the result. First, 62 to 85% of the forest's squared-error gain comes from forecasts made in 2008-2009. Second, the 12-month average of past inflation also beats the AR at 6 and 12 months (by 7 and 13%), and without 2008-2009 the forest does no better than that average (RMSE ratios of 1.00 to 1.04, not tested). After 2020 no model clearly beats the AR.
- **02 Shocks.** On impact, a Fed monetary policy shock of one standard deviation lowers US stocks by 1.1% and an ECB one lowers euro-area industrial production by 0.5% (0.85% after a year). The 90% bands exclude zero. Price responses to the policy shock are not distinguishable from zero at any horizon. The information shock moves stocks the other way on impact: +1.25% for the ECB (90% band excludes zero) and +0.31% for the Fed (band includes zero). Stock responses to the ECB policy shock turn positive after a year. The Euro Stoxx series starts in 2007, so they are not interpreted.
- **03 Yield curve.** Three factors fit nine maturities with errors of 4 to 15 bp, but over the test sample (2005 to 2026) no factor model beats a random walk (RMSE ratios 0.94 to 1.17, none significant). By sub-period, every factor forecast is worse than the random walk in 2005-2019. From 2020 the factor VAR has lower errors than the random walk for the 3-month yield (ratios 0.74 at one month, 0.83 at six months) and the 10-year yield (0.92 to 0.98). These sub-period ratios are not tested and rest on 69 to 80 forecasts. The slope-based recession signal has an in-sample AUC of 0.72 and an out-of-sample AUC of 0.54 before 2020 and 0.50 over the full sample. The sample has three recessions, and the 2022-2024 inversion was not followed by one.
- **04 OAT-Bund.** The bounds test does not reject "no cointegration" (F = 1.60), so no long-run claim is made. Event study: the June 2024 dissolution moved the spread by +15 bp over two days (z = 8.8 against the previous year of two-day moves), the April 2017 first round by -17 bp and the August 2025 announcement of a confidence vote by +9 bp. The censure vote of December 2024 and the confidence vote of September 2025 moved it by 1 to 2 bp: they were priced in. An uncertainty innovation of one standard deviation raises the spread by about 1 bp on impact and 1.4 to 2.8 bp after 4 to 9 months.
- **05 Nowcasting.** Without the COVID quarters, the models beat an AR(1) by 6 to 12%. Against the historical mean of GDP growth the gain is 0 to 5% and not significant, and one model is 1% worse. Adding months 2 and 3 of survey data changes almost nothing. In this design, free survey data add little to the mean for French GDP.

## Reproduce

```bash
pip install -r requirements.txt
python src/build_nb.py nb_src/01_inflation_forecasting_oos.py     # rebuilds and executes one notebook
```

All downloads are cached in `data/` (delete a file to refresh). The `nb_src/*.py` files are the notebook sources in percent format; `src/build_nb.py` converts and executes them. Notebook 01 takes about 20 minutes on an 8-core laptop (nested cross-validation), the others about a minute each.

## Data sources and caveats

- FRED public csv (US macro, Treasury yields, VIX, NBER recession dates).
- French 10-year yield: Banque de France TEC10, interpolated from prices taken at 11 a.m. German 10-year yield: Bundesbank Svensson zero-coupon curve. Both come through DBnomics. They are not the same instrument, so the level of the spread is approximate and one-day changes contain measurement noise. The event study uses two-day changes.
- One day is dropped from the daily spread, 28 June 2022: the series gives a French yield of 1.83% where the Banque de France table of indicative OAT rates gives 2.18%. The rule is in `src/data_utils.py`: a one-day move of the spread above 20 bp is dropped when at least three quarters of it is undone the next day.
- Event dates. News released over a weekend is dated on the Monday. The confidence vote announced in the afternoon of 25 August 2025 is dated on the 26th. Evening votes are dated on the day of the vote. In every case the two-day window starts before the news and ends after it.
- Eurostat quarterly debt and GDP (3-month publication lag respected for debt), ECB Data Portal, Baker-Bloom-Davis EPU index.
- Fed and ECB shocks: Jarocinski and Karadi (2020) updated series, built from the Altavilla et al. EA-MPD.
- All macro series are the latest vintage, not real time.
