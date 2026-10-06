"""Diagnostics of the OLS models, at the 10% level, and what they imply for inference.

For each index, the full model (model 5, excess returns) is estimated by OLS and tested for:
  normality of residuals (Jarque-Bera), functional form (RESET), heteroskedasticity (Breusch-Pagan),
  serial correlation (Breusch-Godfrey, 12 lags), volatility clustering (ARCH-LM, 6 lags),
  multicollinearity (largest VIF), influential observations (largest Cook distance).
It also compares classical, HC3 and HAC standard errors for the policy-rate coefficients, and runs two
unit-root tests with opposite null hypotheses (ADF: unit root, KPSS: stationarity) on the regressors kept in level.

Output: sorties/diagnostics.csv, sorties/standard_errors.csv, sorties/unit_root_tests.csv
"""
import warnings
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_arch, het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tsa.stattools import adfuller, kpss

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "data" / "data_monthly.csv", parse_dates=["date"])
FULL = ["policy_rate_ECB_lag2", "d_inflation_FR", "d_oat10y", "d_taux_chômage_FR", "dlog_usd_eur", "dlog_vix", "dlog_prix_petrole",
        "post2008", "postCOVID", "policy_rate_fed", "dlog_inflation_US", "d_us10y", "unemployment_rate_US ", "QE_EU", "QE_US"]
TARGETS = {"CAC 40": "excess_cac40", "S&P 500": "excess_sp500", "Nasdaq 100": "excess_nasdaq"}
ALPHA = 0.10
pd.set_option("display.width", 250)


def verdict(p):
    return f"{p:.3f} " + ("(rejected)" if p < ALPHA else "(not rejected)")


rows, se_rows = [], []
for label, y in TARGETS.items():
    X = sm.add_constant(df[FULL]); m = sm.OLS(df[y], X).fit(); e = m.resid
    rows.append({
        "index": label, "N": int(m.nobs), "R2": round(m.rsquared, 3),
        "normality (Jarque-Bera)": verdict(stats.jarque_bera(e).pvalue),
        "functional form (RESET)": verdict(linear_reset(m, power=2, use_f=True).pvalue),
        "homoskedasticity (Breusch-Pagan)": verdict(het_breuschpagan(e, X)[1]),
        "no serial correlation (Breusch-Godfrey)": verdict(acorr_breusch_godfrey(m, nlags=12)[1]),
        "no ARCH effects (ARCH-LM)": verdict(het_arch(e, nlags=6)[1]),
        "max VIF": round(max(vif(X.values, i) for i in range(1, X.shape[1])), 2),
        "max Cook distance": round(m.get_influence().cooks_distance[0].max(), 3),
        "excess kurtosis of residuals": round(stats.kurtosis(e), 2)})
    for v in ["policy_rate_ECB_lag2", "policy_rate_fed"]:
        r = {"index": label, "variable": v, "coef": round(m.params[v], 4)}
        for name, kw in [("classical", dict()), ("HC3", dict(cov_type="HC3")), ("HAC (6 lags)", dict(cov_type="HAC", cov_kwds={"maxlags": 6}))]:
            f = sm.OLS(df[y], X).fit(**kw); r[f"p, {name}"] = round(f.pvalues[v], 4)
        se_rows.append(r)
D = pd.DataFrame(rows); D.to_csv(HERE / "sorties" / "diagnostics.csv", index=False)
S = pd.DataFrame(se_rows); S.to_csv(HERE / "sorties" / "standard_errors.csv", index=False)
print(D.T.to_string(header=False))
print("\np-values of the policy-rate coefficients under three covariance estimators\n", S.to_string(index=False))

# unit-root tests on the regressors kept in level (constant, no trend). The KPSS table stops at 0.01 and 0.10: p-values are bounded there.
ur_rows = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, col in [("ECB rate (lag 2)", "policy_rate_ECB_lag2"), ("Fed rate", "policy_rate_fed"), ("US unemployment rate", "unemployment_rate_US ")]:
        s = df[col].dropna(); a = adfuller(s, regression="c", autolag="AIC"); k = kpss(s, regression="c", nlags="auto")
        ur_rows.append({"series": name, "ADF statistic": round(a[0], 2), "ADF p (H0: unit root)": verdict(a[1]),
                        "KPSS statistic": round(k[0], 2), "KPSS p (H0: stationarity)": verdict(k[1])})
U = pd.DataFrame(ur_rows); U.to_csv(HERE / "sorties" / "unit_root_tests.csv", index=False)
print("\nUnit-root tests on the regressors kept in level\n", U.to_string(index=False))
