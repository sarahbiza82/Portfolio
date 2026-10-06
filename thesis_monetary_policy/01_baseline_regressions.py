"""Baseline regressions: monthly index returns on the policy rate, financial conditions and macro controls.

Five nested specifications for each index (CAC 40, S&P 500, Nasdaq 100), HC3 robust standard errors:
  (1) policy rate only                       (2) + long rate, volatility, regime dummies
  (3) + domestic macro variables, QE         (4) + foreign variables
  (5) as (4), with the return in excess of the risk-free rate

Output: sorties/baseline_models.csv
"""
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
(HERE / "sorties").mkdir(exist_ok=True)
df = pd.read_csv(HERE / "data" / "data_monthly.csv", parse_dates=["date"])

FR = ["d_inflation_FR", "d_oat10y", "d_taux_chômage_FR", "dlog_usd_eur", "dlog_vix", "dlog_prix_petrole", "post2008", "postCOVID"]
US = ["dlog_inflation_US", "d_us10y", "unemployment_rate_US ", "dlog_usd_eur", "dlog_vix", "dlog_prix_petrole", "post2008", "postCOVID"]
FULL = ["policy_rate_ECB_lag2", "d_inflation_FR", "d_oat10y", "d_taux_chômage_FR", "dlog_usd_eur", "dlog_vix", "dlog_prix_petrole",
        "post2008", "postCOVID", "policy_rate_fed", "dlog_inflation_US", "d_us10y", "unemployment_rate_US ", "QE_EU", "QE_US"]
SPECS = {
    "CAC 40": ("r_cac40", "excess_cac40", [["policy_rate_ECB_lag2"], ["policy_rate_ECB_lag2", "d_oat10y", "post2008", "postCOVID", "dlog_vix"],
                                           ["policy_rate_ECB_lag2"] + FR + ["QE_EU"], FULL, FULL]),
    "S&P 500": ("r_sp500", "excess_sp500", [["policy_rate_fed"], ["policy_rate_fed", "d_us10y", "post2008", "postCOVID", "dlog_vix"],
                                            ["policy_rate_fed"] + US + ["QE_US"], FULL, FULL]),
    "Nasdaq 100": ("r_nasdaq100", "excess_nasdaq", [["policy_rate_fed"], ["policy_rate_fed", "d_us10y", "post2008", "postCOVID", "dlog_vix"],
                                                    ["policy_rate_fed"] + US + ["QE_US"], FULL, FULL]),
}
KEY = ["policy_rate_ECB_lag2", "policy_rate_fed", "d_oat10y", "d_us10y", "dlog_vix", "dlog_usd_eur"]


def star(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


rows = []
for idx, (y, yex, models) in SPECS.items():
    for k, xs in enumerate(models, 1):
        r = sm.OLS(df[yex if k == 5 else y], sm.add_constant(df[xs]), missing="drop").fit(cov_type="HC3")
        row = {"index": idx, "model": k, "N": int(r.nobs), "R2": round(r.rsquared, 3)}
        row.update({v: f"{r.params[v]:.4f}{star(r.pvalues[v])}" if v in xs else "" for v in KEY})
        rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(HERE / "sorties" / "baseline_models.csv", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False))


# ------------------------------------------------------------------------------ sectors and stocks (model 4 specification)
ASSETS = {"Luxury": "r_luxe", "Finance": "r_finance", "Industry": "r_industrie", "Technology": "r_technologie", "Health": "r_sante",
          "Energy": "r_energie", "Consumer": "r_consommation", "Autos": "r_automobile", "Telecom": "r_telecom",
          "BNP Paribas": "r_bnp_paribas", "LVMH": "r_lvmh", "Microsoft": "r_microsoft", "Apple": "r_apple"}
VARS = ["policy_rate_ECB_lag2", "policy_rate_fed", "d_oat10y", "d_us10y", "dlog_vix", "dlog_usd_eur"]
rows = []
for name, y in ASSETS.items():
    r = sm.OLS(df[y], sm.add_constant(df[FULL]), missing="drop").fit(cov_type="HC3")
    row = {"asset": name, "R2": round(r.rsquared, 3)}
    row.update({v: f"{r.params[v]:+.4f}{star(r.pvalues[v])}" for v in VARS})
    rows.append(row)
assets_out = pd.DataFrame(rows)
assets_out.to_csv(HERE / "sorties" / "baseline_sectors_stocks.csv", index=False)
print("\nSectors and stocks, full specification (return on all controls, HC3; * 10%, ** 5%, *** 1%)\n", assets_out.to_string(index=False))


# ------------------------------------------------------------------------------ figure: effect of the ECB rate on every series
import matplotlib.pyplot as plt

series = {"CAC 40": "excess_cac40", "S&P 500": "excess_sp500", "Nasdaq 100": "excess_nasdaq", **ASSETS}
est = []
for name, y in series.items():
    r = sm.OLS(df[y], sm.add_constant(df[FULL]), missing="drop").fit(cov_type="HC3")
    est.append((name, 100 * r.params["policy_rate_ECB_lag2"], 100 * 1.645 * r.bse["policy_rate_ECB_lag2"], r.pvalues["policy_rate_ECB_lag2"]))
E = pd.DataFrame(est, columns=["series", "b", "ci", "p"])
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(E.series, E.b, yerr=E.ci, color=["#c0392b" if p < .10 else "#9aa5b1" for p in E.p], ecolor="#555", capsize=3)
ax.axhline(0, color="k", lw=.8); ax.axvline(2.5, color="#555", lw=.8, ls=":")
ax.set_ylabel("monthly return, points")
ax.set_title("Effect of a one-point rise in the ECB policy rate, two months later (full model, 90% interval; red = significant at 10%)", fontsize=10)
plt.xticks(rotation=35, ha="right"); plt.tight_layout()
(HERE / "figures").mkdir(exist_ok=True); plt.savefig(HERE / "figures" / "ecb_rate_effect.png"); plt.close()
