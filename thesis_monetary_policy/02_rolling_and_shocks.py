"""Time variation and announcement surprises.

A. Rolling 84-month estimates of the policy-rate coefficient (HAC standard errors).
B. Monthly returns on high-frequency monetary policy shocks (Jarocinski-Karadi), ECB and Fed, split into a
   monetary policy shock (MP) and a central-bank information shock (CBI), each standardised to one s.d.
C. The same regressions for CAC 40 sectors and four individual stocks.

Output: sorties/*.csv and figures/*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(exist_ok=True); (HERE / "sorties").mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
df = pd.read_csv(HERE / "data" / "data_monthly.csv", parse_dates=["date"]).set_index("date")
pd.set_option("display.width", 250)


def star(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def hac(y, X, lags=3):
    return sm.OLS(y, sm.add_constant(X), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})


# ------------------------------------------------------------------------- A. rolling windows
SPECS = {"CAC 40": ("r_cac40", "policy_rate_ECB_lag2", "d_oat10y", "ECB rate (lag 2)", "OAT 10y"),
         "S&P 500": ("r_sp500", "policy_rate_fed", "d_us10y", "Fed rate", "Treasury 10y"),
         "Nasdaq 100": ("r_nasdaq100", "policy_rate_fed", "d_us10y", "Fed rate", "Treasury 10y")}
WINDOW = 84


def rolling(y, rate, longrate):
    rows = []
    for i in range(WINDOW, len(df) + 1):
        sub = df.iloc[i - WINDOW:i]
        r = hac(sub[y], sub[[rate, longrate, "dlog_vix"]])
        rows.append({"date": sub.index[-1], "sd_rate": sub[rate].std(), "rate_values": sub[rate].nunique(), **{f"{k}_b": r.params[k] for k in (rate, longrate, "dlog_vix")},
                     **{f"{k}_p": r.pvalues[k] for k in (rate, longrate, "dlog_vix")}})
    return pd.DataFrame(rows).set_index("date")


RW = {idx: rolling(y, rate, lr) for idx, (y, rate, lr, _, _) in SPECS.items()}
pd.concat({idx: r.rename(columns=lambda c: c.replace(SPECS[idx][1], "rate").replace(SPECS[idx][2], "long_rate").replace("dlog_vix", "vix"))
           for idx, r in RW.items()}, names=["index"]).round(5).to_csv(HERE / "sorties" / "rolling_coefficients.csv")
fig, axes = plt.subplots(2, 3, figsize=(15, 6.2), sharex=True)
for j, (idx, r) in enumerate(RW.items()):
    _, rate, lr, rate_lab, _ = SPECS[idx]
    for i, (var, lab, lim) in enumerate([(rate, f"{rate_lab} coefficient", 12), ("dlog_vix", "VIX coefficient", 30)]):
        ax = axes[i, j]; b = r[f"{var}_b"] * 100
        ax.plot(r.index, b, color="#1f4e79", lw=1.2)
        sig = r[f"{var}_p"] < 0.10
        ax.scatter(r.index[sig], b[sig], s=6, color="#c0392b", zorder=3)
        ax.axhline(0, color="k", lw=.8); ax.set_ylim(-lim, lim); ax.set_title(f"{idx}: {lab}", fontsize=10)
        if i == 0:
            ax2 = ax.twinx(); ax2.plot(r.index, r.sd_rate, color="#9aa5b1", lw=.8); ax2.set_ylim(0, 6); ax2.grid(False)
            ax2.set_ylabel("s.d. of the rate in the window", fontsize=7, color="#9aa5b1"); ax2.tick_params(labelsize=7)
fig.suptitle("Coefficients in 84-month rolling windows (HAC errors; red = significant at 10%; y-axis cut)", y=1.0)
plt.tight_layout(); plt.savefig(HERE / "figures" / "rolling_windows.png", bbox_inches="tight"); plt.close()

PERIODS = [("windows ending 2007-2015", slice("2007", "2015")), ("2016-2021", slice("2016", "2021")), ("2022-2024", slice("2022", None))]
summ = []
for idx, r in RW.items():
    _, rate, lr, rate_lab, lr_lab = SPECS[idx]
    for var, lab in [(rate, rate_lab), (lr, lr_lab), ("dlog_vix", "VIX")]:
        for per, sl in PERIODS:
            z = r.loc[sl]; sig = z[f"{var}_p"] < 0.10
            summ.append({"index": idx, "variable": lab, "period": per, "windows": len(z), "median coefficient (pts)": round(100 * z[f"{var}_b"].median(), 2),
                         "share significant at 10%": round(float(sig.mean()), 2),
                         "of which negative": round(float((sig & (z[f"{var}_b"] < 0)).mean()), 2), "of which positive": round(float((sig & (z[f"{var}_b"] > 0)).mean()), 2),
                         "median s.d. of the policy rate": round(z.sd_rate.median(), 2)})
RS = pd.DataFrame(summ); RS.to_csv(HERE / "sorties" / "rolling_summary.csv", index=False)
print("A. ROLLING WINDOWS")
print(RS.to_string(index=False))


# ------------------------------------------------------------------------- B. announcement surprises
def jk(fname, tag):
    s = pd.read_csv(HERE / "data" / fname)
    s.index = pd.to_datetime(dict(year=s.year, month=s.month, day=1))
    return s[["MP_median", "CBI_median"]].rename(columns={"MP_median": f"MP_{tag}", "CBI_median": f"CBI_{tag}"})


D = df.join(jk("jk_ecb_m.csv", "ECB")).join(jk("jk_fed_m.csv", "Fed"))
SHOCKS = ["MP_ECB", "CBI_ECB", "MP_Fed", "CBI_Fed"]
D[SHOCKS] = D[SHOCKS] / D[SHOCKS].std()
RET = {"CAC 40": "r_cac40", "S&P 500": "r_sp500", "Nasdaq 100": "r_nasdaq100"}

rows = []
for idx, y in RET.items():
    for lab, mask in [("2000-2024", D.index == D.index), ("before 2008", D.pre2008 == 1), ("2008-2019", D.post2008 == 1), ("after COVID", D.postCOVID == 1)]:
        sub = D[mask]; r = hac(100 * sub[y], sub[SHOCKS])
        row = {"index": idx, "sample": lab, "N": int(r.nobs), "R2": round(r.rsquared, 3)}
        row.update({s: f"{r.params[s]:+.2f}{star(r.pvalues[s])}" for s in SHOCKS}); rows.append(row)
tabB = pd.DataFrame(rows); tabB.to_csv(HERE / "sorties" / "shocks_indices.csv", index=False)
print("\nB. MONTHLY RETURN (%) FOR A ONE-S.D. SHOCK\n", tabB.to_string(index=False))

rows = []
for idx, y in RET.items():
    for lab, xs in [("rates only", ["policy_rate_ECB_lag2", "policy_rate_fed"]), ("rates and shocks", ["policy_rate_ECB_lag2", "policy_rate_fed"] + SHOCKS)]:
        r = hac(100 * D[y], D[xs])
        rows.append({"index": idx, "specification": lab, "ECB rate (lag 2)": f"{r.params['policy_rate_ECB_lag2']:+.2f}{star(r.pvalues['policy_rate_ECB_lag2'])}",
                     "Fed rate": f"{r.params['policy_rate_fed']:+.2f}{star(r.pvalues['policy_rate_fed'])}", "R2": round(r.rsquared, 3)})
tabN = pd.DataFrame(rows); tabN.to_csv(HERE / "sorties" / "rates_with_and_without_shocks.csv", index=False)
print("\nPOLICY-RATE LEVELS WITH AND WITHOUT THE SHOCKS\n", tabN.to_string(index=False))

# ------------------------------------------------------------------------- C. sectors and stocks
assets = {"Luxury": "r_luxe", "Finance": "r_finance", "Industry": "r_industrie", "Technology": "r_technologie", "Health": "r_sante",
          "Energy": "r_energie", "Consumer": "r_consommation", "Autos": "r_automobile", "Telecom": "r_telecom",
          "BNP Paribas": "r_bnp_paribas", "LVMH": "r_lvmh", "Microsoft": "r_microsoft", "Apple": "r_apple"}
rows = []
for name, y in assets.items():
    r = hac(100 * D[y], D[SHOCKS])
    rows += [{"asset": name, "shock": s, "coef": r.params[s], "se": r.bse[s], "p": r.pvalues[s]} for s in SHOCKS]
C = pd.DataFrame(rows); C.to_csv(HERE / "sorties" / "shocks_sectors.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, s, t in zip(axes, ["MP_ECB", "MP_Fed"], ["ECB monetary policy shock", "Fed monetary policy shock"]):
    z = C[C.shock == s].set_index("asset").loc[list(assets)]
    ax.barh(z.index, z.coef, xerr=1.645 * z.se, color=["#c0392b" if p < .10 else "#9aa5b1" for p in z.p], ecolor="#555", capsize=2)
    ax.axvline(0, color="k", lw=.8); ax.invert_yaxis()
    ax.set_title(t + "\nmonthly return (%) per s.d.; red = significant at 10%", fontsize=9)
plt.tight_layout(); plt.savefig(HERE / "figures" / "sectors_and_stocks.png"); plt.close()
print("\nC. SECTORS AND STOCKS (monetary policy shock, % per s.d.)")
print(C[C.shock.isin(["MP_ECB", "MP_Fed"])].assign(v=lambda d: d.coef.map("{:+.2f}".format) + d.p.map(star)).pivot(index="asset", columns="shock", values="v").loc[list(assets)].to_string())
