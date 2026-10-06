"""Build the stock and sector return columns from dividend-adjusted prices (Yahoo Finance, monthly).

Indices (CAC 40, S&P 500, Nasdaq 100) are price indices: adjusted and raw closes coincide, so they are left as they are.
Individual stocks use the adjusted close; each of the nine sector series is the equal-weighted average of the adjusted log returns of its two to five constituents.
Input: data/data_monthly_close_prices.csv (monthly dataset built on closing prices). Output: data/data_monthly.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "data" / "data_monthly_close_prices.csv", parse_dates=["date"])

STOCKS = {"apple": "AAPL", "microsoft": "MSFT", "lvmh": "MC.PA", "bnp_paribas": "BNP.PA"}
SECTORS = {
    "luxe": ["MC.PA", "RMS.PA", "KER.PA", "OR.PA"], "finance": ["BNP.PA", "GLE.PA", "ACA.PA", "CS.PA"],
    "industrie": ["SU.PA", "AIR.PA", "SAF.PA", "HO.PA", "DG.PA"], "technologie": ["CAP.PA", "DSY.PA", "STMPA.PA"],
    "sante": ["SAN.PA", "EL.PA"], "energie": ["TTE.PA", "ENGI.PA", "VIE.PA"], "consommation": ["CA.PA", "BN.PA", "RI.PA"],
    "automobile": ["RNO.PA", "STLAP.PA", "ML.PA"], "telecom": ["ORA.PA", "PUB.PA"]}
tickers = sorted({t for ts in SECTORS.values() for t in ts} | set(STOCKS.values()))
raw = yf.download(tickers, start="2000-01-01", end="2024-12-31", interval="1mo", auto_adjust=False, progress=False)
close, adj = raw["Close"], raw["Adj Close"]
close.index = adj.index = pd.to_datetime(close.index).tz_localize(None)

# 1. the re-downloaded raw close must reproduce the closing prices of the input dataset
d = df.set_index("date")
chk = {}
for name, t in STOCKS.items():
    s = close[t].reindex(d.index); chk[name] = float(np.nanmax(np.abs(s - d[name]) / d[name]))
for name, ts in SECTORS.items():
    s = close[ts].mean(axis=1).reindex(d.index); chk[name] = float(np.nanmax(np.abs(s - d[name]) / d[name]))
print("max relative gap between the re-downloaded close and the input dataset:")
print({k: round(v, 5) for k, v in chk.items()})

# 2. replace by adjusted prices and recompute log returns
new = d.copy()
full_idx = adj.index
for name, t in STOCKS.items():
    px = adj[t]; new[name] = px.reindex(d.index)
    new["r_" + name] = np.log(px / px.shift(1)).reindex(d.index)
ok = adj.where(adj > 0)                                   # a few early adjusted prices of one constituent are not positive: excluded
lr = np.log(ok / ok.shift(1))
for name, ts in SECTORS.items():
    new[name] = adj[ts].mean(axis=1).reindex(d.index)       # kept for reference
    new["r_" + name] = lr[ts].mean(axis=1, skipna=True).reindex(d.index)   # equal-weighted average of the constituents' returns
# excess returns of stocks and sectors, monthly units (the indices' excess returns are unchanged)
for c in [c for c in new.columns if c.startswith("excess_") and c not in ("excess_cac40", "excess_nasdaq", "excess_sp500")]:
    base = "r_" + c.split("excess_")[1].replace("bnp", "bnp_paribas") if c != "excess_bnp" else "r_bnp_paribas"
    rf = new["oat3m"] if any(k in c for k in ("lvmh", "bnp", "luxe", "finance", "industrie", "technologie", "sante", "energie", "consommation", "automobile", "telecom")) else new["us3m"]
    if base in new.columns:
        new[c] = new[base] - rf / 1200
print("non-missing returns after the update:", int(new[["r_" + n for n in list(STOCKS) + list(SECTORS)]].notna().all(axis=1).sum()), "of", len(new))
cmp = pd.DataFrame({"close": d[["r_lvmh", "r_bnp_paribas", "r_microsoft", "r_apple", "r_finance", "r_luxe"]].mean() * 100,
                    "adjusted close": new[["r_lvmh", "r_bnp_paribas", "r_microsoft", "r_apple", "r_finance", "r_luxe"]].mean() * 100}).round(3)
print("\nmean monthly return (%), closing price vs adjusted close\n", cmp)
new.reset_index().to_csv(HERE / "data" / "data_monthly.csv", index=False)
