"""Builds macro.db (SQLite) from the data cached by ../macro_econometrie. Run once: python build_db.py"""
import sqlite3, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "macro_econometrie"
sys.path.insert(0, str(ROOT / "src"))
from data_utils import fred, oat_bund_spread   # noqa: E402

con = sqlite3.connect(Path(__file__).with_name("macro.db"))
me = lambda s: s.set_axis(s.index.to_period("M").to_timestamp("M"))

monthly = pd.concat({n: me(fred(i)) for n, i in [("cpi", "CPIAUCSL"), ("unrate", "UNRATE"), ("fedfunds", "FEDFUNDS"),
                                                  ("indpro", "INDPRO"), ("recession", "USREC")]}, axis=1, sort=True).loc["1990":].dropna()
monthly.index.name = "month"; monthly.reset_index().assign(month=lambda d: d.month.dt.strftime("%Y-%m-%d")).to_sql("us_monthly", con, if_exists="replace", index=False)

y = pd.concat({m: fred(k) for k, m in {"DGS3MO": 3, "DGS2": 24, "DGS10": 120}.items()}, axis=1, sort=True).loc["1994":].dropna()
y.columns = ["y_3m", "y_2y", "y_10y"]; y.index.name = "date"
y.reset_index().assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d")).to_sql("us_yields_daily", con, if_exists="replace", index=False)

s = oat_bund_spread().rename(columns={"FR10": "oat10", "DE10": "bund10", "spread": "spread_bp"}); s.index.name = "date"   # one recording error dropped, see data_utils
s.reset_index().assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d")).to_sql("fr_de_daily", con, if_exists="replace", index=False)

ev = pd.DataFrame({"date": ["2017-04-24", "2024-06-10", "2024-07-01", "2024-12-04", "2025-08-26", "2025-09-08"],
                   "label": ["1st round presidential", "Dissolution announced", "1st round legislative", "Barnier censure vote", "Confidence vote announced", "Bayrou confidence vote"]})
ev.to_sql("political_events", con, if_exists="replace", index=False)
for t in ("us_monthly", "us_yields_daily", "fr_de_daily", "political_events"):
    print(t, con.execute(f"select count(*) from {t}").fetchone()[0], "rows")
con.close()
