"""Small data-access helpers shared by the notebooks.

Every download is cached in ../data so that the notebooks re-run offline and give
the same numbers. Delete a csv in data/ to force a fresh download.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)
HEADERS = None  # FRED drops connections that send a browser User-Agent


def _get(url: str, params: dict | None = None, retries: int = 4, timeout: int = 60) -> requests.Response:
    last = None
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:  # network flakiness is common on FRED
            last = exc
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"download failed: {url}") from last


def fred(series_id: str, start: str = "1950-01-01") -> pd.Series:
    """FRED series (public csv endpoint, no API key). Cached."""
    path = DATA_DIR / f"fred_{series_id}.csv"
    if not path.exists():
        r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": series_id})
        path.write_bytes(r.content)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    s.name = series_id
    return s.loc[start:]


def dbnomics(provider: str, dataset: str, series: str, name: str | None = None) -> pd.Series:
    """Series from DBnomics (mirrors ECB, Banque de France, Bundesbank, Eurostat...). Cached."""
    key = f"dbn_{provider}_{dataset}_{series}".replace("/", "_")
    path = DATA_DIR / f"{key}.csv"
    if not path.exists():
        url = f"https://api.db.nomics.world/v22/series/{provider}/{dataset}/{series}"
        doc = _get(url, {"observations": 1}).json()["series"]["docs"][0]
        pd.DataFrame({"period": doc["period"], "value": doc["value"]}).to_csv(path, index=False)
    df = pd.read_csv(path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    idx = pd.to_datetime(df["period"])
    s = pd.Series(df["value"].values, index=idx, name=name or series).dropna()
    return s


def ecb(flow: str, key: str, name: str | None = None) -> pd.Series:
    """Series from the ECB Data Portal (csvdata). Cached."""
    path = DATA_DIR / f"ecb_{flow}_{key}.csv"
    if not path.exists():
        r = _get(f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}", {"format": "csvdata"})
        path.write_bytes(r.content)
    df = pd.read_csv(path)
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values,
                  index=pd.to_datetime(df["TIME_PERIOD"]), name=name or key).dropna()
    return s


def download(url: str, filename: str) -> Path:
    """Plain file download, cached."""
    path = DATA_DIR / filename
    if not path.exists():
        path.write_bytes(_get(url).content)
    return path


def to_month_end(s: pd.Series, how: str = "last") -> pd.Series:
    """Resample to month-end (last obs or mean)."""
    g = s.resample("ME")
    out = g.last() if how == "last" else g.mean()
    return out.dropna()


def oat_bund_spread(max_reversed_move_bp: float = 20.0) -> pd.DataFrame:
    """Daily French and German 10-year yields (%) and their spread (basis points). Cached.

    French yield: Banque de France TEC10, interpolated from prices taken at 11 a.m. German yield: Bundesbank
    Svensson zero-coupon curve. A day on which the spread moves by more than `max_reversed_move_bp` and at
    least three quarters of the move is undone the next day is treated as a recording error and dropped.
    The dropped dates are listed in `.attrs["dropped"]`. In the cached data this removes one day,
    28 June 2022: the series gives 1.83% where the Banque de France table of indicative OAT rates gives 2.18%.
    """
    fr = dbnomics("BDF", "FM", "D.FR.EUR.FR2.BB.FRMOYTEC10.HSTA", "FR10")
    de = dbnomics("BUBA", "BBSIS", "D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A", "DE10")
    d = pd.concat([fr, de], axis=1, sort=True).dropna()
    d["spread"] = (d.FR10 - d.DE10) * 100
    move = d["spread"].diff()
    back = move.shift(-1)
    bad = (move.abs() > max_reversed_move_bp) & ((move + back).abs() < 0.25 * move.abs())
    out = d[~bad].copy()
    out.attrs["dropped"] = [t.date().isoformat() for t in d.index[bad]]
    return out
