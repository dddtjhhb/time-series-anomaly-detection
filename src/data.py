"""Download, validate, and reshape daily price data."""
from pathlib import Path
from urllib.request import urlopen
import pandas as pd

def clean_prices(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    missing = {"Date", "Close"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"])
    out = out[out["Close"] > 0].drop_duplicates("Date", keep="last").sort_values("Date")
    out = out[["Date", "Close"]].reset_index(drop=True)
    out.insert(1, "Ticker", ticker.upper())
    out["Return"] = out["Close"].pct_change()
    return out

SNAPSHOT_URL = "https://raw.githubusercontent.com/hhhx-lab/-ETF-/main/data/raw/etf_prices_raw.csv"

def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Extract adjusted close from a public Yahoo Finance-derived snapshot."""
    url = SNAPSHOT_URL
    with urlopen(url, timeout=30) as response:
        wide = pd.read_csv(response, header=[0, 1], index_col=0)
    if ("Adj Close", ticker.upper()) not in wide.columns:
        raise ValueError(f"Ticker {ticker} is absent from data snapshot")
    frame = wide[("Adj Close", ticker.upper())].rename("Close").reset_index()
    frame.columns = ["Date", "Close"]
    frame = frame[(frame["Date"] >= start) & (frame["Date"] <= end)]
    return clean_prices(frame, ticker)

def load_or_download(ticker: str, start: str, end: str, raw_dir: Path, refresh=False):
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{ticker.upper()}_{start}_{end}.csv"
    if path.exists() and not refresh:
        return clean_prices(pd.read_csv(path), ticker)
    out = download_prices(ticker, start, end)
    out[["Date", "Close"]].to_csv(path, index=False)
    return out

def combine_assets(frames):
    if not frames:
        raise ValueError("At least one data frame is required")
    return pd.concat(frames, ignore_index=True).sort_values(["Ticker", "Date"])
