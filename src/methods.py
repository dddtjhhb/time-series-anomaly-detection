"""Transparent rolling z-score calculations."""
import numpy as np
import pandas as pd

def add_rolling_statistics(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    if window < 2:
        raise ValueError("window must be at least 2")
    out = frame.copy()
    history = out["Return"].rolling(window, min_periods=window)
    out["RollingMean"] = history.mean().shift(1)
    out["RollingStd"] = history.std(ddof=1).shift(1)
    out["ZScore"] = (out["Return"] - out["RollingMean"]) / out["RollingStd"]
    out.loc[~np.isfinite(out["ZScore"]), "ZScore"] = np.nan
    return out

def flag_anomalies(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    out = frame.copy()
    out["IsAnomaly"] = out["ZScore"].abs() > threshold
    return out
