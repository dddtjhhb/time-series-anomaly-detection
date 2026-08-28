"""Simple online CUSUM monitoring for upward changes in a signal mean."""

from __future__ import annotations

import numpy as np
import pandas as pd


def upper_cusum(
    values: pd.Series,
    warmup: int = 100,
    drift: float = 0.5,
    threshold: float = 20.0,
) -> pd.DataFrame:
    """Run a one-sided online CUSUM using a fixed warm-up baseline.

    ``values`` should be a nonnegative volatility proxy such as absolute return.
    The first ``warmup`` observations estimate the baseline mean and standard
    deviation. Each later observation is processed exactly once, in time order.
    No future observation is used to calculate an earlier score.
    """
    if warmup < 20 or warmup >= len(values):
        raise ValueError("warmup must be at least 20 and shorter than the series")
    if drift < 0 or threshold <= 0:
        raise ValueError("drift must be nonnegative and threshold must be positive")

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError("values must not contain missing or nonnumeric entries")

    baseline = numeric.iloc[:warmup]
    baseline_mean = float(baseline.mean())
    baseline_std = float(baseline.std(ddof=1))
    if not np.isfinite(baseline_std) or baseline_std <= 0:
        raise ValueError("warm-up period must have positive standard deviation")

    scores = np.zeros(len(numeric), dtype=float)
    alarms = np.zeros(len(numeric), dtype=bool)
    running_score = 0.0

    for i in range(warmup, len(numeric)):
        standardized = (float(numeric.iloc[i]) - baseline_mean) / baseline_std
        running_score = max(0.0, running_score + standardized - drift)
        scores[i] = running_score
        alarms[i] = running_score > threshold

    return pd.DataFrame(
        {
            "CUSUMScore": scores,
            "IsCUSUMAlarm": alarms,
            "BaselineMean": baseline_mean,
            "BaselineStd": baseline_std,
        },
        index=values.index,
    )


def first_alarm_summary(alarms: pd.Series, true_change_index: int) -> dict[str, int | bool | None]:
    """Summarize the first online alarm relative to one known change point."""
    alarm_positions = np.flatnonzero(alarms.to_numpy(dtype=bool))
    first_alarm = int(alarm_positions[0]) if len(alarm_positions) else None
    false_alarm = first_alarm is not None and first_alarm < true_change_index
    detected = first_alarm is not None and first_alarm >= true_change_index
    delay = first_alarm - true_change_index if detected else None
    return {
        "TrueChangeIndex": true_change_index,
        "FirstAlarmIndex": first_alarm,
        "Detected": detected,
        "FalseAlarmBeforeChange": false_alarm,
        "DetectionDelay": delay,
    }
