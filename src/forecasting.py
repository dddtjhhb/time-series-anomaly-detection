"""Past-only features and simple linear models for volatility forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_volatility_forecast_features(
    frame: pd.DataFrame,
    return_column: str = "Return",
    volatility_window: int = 20,
    cusum_window: int = 60,
    horizon: int = 5,
    drift: float = 0.5,
) -> pd.DataFrame:
    """Add past volatility, adaptive CUSUM, and future realized volatility.

    Features at time t use observations no later than t. The target at time t
    uses returns t+1 through t+horizon and is therefore used only for evaluation,
    never as an input available at prediction time.
    """
    if volatility_window < 2 or cusum_window < 20 or horizon < 1:
        raise ValueError("choose valid windows and a positive forecast horizon")
    if drift < 0:
        raise ValueError("drift must be nonnegative")

    result = frame.copy()
    returns = pd.to_numeric(result[return_column], errors="coerce")
    annualization = np.sqrt(252.0)

    result["PastVolatility20"] = (
        returns.pow(2).rolling(volatility_window).mean().pow(0.5) * annualization
    )

    absolute_returns = returns.abs()
    past_center = absolute_returns.rolling(cusum_window).mean().shift(1)
    past_scale = absolute_returns.rolling(cusum_window).std(ddof=1).shift(1)
    standardized = (absolute_returns - past_center) / past_scale
    standardized = standardized.where(np.isfinite(standardized))

    scores = np.full(len(result), np.nan, dtype=float)
    running_score = 0.0
    for i, value in enumerate(standardized):
        if pd.isna(value):
            continue
        running_score = max(0.0, running_score + float(value) - drift)
        scores[i] = running_score
    result["AdaptiveCUSUMScore"] = scores

    future_returns = pd.concat(
        [returns.shift(-step) for step in range(1, horizon + 1)], axis=1
    )
    result["FutureVolatility5"] = (
        future_returns.pow(2).mean(axis=1).pow(0.5) * annualization
    )
    result.loc[future_returns.isna().any(axis=1), "FutureVolatility5"] = np.nan
    return result


def fit_linear_regression(
    frame: pd.DataFrame, feature_columns: list[str], target_column: str
) -> np.ndarray:
    """Fit ordinary least squares with an intercept and return coefficients."""
    clean = frame.dropna(subset=feature_columns + [target_column])
    if len(clean) <= len(feature_columns):
        raise ValueError("not enough complete observations to fit the model")
    design = np.column_stack(
        [np.ones(len(clean)), clean[feature_columns].to_numpy(dtype=float)]
    )
    target = clean[target_column].to_numpy(dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def predict_linear(frame: pd.DataFrame, feature_columns: list[str], coefficients: np.ndarray) -> pd.Series:
    """Generate nonnegative predictions from fitted linear coefficients."""
    if len(coefficients) != len(feature_columns) + 1:
        raise ValueError("coefficient count does not match feature columns")
    design = np.column_stack(
        [np.ones(len(frame)), frame[feature_columns].to_numpy(dtype=float)]
    )
    predictions = design @ coefficients
    return pd.Series(np.maximum(predictions, 0.0), index=frame.index)


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Calculate mean absolute error and root mean squared error."""
    errors = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
    }
