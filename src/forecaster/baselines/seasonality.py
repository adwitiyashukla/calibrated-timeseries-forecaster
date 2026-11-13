from __future__ import annotations

import numpy as np

SEASONALITY_CRITICAL_VALUE = 1.645


def autocorrelation(series: np.ndarray, lag: int) -> float:
    if lag >= series.size:
        return 0.0
    centered = series - series.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return 0.0
    numerator = float(np.dot(centered[lag:], centered[:-lag]))
    return numerator / denominator


def is_seasonal(series: np.ndarray, period: int) -> bool:
    if period <= 1 or series.size < 3 * period:
        return False
    accumulated = autocorrelation(series, 1)
    for lag in range(2, period):
        accumulated += autocorrelation(series, lag) ** 2
    limit = SEASONALITY_CRITICAL_VALUE * np.sqrt((1.0 + 2.0 * accumulated) / series.size)
    return bool(abs(autocorrelation(series, period)) > limit)


def centered_moving_average(series: np.ndarray, window: int) -> np.ndarray:
    size = series.size
    result = np.full(size, np.nan, dtype=np.float64)
    if window <= 1 or size < window + 1:
        return result
    if window % 2 == 0:
        half = window // 2
        weights = np.ones(window + 1, dtype=np.float64)
        weights[0] = 0.5
        weights[-1] = 0.5
        weights /= window
        valid = np.convolve(series, weights[::-1], mode="valid")
        result[half : size - half] = valid
    else:
        half = window // 2
        weights = np.full(window, 1.0 / window, dtype=np.float64)
        valid = np.convolve(series, weights[::-1], mode="valid")
        result[half : size - half] = valid
    return result


def seasonal_indices(series: np.ndarray, period: int) -> np.ndarray:
    if period <= 1 or not is_seasonal(series, period):
        return np.ones(period, dtype=np.float64)
    trend = centered_moving_average(series.astype(np.float64), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(trend > 0.0, series / trend, np.nan)
    positions = np.arange(series.size) % period
    indices = np.ones(period, dtype=np.float64)
    for position in range(period):
        selected = ratio[positions == position]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            indices[position] = float(selected.mean())
    total = float(indices.mean())
    if total <= 0.0 or not np.isfinite(total):
        return np.ones(period, dtype=np.float64)
    indices = indices / total
    indices[indices <= 0.0] = 1.0
    return indices


def deseasonalize(series: np.ndarray, indices: np.ndarray) -> np.ndarray:
    period = indices.size
    if period <= 1:
        return series.astype(np.float64)
    positions = np.arange(series.size) % period
    return series.astype(np.float64) / indices[positions]


def reseasonalize(forecast: np.ndarray, indices: np.ndarray, offset: int) -> np.ndarray:
    period = indices.size
    if period <= 1:
        return forecast
    positions = (np.arange(forecast.size) + offset) % period
    return forecast * indices[positions]
