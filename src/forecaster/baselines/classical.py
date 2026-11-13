from __future__ import annotations

from collections.abc import Callable

import numpy as np

from forecaster.baselines.seasonality import deseasonalize, reseasonalize, seasonal_indices
from forecaster.baselines.smoothing import simple_exponential_smoothing
from forecaster.data.panel import SeriesPanel

Method = Callable[[np.ndarray, int, int], np.ndarray]


def naive(history: np.ndarray, horizon: int, period: int) -> np.ndarray:
    return np.full(horizon, float(history[-1]), dtype=np.float64)


def seasonal_naive(history: np.ndarray, horizon: int, period: int) -> np.ndarray:
    if period <= 1 or history.size < period:
        return naive(history, horizon, period)
    recent = history[-period:].astype(np.float64)
    repeats = int(np.ceil(horizon / period))
    return np.tile(recent, repeats)[:horizon]


def naive2(history: np.ndarray, horizon: int, period: int) -> np.ndarray:
    indices = seasonal_indices(history, period)
    adjusted = deseasonalize(history, indices)
    flat = np.full(horizon, float(adjusted[-1]), dtype=np.float64)
    return reseasonalize(flat, indices, history.size)


def theta(history: np.ndarray, horizon: int, period: int) -> np.ndarray:
    indices = seasonal_indices(history, period)
    adjusted = deseasonalize(history, indices)
    size = adjusted.size
    if size < 3:
        return reseasonalize(np.full(horizon, float(adjusted[-1])), indices, size)
    positions = np.arange(1, size + 1, dtype=np.float64)
    coefficients = np.polyfit(positions, adjusted, 1)
    trend_line = np.polyval(coefficients, positions)
    second_line = 2.0 * adjusted - trend_line
    smoothed = simple_exponential_smoothing(second_line, horizon)
    future = np.arange(size + 1, size + horizon + 1, dtype=np.float64)
    extrapolated = np.polyval(coefficients, future)
    combined = 0.5 * smoothed + 0.5 * extrapolated
    return reseasonalize(combined, indices, size)


METHODS: dict[str, Method] = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "naive2": naive2,
    "theta": theta,
}


def forecast_panel(panel: SeriesPanel, horizon: int, period: int, method: str) -> np.ndarray:
    function = METHODS[method]
    output = np.empty((panel.size, horizon), dtype=np.float32)
    for index in range(panel.size):
        history = panel.series(index).astype(np.float64)
        output[index] = function(history, horizon, period)
    return output
