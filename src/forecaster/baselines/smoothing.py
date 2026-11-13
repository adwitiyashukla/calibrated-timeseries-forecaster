from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter

ALPHA_LOWER = 0.01
ALPHA_UPPER = 0.99


def exponential_levels(series: np.ndarray, alpha: float) -> np.ndarray:
    if series.size == 1:
        return series.astype(np.float64)
    initial = float(series[0])
    tail = lfilter([alpha], [1.0, alpha - 1.0], series[1:], zi=[(1.0 - alpha) * initial])[0]
    return np.concatenate(([initial], tail))


def _sum_squared_error(alpha: float, series: np.ndarray) -> float:
    levels = exponential_levels(series, alpha)
    residuals = series[1:] - levels[:-1]
    return float(np.dot(residuals, residuals))


def optimal_alpha(series: np.ndarray) -> float:
    if series.size < 3:
        return 0.5
    result = minimize_scalar(
        _sum_squared_error,
        bounds=(ALPHA_LOWER, ALPHA_UPPER),
        args=(series,),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


def simple_exponential_smoothing(series: np.ndarray, horizon: int) -> np.ndarray:
    alpha = optimal_alpha(series)
    levels = exponential_levels(series, alpha)
    return np.full(horizon, levels[-1], dtype=np.float64)
