from __future__ import annotations

import numpy as np

from forecaster.baselines.classical import naive, naive2, seasonal_naive, theta
from forecaster.baselines.seasonality import (
    autocorrelation,
    centered_moving_average,
    deseasonalize,
    is_seasonal,
    reseasonalize,
    seasonal_indices,
)
from forecaster.baselines.smoothing import exponential_levels, optimal_alpha


def _seasonal_series(length: int, period: int) -> np.ndarray:
    steps = np.arange(length, dtype=np.float64)
    return 100.0 + 0.5 * steps + 10.0 * np.sin(2.0 * np.pi * steps / period)


def test_naive_repeats_last_value() -> None:
    series = np.array([1.0, 2.0, 7.5])
    assert np.allclose(naive(series, 3, 1), 7.5)


def test_seasonal_naive_repeats_last_period() -> None:
    series = np.arange(10.0)
    forecast = seasonal_naive(series, 5, 4)
    assert np.allclose(forecast, [6.0, 7.0, 8.0, 9.0, 6.0])


def test_autocorrelation_of_constant_series_is_zero() -> None:
    assert autocorrelation(np.ones(10), 1) == 0.0


def test_seasonality_detected_for_seasonal_series() -> None:
    assert is_seasonal(_seasonal_series(120, 12), 12)


def test_seasonality_not_detected_for_white_noise() -> None:
    noise = np.random.default_rng(0).normal(size=200)
    assert not is_seasonal(noise, 12)


def test_centered_moving_average_recovers_linear_trend() -> None:
    series = 10.0 + 0.1 * np.arange(60.0)
    smoothed = centered_moving_average(series, 12)
    assert np.allclose(smoothed[6:-6], series[6:-6], atol=1e-8)


def test_deseasonalize_and_reseasonalize_round_trip() -> None:
    series = _seasonal_series(144, 12)
    indices = seasonal_indices(series, 12)
    adjusted = deseasonalize(series, indices)
    restored = reseasonalize(adjusted, indices, 0)
    assert np.allclose(restored, series, atol=1e-8)


def test_seasonal_indices_are_unit_mean() -> None:
    indices = seasonal_indices(_seasonal_series(144, 12), 12)
    assert abs(float(indices.mean()) - 1.0) < 1e-9


def test_exponential_levels_match_reference_recursion() -> None:
    series = np.cumsum(np.random.default_rng(1).normal(size=50)) + 100.0
    alpha = 0.37
    reference = [series[0]]
    for value in series[1:]:
        reference.append(alpha * value + (1.0 - alpha) * reference[-1])
    assert np.allclose(exponential_levels(series, alpha), reference)


def test_optimal_alpha_stays_inside_bounds() -> None:
    series = np.cumsum(np.random.default_rng(2).normal(size=80)) + 50.0
    alpha = optimal_alpha(series)
    assert 0.01 <= alpha <= 0.99


def test_naive2_removes_seasonality_before_forecasting() -> None:
    series = _seasonal_series(144, 12)
    flat = naive(series, 12, 12)
    adjusted = naive2(series, 12, 12)
    assert float(np.std(adjusted)) > float(np.std(flat))


def test_theta_tracks_a_linear_trend() -> None:
    series = 5.0 + 2.0 * np.arange(60.0)
    forecast = theta(series, 6, 1)
    expected = 5.0 + 2.0 * np.arange(60.0, 66.0)
    assert np.allclose(forecast, expected, rtol=0.05)
