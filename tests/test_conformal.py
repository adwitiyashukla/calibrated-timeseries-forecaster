from __future__ import annotations

import numpy as np

from forecaster.conformal.intervals import calibrate, conformal_quantile, coverage


def test_conformal_quantile_returns_the_maximum_for_small_samples() -> None:
    scores = np.array([[1.0], [2.0], [3.0]])
    assert conformal_quantile(scores, 0.95)[0] == 3.0


def test_conformal_quantile_is_monotone_in_level() -> None:
    scores = np.random.default_rng(0).random((200, 4))
    lower = conformal_quantile(scores, 0.8)
    upper = conformal_quantile(scores, 0.95)
    assert np.all(upper >= lower)


def test_pooled_calibration_reaches_nominal_coverage() -> None:
    generator = np.random.default_rng(4)
    horizon = 3
    series_count = 60
    calibration_scores = np.abs(generator.normal(size=(series_count * 8, horizon)))
    index = np.repeat(np.arange(series_count, dtype=np.int32), 8)
    calibration = calibrate(calibration_scores, index, series_count, (0.8, 0.95), 0.5, 1e-8)

    point = np.zeros((series_count, horizon))
    actual = generator.normal(size=(series_count, horizon))
    scale = np.ones(series_count)
    lower, upper = calibration.intervals(point, scale, np.arange(series_count), adaptive=False)
    observed = coverage(actual, lower, upper)
    assert observed[0] > 0.72
    assert observed[1] > 0.90


def test_adaptive_widths_fall_back_to_pooled_for_unseen_series() -> None:
    generator = np.random.default_rng(5)
    horizon = 2
    scores = np.abs(generator.normal(size=(20, horizon)))
    index = np.repeat(np.arange(5, dtype=np.int32), 4)
    calibration = calibrate(scores, index, 8, (0.9,), 1.0, 1e-8)
    widths = calibration.widths(np.array([7]), adaptive=True)
    assert np.allclose(widths[0, 0], calibration.global_quantiles[0])


def test_wider_level_gives_wider_intervals() -> None:
    generator = np.random.default_rng(6)
    scores = np.abs(generator.normal(size=(400, 2)))
    index = np.repeat(np.arange(100, dtype=np.int32), 4)
    calibration = calibrate(scores, index, 100, (0.8, 0.95), 0.5, 1e-8)
    assert np.all(calibration.global_quantiles[1] >= calibration.global_quantiles[0])
