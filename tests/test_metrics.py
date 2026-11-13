from __future__ import annotations

import numpy as np

from forecaster.data.panel import SeriesPanel
from forecaster.metrics.m4 import (
    guard_denominator,
    mase_denominator,
    mase_per_series,
    smape_per_series,
)


def _panel(rows: list[np.ndarray]) -> SeriesPanel:
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum([row.size for row in rows], out=offsets[1:])
    return SeriesPanel(
        frequency="Hourly",
        series_ids=tuple(f"S{index}" for index in range(len(rows))),
        values=np.concatenate(rows).astype(np.float32),
        offsets=offsets,
    )


def test_smape_is_zero_for_perfect_forecast() -> None:
    actual = np.array([[1.0, 2.0, 3.0]])
    assert smape_per_series(actual, actual.copy())[0] == 0.0


def test_smape_matches_manual_calculation() -> None:
    actual = np.array([[100.0, 200.0]])
    forecast = np.array([[110.0, 180.0]])
    expected = 200.0 * np.mean([10.0 / 210.0, 20.0 / 380.0])
    assert abs(smape_per_series(actual, forecast)[0] - expected) < 1e-12


def test_smape_ignores_zero_over_zero() -> None:
    actual = np.array([[0.0, 4.0]])
    forecast = np.array([[0.0, 4.0]])
    assert smape_per_series(actual, forecast)[0] == 0.0


def test_mase_denominator_uses_seasonal_difference() -> None:
    series = np.array([1.0, 3.0, 2.0, 6.0, 4.0, 9.0], dtype=np.float32)
    panel = _panel([series])
    expected = np.mean(np.abs(series[2:] - series[:-2]))
    assert abs(mase_denominator(panel, 2)[0] - expected) < 1e-6


def test_mase_is_one_when_error_matches_denominator() -> None:
    actual = np.array([[10.0, 10.0]])
    forecast = np.array([[8.0, 12.0]])
    denominator = np.array([2.0])
    assert abs(mase_per_series(actual, forecast, denominator)[0] - 1.0) < 1e-12


def test_guard_replaces_degenerate_denominator() -> None:
    panel = _panel([np.array([5.0, 5.0, 5.0], dtype=np.float32)])
    guarded, count = guard_denominator(np.array([0.0]), panel)
    assert count == 1
    assert guarded[0] == 5.0
