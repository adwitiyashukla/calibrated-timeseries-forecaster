from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecaster.config import Config, FrequencyConfig
from forecaster.data.panel import SeriesPanel


@dataclass(frozen=True)
class OriginSet:
    series_index: np.ndarray
    origins: np.ndarray

    @property
    def size(self) -> int:
        return int(self.series_index.size)


@dataclass(frozen=True)
class WindowPlan:
    train: OriginSet
    validation: OriginSet
    calibration: OriginSet
    test_origins: np.ndarray


def minimum_context(frequency: FrequencyConfig) -> int:
    return max(2, frequency.period)


def _evenly_spaced(low: int, high: int, count: int) -> np.ndarray:
    span = high - low + 1
    if count >= span:
        return np.arange(low, high + 1, dtype=np.int64)
    return np.unique(np.round(np.linspace(low, high, count)).astype(np.int64))


def build_window_plan(
    config: Config, frequency: FrequencyConfig, panel: SeriesPanel
) -> WindowPlan:
    lengths = panel.lengths()
    horizon = frequency.horizon
    floor = minimum_context(frequency)
    cap = config.window_cap(frequency.name)
    calibration_count = config.conformal.calibration_origins

    generator = np.random.default_rng(config.run.seed)
    held_out = generator.random(panel.size) < config.run.holdout_series_fraction

    train_series: list[np.ndarray] = []
    train_origins: list[np.ndarray] = []
    validation_series: list[np.ndarray] = []
    validation_origins: list[np.ndarray] = []
    calibration_series: list[np.ndarray] = []
    calibration_origins: list[np.ndarray] = []

    for index in range(panel.size):
        last_origin = int(lengths[index]) - horizon
        if last_origin < floor:
            continue
        calibration_low = max(floor, last_origin - calibration_count + 1)
        calibration = np.arange(calibration_low, last_origin + 1, dtype=np.int64)
        calibration_series.append(np.full(calibration.size, index, dtype=np.int32))
        calibration_origins.append(calibration)
        upper = calibration_low - 1
        if upper < floor:
            continue
        selected = _evenly_spaced(floor, upper, cap)
        target_series = validation_series if held_out[index] else train_series
        target_origins = validation_origins if held_out[index] else train_origins
        target_series.append(np.full(selected.size, index, dtype=np.int32))
        target_origins.append(selected)

    def pack(series: list[np.ndarray], origins: list[np.ndarray]) -> OriginSet:
        if not series:
            return OriginSet(np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64))
        return OriginSet(np.concatenate(series), np.concatenate(origins))

    return WindowPlan(
        train=pack(train_series, train_origins),
        validation=pack(validation_series, validation_origins),
        calibration=pack(calibration_series, calibration_origins),
        test_origins=lengths.astype(np.int64),
    )
