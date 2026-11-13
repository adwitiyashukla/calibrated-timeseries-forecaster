from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecaster.config import Config, FrequencyConfig
from forecaster.data.panel import SeriesPanel

STATISTIC_NAMES: tuple[str, ...] = (
    "scaled_std",
    "scaled_min",
    "scaled_max",
    "scaled_slope",
    "recent_period_ratio",
    "autocorrelation_1",
    "autocorrelation_period",
    "trend_strength",
)


@dataclass(frozen=True)
class FeatureSpec:
    lookback: int
    horizon: int
    period: int
    scale_floor: float
    include_statistics: bool

    @classmethod
    def from_config(cls, config: Config, frequency: FrequencyConfig) -> FeatureSpec:
        return cls(
            lookback=frequency.lookback,
            horizon=frequency.horizon,
            period=frequency.period,
            scale_floor=config.features.scale_floor,
            include_statistics=config.features.include_window_statistics,
        )

    def names(self) -> list[str]:
        names = [f"lag_{offset}" for offset in range(1, self.lookback + 1)]
        names.append("log_scale")
        names.append("context_ratio")
        if self.include_statistics:
            names.extend(STATISTIC_NAMES)
        if self.period > 1:
            names.append("origin_phase")
        return names

    def width(self) -> int:
        return len(self.names())

    def scale(self, windows: np.ndarray) -> np.ndarray:
        magnitude = np.abs(windows).mean(axis=1)
        return np.maximum(magnitude, self.scale_floor).astype(np.float32)

    def scaled_windows(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = self.scale(windows)
        return windows / scale[:, None], scale

    def transform(
        self, windows: np.ndarray, context_lengths: np.ndarray, origins: np.ndarray
    ) -> np.ndarray:
        scaled, scale = self.scaled_windows(windows)
        columns = [scaled[:, ::-1]]
        columns.append(np.log1p(scale)[:, None])
        columns.append((context_lengths / self.lookback).astype(np.float32)[:, None])
        if self.include_statistics:
            columns.append(self._statistics(scaled))
        if self.period > 1:
            columns.append((origins % self.period).astype(np.float32)[:, None])
        return np.concatenate(columns, axis=1).astype(np.float32)

    def _statistics(self, scaled: np.ndarray) -> np.ndarray:
        length = scaled.shape[1]
        positions = np.arange(length, dtype=np.float32)
        centered_positions = positions - positions.mean()
        position_energy = float(np.dot(centered_positions, centered_positions))
        deviation = scaled - scaled.mean(axis=1, keepdims=True)
        standard = scaled.std(axis=1)
        slope = deviation @ centered_positions / max(position_energy, 1e-8)
        window_mean = scaled.mean(axis=1)
        recent = scaled[:, -min(self.period, length) :].mean(axis=1)
        safe_mean = np.where(np.abs(window_mean) > 1e-8, window_mean, 1.0)
        energy = np.maximum((deviation * deviation).sum(axis=1), 1e-8)
        first = (deviation[:, 1:] * deviation[:, :-1]).sum(axis=1) / energy
        if self.period > 1 and length > self.period:
            seasonal = (
                deviation[:, self.period :] * deviation[:, : -self.period]
            ).sum(axis=1) / energy
        else:
            seasonal = np.zeros(scaled.shape[0], dtype=np.float32)
        strength = np.abs(slope) * length / np.maximum(standard, 1e-8)
        return np.stack(
            [
                standard,
                scaled.min(axis=1),
                scaled.max(axis=1),
                slope,
                recent / safe_mean,
                first,
                seasonal,
                strength,
            ],
            axis=1,
        ).astype(np.float32)


def assemble_windows(
    panel: SeriesPanel, series_index: np.ndarray, origins: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    count = series_index.size
    windows = np.empty((count, lookback), dtype=np.float32)
    context_lengths = np.empty(count, dtype=np.float32)
    values = panel.values
    offsets = panel.offsets
    for row in range(count):
        start = offsets[series_index[row]]
        origin = int(origins[row])
        available = min(origin, lookback)
        segment = values[start + origin - available : start + origin]
        if available < lookback:
            windows[row, : lookback - available] = segment[0]
            windows[row, lookback - available :] = segment
        else:
            windows[row] = segment
        context_lengths[row] = available
    return windows, context_lengths


def assemble_targets(
    panel: SeriesPanel, series_index: np.ndarray, origins: np.ndarray, horizon: int
) -> np.ndarray:
    count = series_index.size
    targets = np.empty((count, horizon), dtype=np.float32)
    values = panel.values
    offsets = panel.offsets
    for row in range(count):
        start = offsets[series_index[row]] + int(origins[row])
        targets[row] = values[start : start + horizon]
    return targets
