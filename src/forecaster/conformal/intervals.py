from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def conformal_quantile(scores: np.ndarray, level: float) -> np.ndarray:
    count = scores.shape[0]
    if count == 0:
        return np.zeros(scores.shape[1], dtype=np.float64)
    rank = int(np.ceil((count + 1) * level))
    if rank > count:
        return scores.max(axis=0).astype(np.float64)
    position = (rank - 1) / max(count - 1, 1)
    return np.quantile(scores, position, axis=0, method="higher").astype(np.float64)


@dataclass(frozen=True)
class ConformalCalibration:
    levels: tuple[float, ...]
    global_quantiles: np.ndarray
    series_quantiles: np.ndarray
    series_observed: np.ndarray
    shrinkage: float

    def widths(self, series_index: np.ndarray, adaptive: bool) -> np.ndarray:
        pooled = self.global_quantiles[:, None, :]
        if not adaptive:
            return np.broadcast_to(
                pooled, (len(self.levels), series_index.size, self.global_quantiles.shape[1])
            ).copy()
        local = self.series_quantiles[:, series_index, :]
        observed = self.series_observed[series_index][None, :, None]
        blended = (1.0 - self.shrinkage) * pooled + self.shrinkage * local
        return np.where(observed, blended, np.broadcast_to(pooled, blended.shape))

    def intervals(
        self, point: np.ndarray, scale: np.ndarray, series_index: np.ndarray, adaptive: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        widths = self.widths(series_index, adaptive) * scale[None, :, None]
        return point[None, :, :] - widths, point[None, :, :] + widths

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "conformal.npz",
            global_quantiles=self.global_quantiles,
            series_quantiles=self.series_quantiles,
            series_observed=self.series_observed,
        )
        (directory / "conformal.json").write_text(
            json.dumps({"levels": list(self.levels), "shrinkage": self.shrinkage})
        )

    @classmethod
    def load(cls, directory: Path) -> ConformalCalibration:
        payload = np.load(directory / "conformal.npz")
        meta = json.loads((directory / "conformal.json").read_text())
        return cls(
            levels=tuple(meta["levels"]),
            global_quantiles=payload["global_quantiles"],
            series_quantiles=payload["series_quantiles"],
            series_observed=payload["series_observed"],
            shrinkage=float(meta["shrinkage"]),
        )


def calibrate(
    scores: np.ndarray,
    series_index: np.ndarray,
    series_count: int,
    levels: tuple[float, ...],
    shrinkage: float,
    floor: float,
) -> ConformalCalibration:
    horizon = scores.shape[1]
    global_quantiles = np.empty((len(levels), horizon), dtype=np.float32)
    series_quantiles = np.zeros((len(levels), series_count, horizon), dtype=np.float32)
    observed = np.zeros(series_count, dtype=bool)

    order = np.argsort(series_index, kind="stable")
    sorted_index = series_index[order]
    sorted_scores = scores[order]
    boundaries = np.flatnonzero(np.diff(sorted_index)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [sorted_index.size]))

    for position, level in enumerate(levels):
        global_quantiles[position] = np.maximum(
            conformal_quantile(sorted_scores, level), floor
        ).astype(np.float32)

    for start, end in zip(starts, ends, strict=True):
        identifier = int(sorted_index[start])
        block = sorted_scores[start:end]
        observed[identifier] = True
        for position, level in enumerate(levels):
            series_quantiles[position, identifier] = np.maximum(
                conformal_quantile(block, level), floor
            ).astype(np.float32)

    return ConformalCalibration(
        levels=levels,
        global_quantiles=global_quantiles,
        series_quantiles=series_quantiles,
        series_observed=observed,
        shrinkage=shrinkage,
    )


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    inside = (actual[None, :, :] >= lower) & (actual[None, :, :] <= upper)
    return inside.mean(axis=(1, 2))


def coverage_by_step(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    inside = (actual[None, :, :] >= lower) & (actual[None, :, :] <= upper)
    return inside.mean(axis=1)


def scaled_width(lower: np.ndarray, upper: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return ((upper - lower) / denominator[None, :, None]).mean(axis=(1, 2))


def pinball_loss(
    actual: np.ndarray, prediction: np.ndarray, quantile: float, denominator: np.ndarray
) -> float:
    difference = actual - prediction
    loss = np.maximum(quantile * difference, (quantile - 1.0) * difference)
    return float((loss / denominator[:, None]).mean())
