from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecaster.data.panel import SeriesPanel


def smape_per_series(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    numerator = np.abs(actual - forecast)
    denominator = np.abs(actual) + np.abs(forecast)
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return 200.0 * ratio.mean(axis=1)


def mase_denominator(panel: SeriesPanel, period: int) -> np.ndarray:
    values = np.empty(panel.size, dtype=np.float64)
    for index in range(panel.size):
        series = panel.series(index).astype(np.float64)
        if series.size <= period:
            values[index] = np.mean(np.abs(np.diff(series))) if series.size > 1 else 0.0
        else:
            values[index] = np.mean(np.abs(series[period:] - series[:-period]))
    return values


def guard_denominator(denominator: np.ndarray, panel: SeriesPanel) -> tuple[np.ndarray, int]:
    guarded = denominator.copy()
    degenerate = guarded <= 0.0
    count = int(degenerate.sum())
    for index in np.flatnonzero(degenerate):
        series = panel.series(int(index)).astype(np.float64)
        fallback = float(np.mean(np.abs(series)))
        guarded[index] = fallback if fallback > 0.0 else 1.0
    return guarded, count


def mase_per_series(actual: np.ndarray, forecast: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.abs(actual - forecast).mean(axis=1) / denominator


@dataclass(frozen=True)
class FrequencyScore:
    frequency: str
    series: int
    smape: float
    mase: float

    def owa(self, base: FrequencyScore) -> float:
        return 0.5 * (self.smape / base.smape + self.mase / base.mase)


@dataclass(frozen=True)
class ModelScore:
    model: str
    by_frequency: dict[str, FrequencyScore]
    smape: float
    mase: float
    series: int

    def owa(self, base: ModelScore) -> float:
        return 0.5 * (self.smape / base.smape + self.mase / base.mase)


def score_frequency(
    frequency: str, actual: np.ndarray, forecast: np.ndarray, denominator: np.ndarray
) -> FrequencyScore:
    smape = smape_per_series(actual, forecast)
    mase = mase_per_series(actual, forecast, denominator)
    return FrequencyScore(
        frequency=frequency,
        series=int(actual.shape[0]),
        smape=float(smape.mean()),
        mase=float(mase.mean()),
    )


def combine(model: str, scores: dict[str, FrequencyScore]) -> ModelScore:
    total = sum(score.series for score in scores.values())
    smape = sum(score.smape * score.series for score in scores.values()) / total
    mase = sum(score.mase * score.series for score in scores.values()) / total
    return ModelScore(model=model, by_frequency=scores, smape=smape, mase=mase, series=total)
