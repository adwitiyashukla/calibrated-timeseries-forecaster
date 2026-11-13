from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


def overall_owa(
    smape: np.ndarray, mase: np.ndarray, base_smape: np.ndarray, base_mase: np.ndarray
) -> float:
    return 0.5 * (smape.mean() / base_smape.mean() + mase.mean() / base_mase.mean())


def bootstrap_owa_difference(
    challenger_smape: np.ndarray,
    challenger_mase: np.ndarray,
    champion_smape: np.ndarray,
    champion_mase: np.ndarray,
    base_smape: np.ndarray,
    base_mase: np.ndarray,
    samples: int,
    seed: int,
    series_fraction: float = 1.0,
) -> Interval:
    generator = np.random.default_rng(seed)
    total = challenger_smape.size
    draw = max(1, int(round(total * series_fraction)))
    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selection = generator.integers(0, total, draw)
        differences[index] = overall_owa(
            challenger_smape[selection],
            challenger_mase[selection],
            base_smape[selection],
            base_mase[selection],
        ) - overall_owa(
            champion_smape[selection],
            champion_mase[selection],
            base_smape[selection],
            base_mase[selection],
        )
    estimate = overall_owa(
        challenger_smape, challenger_mase, base_smape, base_mase
    ) - overall_owa(champion_smape, champion_mase, base_smape, base_mase)
    low, high = np.quantile(differences, [0.025, 0.975])
    return Interval(estimate=float(estimate), low=float(low), high=float(high))
