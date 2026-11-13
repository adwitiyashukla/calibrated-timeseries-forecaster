from __future__ import annotations

import numpy as np

from forecaster.baselines.classical import METHODS, forecast_panel
from forecaster.config import Config
from forecaster.data.panel import load_panel
from forecaster.evaluation.paths import Artifacts


def run(config: Config) -> dict[str, object]:
    artifacts = Artifacts(config)
    summary: dict[str, object] = {}
    for frequency in config.run.frequencies:
        settings = config.frequency[frequency]
        panel = load_panel(config, frequency)
        names = sorted(METHODS)
        stacked = np.stack(
            [forecast_panel(panel, settings.horizon, settings.period, name) for name in names]
        )
        destination = artifacts.ensure(artifacts.baselines(frequency))
        np.savez_compressed(destination, methods=np.asarray(names), values=stacked)
        summary[frequency] = {"series": panel.size, "methods": names}
    return summary
