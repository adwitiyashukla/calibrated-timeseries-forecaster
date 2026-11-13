from __future__ import annotations

import json

import numpy as np

from forecaster.config import Config
from forecaster.conformal.intervals import calibrate
from forecaster.data.panel import load_panel
from forecaster.evaluation.paths import Artifacts
from forecaster.features.spec import FeatureSpec
from forecaster.models.lgbm import LgbmForecaster
from forecaster.models.matrices import build_batch
from forecaster.models.nhits import NhitsForecaster
from forecaster.models.windows import build_window_plan


def run(config: Config) -> dict[str, object]:
    artifacts = Artifacts(config)
    summary: dict[str, object] = {}
    for frequency in config.run.frequencies:
        settings = config.frequency[frequency]
        panel = load_panel(config, frequency)
        spec = FeatureSpec.from_config(config, settings)
        plan = build_window_plan(config, settings, panel)
        batch = build_batch(panel, spec, plan.calibration, True)

        lgbm = LgbmForecaster.load(artifacts.model_directory(frequency, "lgbm"))
        nhits = NhitsForecaster.load(config, artifacts.model_directory(frequency, "nhits"))
        weight = json.loads(artifacts.blend(frequency).read_text())["weight"]
        prediction = weight * lgbm.predict(batch) + (1.0 - weight) * nhits.predict(batch)

        targets = batch.scaled_targets * batch.scale[:, None]
        scores = np.abs(prediction - targets) / batch.scale[:, None]
        calibration = calibrate(
            scores.astype(np.float64),
            batch.series_index,
            panel.size,
            config.conformal.levels,
            config.conformal.shrinkage_weight,
            config.conformal.score_floor,
        )
        calibration.save(artifacts.conformal(frequency))
        summary[frequency] = {
            "calibration_windows": int(batch.series_index.size),
            "series_with_calibration": int(calibration.series_observed.sum()),
            "global_quantiles": [
                [round(float(value), 4) for value in row] for row in calibration.global_quantiles
            ],
        }
    return summary
