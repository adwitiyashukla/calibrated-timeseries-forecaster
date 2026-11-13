from __future__ import annotations

import json

import numpy as np

from forecaster.baselines.classical import naive2
from forecaster.config import Config
from forecaster.data.panel import SeriesPanel, load_panel
from forecaster.evaluation.paths import Artifacts
from forecaster.features.spec import FeatureSpec
from forecaster.metrics.m4 import mase_per_series, smape_per_series
from forecaster.models.lgbm import LgbmForecaster
from forecaster.models.matrices import build_batch
from forecaster.models.nhits import NhitsForecaster
from forecaster.models.windows import build_window_plan
from forecaster.stages.backtest import _history_denominator


def _reference(
    panel: SeriesPanel,
    series_index: np.ndarray,
    origins: np.ndarray,
    horizon: int,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    forecast = np.empty((series_index.size, horizon), dtype=np.float64)
    denominator = np.empty(series_index.size, dtype=np.float64)
    for row in range(series_index.size):
        history = panel.series(int(series_index[row]))[: int(origins[row])].astype(np.float64)
        forecast[row] = naive2(history, horizon, period)
        denominator[row] = _history_denominator(history, period)
    return forecast, denominator


def run(config: Config) -> dict[str, object]:
    artifacts = Artifacts(config)
    summary: dict[str, object] = {}
    for frequency in config.run.frequencies:
        settings = config.frequency[frequency]
        panel = load_panel(config, frequency)
        spec = FeatureSpec.from_config(config, settings)
        plan = build_window_plan(config, settings, panel)
        batch = build_batch(panel, spec, plan.validation, True)

        lgbm = LgbmForecaster.load(artifacts.model_directory(frequency, "lgbm"))
        nhits = NhitsForecaster.load(config, artifacts.model_directory(frequency, "nhits"))
        tabular = lgbm.predict(batch).astype(np.float64)
        neural = nhits.predict(batch).astype(np.float64)
        actual = (batch.scaled_targets * batch.scale[:, None]).astype(np.float64)
        reference, denominator = _reference(
            panel, batch.series_index, batch.origins, settings.horizon, settings.period
        )
        base_smape = float(smape_per_series(actual, reference).mean())
        base_mase = float(mase_per_series(actual, reference, denominator).mean())

        weights = np.linspace(0.0, 1.0, config.blend.weight_grid)
        curve = np.empty(weights.size, dtype=np.float64)
        for index, weight in enumerate(weights):
            combined = weight * tabular + (1.0 - weight) * neural
            value_smape = float(smape_per_series(actual, combined).mean())
            value_mase = float(mase_per_series(actual, combined, denominator).mean())
            curve[index] = 0.5 * (value_smape / base_smape + value_mase / base_mase)
        best = int(np.argmin(curve))
        weight = float(weights[best])

        artifacts.ensure(artifacts.blend(frequency)).write_text(
            json.dumps(
                {
                    "weight": weight,
                    "validation_windows": int(batch.series_index.size),
                    "validation_owa_blend": round(float(curve[best]), 4),
                    "validation_owa_lgbm": round(float(curve[-1]), 4),
                    "validation_owa_nhits": round(float(curve[0]), 4),
                }
            )
        )

        stored = np.load(artifacts.forecasts(frequency))
        combined = (weight * stored["lgbm"] + (1.0 - weight) * stored["nhits"]).astype(np.float32)
        np.savez_compressed(
            artifacts.ensure(artifacts.forecasts(frequency)),
            lgbm=stored["lgbm"],
            nhits=stored["nhits"],
            blend=combined,
            scale=stored["scale"],
        )
        summary[frequency] = {
            "weight": weight,
            "validation_owa_blend": round(float(curve[best]), 4),
            "validation_owa_lgbm": round(float(curve[-1]), 4),
            "validation_owa_nhits": round(float(curve[0]), 4),
        }
    return summary
