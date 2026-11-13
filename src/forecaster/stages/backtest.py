from __future__ import annotations

import json

import numpy as np

from forecaster.baselines.classical import METHODS
from forecaster.config import Config
from forecaster.data.panel import SeriesPanel, load_panel
from forecaster.evaluation.paths import Artifacts
from forecaster.features.spec import FeatureSpec
from forecaster.metrics.m4 import mase_per_series, smape_per_series
from forecaster.models.lgbm import LgbmForecaster
from forecaster.models.matrices import build_batch
from forecaster.models.nhits import NhitsForecaster
from forecaster.models.windows import build_window_plan

REFERENCE = "naive2"


def _history_denominator(history: np.ndarray, period: int) -> float:
    if history.size > period:
        value = float(np.mean(np.abs(history[period:] - history[:-period])))
    elif history.size > 1:
        value = float(np.mean(np.abs(np.diff(history))))
    else:
        value = 0.0
    if value > 0.0:
        return value
    fallback = float(np.mean(np.abs(history)))
    return fallback if fallback > 0.0 else 1.0


def _baseline_forecasts(
    panel: SeriesPanel,
    series_index: np.ndarray,
    origins: np.ndarray,
    horizon: int,
    period: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    names = sorted(METHODS)
    output = {name: np.empty((series_index.size, horizon), dtype=np.float64) for name in names}
    denominator = np.empty(series_index.size, dtype=np.float64)
    for row in range(series_index.size):
        history = panel.series(int(series_index[row]))[: int(origins[row])].astype(np.float64)
        denominator[row] = _history_denominator(history, period)
        for name in names:
            output[name][row] = METHODS[name](history, horizon, period)
    return output, denominator


def run(config: Config) -> dict[str, object]:
    artifacts = Artifacts(config)
    report: dict[str, object] = {}
    pooled: dict[str, list[np.ndarray]] = {}
    pooled_mase: dict[str, list[np.ndarray]] = {}

    for frequency in config.run.frequencies:
        settings = config.frequency[frequency]
        panel = load_panel(config, frequency)
        spec = FeatureSpec.from_config(config, settings)
        plan = build_window_plan(config, settings, panel)
        batch = build_batch(panel, spec, plan.calibration, True)

        lgbm = LgbmForecaster.load(artifacts.model_directory(frequency, "lgbm"))
        nhits = NhitsForecaster.load(config, artifacts.model_directory(frequency, "nhits"))
        weight = json.loads(artifacts.blend(frequency).read_text())["weight"]
        forecasts: dict[str, np.ndarray] = {
            "lgbm": lgbm.predict(batch).astype(np.float64),
            "nhits": nhits.predict(batch).astype(np.float64),
        }
        forecasts["blend"] = weight * forecasts["lgbm"] + (1.0 - weight) * forecasts["nhits"]
        baselines, denominator = _baseline_forecasts(
            panel, batch.series_index, batch.origins, settings.horizon, settings.period
        )
        forecasts.update(baselines)
        actual = (batch.scaled_targets * batch.scale[:, None]).astype(np.float64)

        lengths = panel.lengths()
        offset = (lengths[batch.series_index] - settings.horizon) - batch.origins

        smape = {name: smape_per_series(actual, value) for name, value in forecasts.items()}
        mase = {
            name: mase_per_series(actual, value, denominator) for name, value in forecasts.items()
        }
        for name in forecasts:
            pooled.setdefault(name, []).append(smape[name])
            pooled_mase.setdefault(name, []).append(mase[name])

        report[frequency] = {
            "windows": int(batch.series_index.size),
            "overall": _table(smape, mase),
            "by_origin_offset": {
                str(int(value)): _table(
                    {name: smape[name][offset == value] for name in forecasts},
                    {name: mase[name][offset == value] for name in forecasts},
                )
                for value in np.unique(offset)
            },
        }

    stacked_smape = {name: np.concatenate(values) for name, values in pooled.items()}
    stacked_mase = {name: np.concatenate(values) for name, values in pooled_mase.items()}
    summary = {"windows": int(stacked_smape[REFERENCE].size), "overall": _table(stacked_smape, stacked_mase)}
    report["all"] = summary
    artifacts.ensure(artifacts.results("backtest")).write_text(json.dumps(report, indent=2))
    return summary


def _table(
    smape: dict[str, np.ndarray], mase: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    base_smape = float(smape[REFERENCE].mean())
    base_mase = float(mase[REFERENCE].mean())
    table = {}
    for name in sorted(smape):
        value_smape = float(smape[name].mean())
        value_mase = float(mase[name].mean())
        table[name] = {
            "smape": round(value_smape, 4),
            "mase": round(value_mase, 4),
            "owa": round(0.5 * (value_smape / base_smape + value_mase / base_mase), 4),
        }
    return table
