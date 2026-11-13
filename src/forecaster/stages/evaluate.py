from __future__ import annotations

import json

import numpy as np

from forecaster.config import Config
from forecaster.conformal.intervals import (
    ConformalCalibration,
    coverage,
    coverage_by_step,
    pinball_loss,
    scaled_width,
)
from forecaster.data.panel import load_panel, load_test
from forecaster.evaluation.bootstrap import bootstrap_owa_difference, overall_owa
from forecaster.evaluation.paths import Artifacts
from forecaster.metrics.m4 import (
    guard_denominator,
    mase_denominator,
    mase_per_series,
    smape_per_series,
)

BASELINE_MODELS: tuple[str, ...] = ("naive", "seasonal_naive", "naive2", "theta")
LEARNED_MODELS: tuple[str, ...] = ("lgbm", "nhits", "blend")


def _frequency_table(
    smape: dict[str, np.ndarray], mase: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    base_smape = float(smape["naive2"].mean())
    base_mase = float(mase["naive2"].mean())
    for model in smape:
        model_smape = float(smape[model].mean())
        model_mase = float(mase[model].mean())
        table[model] = {
            "smape": round(model_smape, 4),
            "mase": round(model_mase, 4),
            "owa": round(0.5 * (model_smape / base_smape + model_mase / base_mase), 4),
        }
    return table


def run(config: Config) -> dict[str, object]:
    artifacts = Artifacts(config)
    models = BASELINE_MODELS + LEARNED_MODELS
    pooled_smape: dict[str, list[np.ndarray]] = {model: [] for model in models}
    pooled_mase: dict[str, list[np.ndarray]] = {model: [] for model in models}
    by_frequency: dict[str, object] = {}
    coverage_report: dict[str, object] = {}

    for frequency in config.run.frequencies:
        settings = config.frequency[frequency]
        panel = load_panel(config, frequency)
        actual = load_test(config, frequency)
        denominator, degenerate = guard_denominator(
            mase_denominator(panel, settings.period), panel
        )
        stored = np.load(artifacts.baselines(frequency))
        baseline_forecasts = {
            str(name): stored["values"][position]
            for position, name in enumerate(stored["methods"])
        }
        learned_forecasts = np.load(artifacts.forecasts(frequency))

        smape: dict[str, np.ndarray] = {}
        mase: dict[str, np.ndarray] = {}
        for model in models:
            source = baseline_forecasts if model in BASELINE_MODELS else learned_forecasts
            forecast = source[model].astype(np.float64)
            smape[model] = smape_per_series(actual.astype(np.float64), forecast)
            mase[model] = mase_per_series(actual.astype(np.float64), forecast, denominator)
            pooled_smape[model].append(smape[model])
            pooled_mase[model].append(mase[model])

        by_frequency[frequency] = {
            "series": panel.size,
            "degenerate_denominators": degenerate,
            "scores": _frequency_table(smape, mase),
        }
        coverage_report[frequency] = _coverage(
            config, artifacts, frequency, actual, learned_forecasts, denominator
        )

    overall: dict[str, dict[str, float]] = {}
    base_smape = np.concatenate(pooled_smape["naive2"])
    base_mase = np.concatenate(pooled_mase["naive2"])
    stacked_smape = {model: np.concatenate(pooled_smape[model]) for model in models}
    stacked_mase = {model: np.concatenate(pooled_mase[model]) for model in models}
    for model in models:
        overall[model] = {
            "smape": round(float(stacked_smape[model].mean()), 4),
            "mase": round(float(stacked_mase[model].mean()), 4),
            "owa": round(
                overall_owa(stacked_smape[model], stacked_mase[model], base_smape, base_mase), 4
            ),
        }

    comparisons = {}
    for challenger, champion in (
        ("blend", "lgbm"),
        ("blend", "nhits"),
        ("blend", "theta"),
        ("lgbm", "theta"),
        ("nhits", "theta"),
    ):
        interval = bootstrap_owa_difference(
            stacked_smape[challenger],
            stacked_mase[challenger],
            stacked_smape[champion],
            stacked_mase[champion],
            base_smape,
            base_mase,
            config.evaluation.bootstrap_samples,
            config.evaluation.bootstrap_seed,
            config.evaluation.bootstrap_series_fraction,
        )
        comparisons[f"{challenger}_minus_{champion}"] = {
            "owa_difference": round(interval.estimate, 5),
            "interval_low": round(interval.low, 5),
            "interval_high": round(interval.high, 5),
            "significant": interval.excludes_zero(),
        }

    scores = {
        "series": int(base_smape.size),
        "overall": overall,
        "by_frequency": by_frequency,
        "bootstrap": comparisons,
    }
    artifacts.ensure(artifacts.results("scores")).write_text(json.dumps(scores, indent=2))
    artifacts.ensure(artifacts.results("coverage")).write_text(
        json.dumps(coverage_report, indent=2)
    )
    return {"overall": overall, "bootstrap": comparisons}


def _coverage(
    config: Config,
    artifacts: Artifacts,
    frequency: str,
    actual: np.ndarray,
    learned_forecasts: np.lib.npyio.NpzFile,
    denominator: np.ndarray,
) -> dict[str, object]:
    calibration = ConformalCalibration.load(artifacts.conformal(frequency))
    point = learned_forecasts["blend"].astype(np.float64)
    scale = learned_forecasts["scale"].astype(np.float64)
    series_index = np.arange(point.shape[0])
    report: dict[str, object] = {}
    for method, adaptive in (("split", False), ("adaptive", True)):
        lower, upper = calibration.intervals(point, scale, series_index, adaptive)
        observed = coverage(actual.astype(np.float64), lower, upper)
        steps = coverage_by_step(actual.astype(np.float64), lower, upper)
        widths = scaled_width(lower, upper, denominator)
        entries = {}
        for position, level in enumerate(calibration.levels):
            tail = (1.0 - level) / 2.0
            entries[f"{level:.2f}"] = {
                "coverage": round(float(observed[position]), 4),
                "scaled_width": round(float(widths[position]), 4),
                "coverage_by_step": [round(float(value), 4) for value in steps[position]],
                "pinball_lower": round(
                    pinball_loss(
                        actual.astype(np.float64), lower[position], tail, denominator
                    ),
                    5,
                ),
                "pinball_upper": round(
                    pinball_loss(
                        actual.astype(np.float64), upper[position], 1.0 - tail, denominator
                    ),
                    5,
                ),
            }
        report[method] = entries
    return report
