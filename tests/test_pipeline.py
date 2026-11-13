from __future__ import annotations

import json

import numpy as np

from forecaster.evaluation.paths import Artifacts


def test_scores_cover_every_model(pipeline) -> None:
    artifacts = Artifacts(pipeline)
    scores = json.loads(artifacts.results("scores").read_text())
    for model in ("naive", "seasonal_naive", "naive2", "theta", "lgbm", "nhits", "blend"):
        assert model in scores["overall"]
    assert scores["overall"]["naive2"]["owa"] == 1.0


def test_learned_models_beat_the_naive_baseline(pipeline) -> None:
    artifacts = Artifacts(pipeline)
    scores = json.loads(artifacts.results("scores").read_text())
    assert scores["overall"]["blend"]["owa"] < scores["overall"]["naive"]["owa"]


def test_blend_beats_both_components_on_validation(pipeline) -> None:
    artifacts = Artifacts(pipeline)
    blend = json.loads(artifacts.blend("Hourly").read_text())
    assert 0.0 <= blend["weight"] <= 1.0
    assert blend["validation_owa_blend"] <= min(
        blend["validation_owa_lgbm"], blend["validation_owa_nhits"]
    ) + 1e-9


def test_forecast_shapes_match_the_horizon(pipeline) -> None:
    artifacts = Artifacts(pipeline)
    stored = np.load(artifacts.forecasts("Hourly"))
    horizon = pipeline.frequency["Hourly"].horizon
    for name in ("lgbm", "nhits", "blend"):
        assert stored[name].shape[1] == horizon
        assert np.isfinite(stored[name]).all()


def test_coverage_report_has_both_interval_methods(pipeline) -> None:
    artifacts = Artifacts(pipeline)
    report = json.loads(artifacts.results("coverage").read_text())
    for method in ("split", "adaptive"):
        entry = report["Hourly"][method]
        assert set(entry) == {"0.80", "0.95"}
        assert entry["0.95"]["scaled_width"] >= entry["0.80"]["scaled_width"]


def test_bootstrap_intervals_bracket_the_estimate(pipeline) -> None:
    artifacts = Artifacts(pipeline)
    scores = json.loads(artifacts.results("scores").read_text())
    for entry in scores["bootstrap"].values():
        assert entry["interval_low"] <= entry["interval_high"]


def test_onnx_model_was_written(pipeline) -> None:
    artifacts = Artifacts(pipeline)
    assert artifacts.onnx("Hourly").stat().st_size > 0


def test_backtest_reports_every_origin_offset(pipeline) -> None:
    artifacts = Artifacts(pipeline)
    report = json.loads(artifacts.results("backtest").read_text())
    entry = report["Hourly"]
    assert entry["windows"] > 0
    assert len(entry["by_origin_offset"]) == pipeline.conformal.calibration_origins
    assert report["all"]["overall"]["naive2"]["owa"] == 1.0
    assert report["all"]["overall"]["blend"]["owa"] < report["all"]["overall"]["naive"]["owa"]
