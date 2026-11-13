from __future__ import annotations

import numpy as np

from forecaster.data.panel import load_panel
from forecaster.features.spec import FeatureSpec, assemble_targets, assemble_windows
from forecaster.stages import prepare


def _spec(config) -> FeatureSpec:
    return FeatureSpec.from_config(config, config.frequency["Hourly"])


def test_feature_names_match_width(config) -> None:
    spec = _spec(config)
    assert len(spec.names()) == spec.width()
    assert len(set(spec.names())) == spec.width()


def test_transform_is_row_independent(config) -> None:
    spec = _spec(config)
    generator = np.random.default_rng(0)
    windows = generator.normal(size=(9, spec.lookback)).astype(np.float32) + 20.0
    context = np.full(9, spec.lookback, dtype=np.float32)
    origins = np.arange(50, 59, dtype=np.int64)
    batch = spec.transform(windows, context, origins)
    for row in range(9):
        single = spec.transform(windows[row : row + 1], context[row : row + 1], origins[row : row + 1])
        assert np.allclose(batch[row], single[0], atol=1e-6)


def test_scale_is_mean_absolute_value(config) -> None:
    spec = _spec(config)
    windows = np.array([[1.0, -3.0, 2.0, 6.0] * (spec.lookback // 4)], dtype=np.float32)
    assert abs(float(spec.scale(windows)[0]) - 3.0) < 1e-6


def test_lag_one_is_the_most_recent_observation(config) -> None:
    spec = _spec(config)
    windows = np.arange(spec.lookback, dtype=np.float32)[None, :] + 1.0
    features = spec.transform(windows, np.array([spec.lookback], dtype=np.float32), np.array([0]))
    scale = float(spec.scale(windows)[0])
    assert abs(features[0, 0] - windows[0, -1] / scale) < 1e-6


def test_short_history_is_padded_with_the_first_value(config) -> None:
    prepare.run(config)
    spec = _spec(config)
    panel = load_panel(config, "Hourly")
    origins = np.array([3], dtype=np.int64)
    windows, context = assemble_windows(panel, np.array([0], dtype=np.int32), origins, spec.lookback)
    series = panel.series(0)
    assert context[0] == 3
    assert np.allclose(windows[0, : spec.lookback - 3], series[0])
    assert np.allclose(windows[0, spec.lookback - 3 :], series[:3])


def test_targets_follow_the_origin(config) -> None:
    prepare.run(config)
    spec = _spec(config)
    panel = load_panel(config, "Hourly")
    origins = np.array([40], dtype=np.int64)
    targets = assemble_targets(panel, np.array([0], dtype=np.int32), origins, spec.horizon)
    assert np.allclose(targets[0], panel.series(0)[40 : 40 + spec.horizon])
