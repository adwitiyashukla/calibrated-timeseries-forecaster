from __future__ import annotations

import numpy as np

from forecaster.data.panel import load_panel
from forecaster.models.windows import build_window_plan, minimum_context
from forecaster.stages import prepare


def test_windows_never_reach_into_the_test_horizon(config) -> None:
    prepare.run(config)
    settings = config.frequency["Hourly"]
    panel = load_panel(config, "Hourly")
    plan = build_window_plan(config, settings, panel)
    lengths = panel.lengths()
    for origin_set in (plan.train, plan.validation, plan.calibration):
        latest = origin_set.origins + settings.horizon
        assert np.all(latest <= lengths[origin_set.series_index])
        assert np.all(origin_set.origins >= minimum_context(settings))


def test_training_origins_precede_calibration_origins(config) -> None:
    prepare.run(config)
    settings = config.frequency["Hourly"]
    panel = load_panel(config, "Hourly")
    plan = build_window_plan(config, settings, panel)
    earliest_calibration = np.full(panel.size, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(earliest_calibration, plan.calibration.series_index, plan.calibration.origins)
    for origin_set in (plan.train, plan.validation):
        assert np.all(origin_set.origins < earliest_calibration[origin_set.series_index])


def test_holdout_series_are_disjoint_from_training_series(config) -> None:
    prepare.run(config)
    settings = config.frequency["Hourly"]
    panel = load_panel(config, "Hourly")
    plan = build_window_plan(config, settings, panel)
    assert not set(plan.train.series_index.tolist()) & set(plan.validation.series_index.tolist())
    assert plan.validation.size > 0


def test_test_origins_equal_series_lengths(config) -> None:
    prepare.run(config)
    settings = config.frequency["Hourly"]
    panel = load_panel(config, "Hourly")
    plan = build_window_plan(config, settings, panel)
    assert np.array_equal(plan.test_origins, panel.lengths())
