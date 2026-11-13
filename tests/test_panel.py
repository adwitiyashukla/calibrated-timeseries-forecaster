from __future__ import annotations

import csv

import numpy as np

from forecaster.data.download import missing_files, required_files
from forecaster.data.panel import load_panel, load_test, panel_exists
from forecaster.stages import prepare


def test_required_files_cover_every_frequency(config) -> None:
    names = required_files(config)
    assert "M4-info.csv" in names
    for frequency in config.run.frequencies:
        assert f"{frequency}-train.csv" in names
        assert f"{frequency}-test.csv" in names


def test_no_files_are_missing_in_the_fixture(config) -> None:
    assert missing_files(config) == []


def test_panel_round_trips_the_source_csv(config) -> None:
    prepare.run(config)
    assert panel_exists(config, "Hourly")
    panel = load_panel(config, "Hourly")
    with (config.data.raw_dir / "Hourly-train.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        rows = [row for row in reader]
    assert panel.size == len(rows)
    assert panel.series_ids[0] == rows[0][0]
    expected = np.asarray([float(cell) for cell in rows[0][1:] if cell.strip()], dtype=np.float32)
    assert np.allclose(panel.series(0), expected, atol=1e-5)


def test_test_panel_has_horizon_columns(config) -> None:
    prepare.run(config)
    actual = load_test(config, "Hourly")
    assert actual.shape == (load_panel(config, "Hourly").size, config.frequency["Hourly"].horizon)
