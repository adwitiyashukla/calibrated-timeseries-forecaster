from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from forecaster.config import Config, parse_config

SERIES_COUNT = 40
SERIES_LENGTH = 220
HORIZON = 12
PERIOD = 4
LOOKBACK = 16


def _synthetic_series(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    steps = np.arange(SERIES_LENGTH + HORIZON, dtype=np.float64)
    level = 40.0 + 4.0 * generator.random()
    trend = 0.05 * generator.random() * steps
    season = 3.0 * np.sin(2.0 * np.pi * steps / PERIOD + generator.random())
    noise = generator.normal(scale=0.4, size=steps.size)
    return level + trend + season + noise


def _write_wide(path: Path, rows: list[tuple[str, np.ndarray]]) -> None:
    width = max(row.size for _, row in rows)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"V{index + 1}" for index in range(width + 1)])
        for identifier, row in rows:
            padded = [f"{value:.6f}" for value in row] + [""] * (width - row.size)
            writer.writerow([identifier, *padded])


@pytest.fixture(scope="session")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("forecaster")
    raw = root / "raw"
    raw.mkdir()
    train_rows = []
    test_rows = []
    for index in range(SERIES_COUNT):
        full = _synthetic_series(index)
        identifier = f"H{index + 1}"
        train_rows.append((identifier, full[:SERIES_LENGTH]))
        test_rows.append((identifier, full[SERIES_LENGTH:]))
    _write_wide(raw / "Hourly-train.csv", train_rows)
    _write_wide(raw / "Hourly-test.csv", test_rows)
    (raw / "M4-info.csv").write_text("M4id,category,Frequency,Horizon,SP,StartingDate\n")
    return root


@pytest.fixture(scope="session")
def config(workspace: Path) -> Config:
    return parse_config(
        {
            "data": {
                "source_base_url": "https://example.invalid/dataset",
                "raw_dir": str(workspace / "raw"),
                "panel_dir": str(workspace / "panel"),
                "download_timeout_seconds": 5,
            },
            "run": {
                "seed": 3,
                "artifact_dir": str(workspace / "artifacts"),
                "frequencies": ["Hourly"],
                "window_cap_scale": 1.0,
                "holdout_series_fraction": 0.2,
                "threads": 1,
            },
            "frequency": {
                "Hourly": {
                    "horizon": HORIZON,
                    "period": PERIOD,
                    "lookback": LOOKBACK,
                    "max_windows_per_series": 60,
                }
            },
            "features": {"scale_floor": 1e-8, "include_window_statistics": True},
            "lgbm": {
                "objective": "l1",
                "num_leaves": 15,
                "learning_rate": 0.1,
                "num_boost_round": 25,
                "min_data_in_leaf": 20,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "lambda_l2": 1.0,
                "max_bin": 63,
                "early_stopping_rounds": 10,
            },
            "nhits": {
                "stacks": 2,
                "mlp_width": 32,
                "mlp_layers": 2,
                "pooling_sizes": [2, 1],
                "interpolation_factors": [2, 1],
                "dropout": 0.0,
                "batch_size": 128,
                "epochs": 3,
                "learning_rate": 0.005,
                "weight_decay": 0.0,
                "early_stopping_patience": 2,
                "max_train_samples": 5000,
            },
            "blend": {"weight_grid": 11},
            "conformal": {
                "levels": [0.8, 0.95],
                "calibration_origins": 4,
                "shrinkage_weight": 0.5,
                "score_floor": 1e-8,
            },
            "evaluation": {
                "bootstrap_samples": 50,
                "bootstrap_seed": 5,
                "bootstrap_series_fraction": 1.0,
            },
            "serving": {
                "host": "127.0.0.1",
                "port": 8123,
                "onnx_opset": 17,
                "latency_iterations": 5,
                "latency_warmup": 1,
            },
        }
    )


@pytest.fixture(scope="session")
def pipeline(config: Config) -> Config:
    from forecaster.stages import (
        backtest,
        baselines,
        blend,
        calibrate,
        evaluate,
        export,
        prepare,
        train,
    )

    prepare.run(config)
    baselines.run(config)
    train.run(config, progress=False)
    blend.run(config)
    calibrate.run(config)
    backtest.run(config)
    export.run(config)
    evaluate.run(config)
    return config
