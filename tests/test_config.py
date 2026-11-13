from __future__ import annotations

import copy

import pytest

from forecaster.config import ConfigError, load_config, parse_config


def test_repository_config_loads() -> None:
    config = load_config("config.toml")
    assert config.run.frequencies
    for settings in config.frequencies():
        assert settings.lookback >= settings.period
        assert settings.horizon >= 1


def test_unknown_key_is_rejected(config) -> None:
    document = _document(config)
    document["run"]["unexpected"] = 1
    with pytest.raises(ConfigError, match="unknown keys"):
        parse_config(document)


def test_out_of_range_value_is_rejected(config) -> None:
    document = _document(config)
    document["lgbm"]["feature_fraction"] = 1.5
    with pytest.raises(ConfigError, match="feature_fraction"):
        parse_config(document)


def test_pooling_sizes_must_match_stacks(config) -> None:
    document = _document(config)
    document["nhits"]["pooling_sizes"] = [2]
    document["nhits"]["stacks"] = 2
    with pytest.raises(ConfigError, match="pooling_sizes"):
        parse_config(document)


def test_missing_table_is_reported(config) -> None:
    document = _document(config)
    del document["conformal"]
    with pytest.raises(ConfigError, match="conformal"):
        parse_config(document)


def _document(config) -> dict:
    return copy.deepcopy(
        {
            "data": {
                "source_base_url": config.data.source_base_url,
                "raw_dir": str(config.data.raw_dir),
                "panel_dir": str(config.data.panel_dir),
                "download_timeout_seconds": config.data.download_timeout_seconds,
            },
            "run": {
                "seed": config.run.seed,
                "artifact_dir": str(config.run.artifact_dir),
                "frequencies": list(config.run.frequencies),
                "window_cap_scale": config.run.window_cap_scale,
                "holdout_series_fraction": config.run.holdout_series_fraction,
                "threads": config.run.threads,
            },
            "frequency": {
                name: {
                    "horizon": settings.horizon,
                    "period": settings.period,
                    "lookback": settings.lookback,
                    "max_windows_per_series": settings.max_windows_per_series,
                }
                for name, settings in config.frequency.items()
            },
            "features": {
                "scale_floor": config.features.scale_floor,
                "include_window_statistics": config.features.include_window_statistics,
            },
            "lgbm": {
                "objective": config.lgbm.objective,
                "num_leaves": config.lgbm.num_leaves,
                "learning_rate": config.lgbm.learning_rate,
                "num_boost_round": config.lgbm.num_boost_round,
                "min_data_in_leaf": config.lgbm.min_data_in_leaf,
                "feature_fraction": config.lgbm.feature_fraction,
                "bagging_fraction": config.lgbm.bagging_fraction,
                "bagging_freq": config.lgbm.bagging_freq,
                "lambda_l2": config.lgbm.lambda_l2,
                "max_bin": config.lgbm.max_bin,
                "early_stopping_rounds": config.lgbm.early_stopping_rounds,
            },
            "nhits": {
                "stacks": config.nhits.stacks,
                "mlp_width": config.nhits.mlp_width,
                "mlp_layers": config.nhits.mlp_layers,
                "pooling_sizes": list(config.nhits.pooling_sizes),
                "interpolation_factors": list(config.nhits.interpolation_factors),
                "dropout": config.nhits.dropout,
                "batch_size": config.nhits.batch_size,
                "epochs": config.nhits.epochs,
                "learning_rate": config.nhits.learning_rate,
                "weight_decay": config.nhits.weight_decay,
                "early_stopping_patience": config.nhits.early_stopping_patience,
                "max_train_samples": config.nhits.max_train_samples,
            },
            "blend": {"weight_grid": config.blend.weight_grid},
            "conformal": {
                "levels": list(config.conformal.levels),
                "calibration_origins": config.conformal.calibration_origins,
                "shrinkage_weight": config.conformal.shrinkage_weight,
                "score_floor": config.conformal.score_floor,
            },
            "evaluation": {
                "bootstrap_samples": config.evaluation.bootstrap_samples,
                "bootstrap_seed": config.evaluation.bootstrap_seed,
                "bootstrap_series_fraction": config.evaluation.bootstrap_series_fraction,
            },
            "serving": {
                "host": config.serving.host,
                "port": config.serving.port,
                "onnx_opset": config.serving.onnx_opset,
                "latency_iterations": config.serving.latency_iterations,
                "latency_warmup": config.serving.latency_warmup,
            },
        }
    )
