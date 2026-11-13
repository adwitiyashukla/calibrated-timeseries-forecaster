from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import onnxruntime
import polars as pl

from forecaster.config import Config
from forecaster.conformal.intervals import ConformalCalibration
from forecaster.evaluation.paths import Artifacts
from forecaster.features.spec import FeatureSpec
from forecaster.models.lgbm import LgbmForecaster


@dataclass(frozen=True)
class Prediction:
    point: np.ndarray
    lower: dict[str, np.ndarray]
    upper: dict[str, np.ndarray]


class FrequencyPredictor:
    def __init__(self, config: Config, frequency: str) -> None:
        artifacts = Artifacts(config)
        settings = config.frequency[frequency]
        self.frequency = frequency
        self.spec = FeatureSpec.from_config(config, settings)
        self.lgbm = LgbmForecaster.load(artifacts.model_directory(frequency, "lgbm"))
        self.session = onnxruntime.InferenceSession(
            str(artifacts.onnx(frequency)), providers=["CPUExecutionProvider"]
        )
        self.weight = float(json.loads(artifacts.blend(frequency).read_text())["weight"])
        self.calibration = ConformalCalibration.load(artifacts.conformal(frequency))
        index = pl.read_parquet(config.data.panel_dir / frequency / "index.parquet")
        self.series_lookup = dict(
            zip(index["series_id"].to_list(), index["series_index"].to_list(), strict=True)
        )

    def prepare_window(self, history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lookback = self.spec.lookback
        available = min(history.size, lookback)
        window = np.empty((1, lookback), dtype=np.float32)
        if available < lookback:
            window[0, : lookback - available] = history[0]
            window[0, lookback - available :] = history[-available:]
        else:
            window[0] = history[-lookback:]
        return window, np.asarray([available], dtype=np.float32)

    def predict(self, history: np.ndarray, series_id: str | None = None) -> Prediction:
        window, context = self.prepare_window(history)
        origins = np.asarray([history.size], dtype=np.int64)
        features = self.spec.transform(window, context, origins)
        scaled_window, scale = self.spec.scaled_windows(window)
        tabular = self.lgbm.predict_scaled(features)
        neural = self.session.run(["forecast"], {"window": scaled_window.astype(np.float32)})[0]
        blended = self.weight * tabular + (1.0 - self.weight) * neural
        point = (blended * scale[:, None]).astype(np.float64)

        adaptive = series_id is not None and series_id in self.series_lookup
        index = np.asarray(
            [self.series_lookup[series_id] if adaptive else 0], dtype=np.int64
        )
        lower, upper = self.calibration.intervals(point, scale.astype(np.float64), index, adaptive)
        levels = self.calibration.levels
        return Prediction(
            point=point[0],
            lower={f"{level:.2f}": lower[position, 0] for position, level in enumerate(levels)},
            upper={f"{level:.2f}": upper[position, 0] for position, level in enumerate(levels)},
        )


class PredictorRegistry:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.cache: dict[str, FrequencyPredictor] = {}

    def get(self, frequency: str) -> FrequencyPredictor:
        if frequency not in self.config.frequency:
            raise KeyError(frequency)
        if frequency not in self.cache:
            self.cache[frequency] = FrequencyPredictor(self.config, frequency)
        return self.cache[frequency]
