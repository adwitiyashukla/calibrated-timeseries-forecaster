from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from forecaster.config import Config
from forecaster.features.spec import FeatureSpec
from forecaster.models.matrices import WindowBatch


@dataclass(frozen=True)
class LgbmForecaster:
    boosters: tuple[lgb.Booster, ...]
    feature_names: tuple[str, ...]

    def predict_scaled(self, features: np.ndarray) -> np.ndarray:
        columns = [booster.predict(features, num_threads=0) for booster in self.boosters]
        return np.stack(columns, axis=1).astype(np.float32)

    def predict(self, batch: WindowBatch) -> np.ndarray:
        return self.predict_scaled(batch.features) * batch.scale[:, None]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for step, booster in enumerate(self.boosters):
            booster.save_model(str(directory / f"step_{step:03d}.txt"))

    @classmethod
    def load(cls, directory: Path) -> LgbmForecaster:
        paths = sorted(directory.glob("step_*.txt"))
        if not paths:
            raise FileNotFoundError(f"no LightGBM models in {directory}")
        boosters = tuple(lgb.Booster(model_file=str(path)) for path in paths)
        return cls(boosters=boosters, feature_names=tuple(boosters[0].feature_name()))


def _parameters(config: Config) -> dict[str, object]:
    settings = config.lgbm
    return {
        "objective": settings.objective,
        "metric": "l1",
        "num_leaves": settings.num_leaves,
        "learning_rate": settings.learning_rate,
        "min_data_in_leaf": settings.min_data_in_leaf,
        "feature_fraction": settings.feature_fraction,
        "bagging_fraction": settings.bagging_fraction,
        "bagging_freq": settings.bagging_freq,
        "lambda_l2": settings.lambda_l2,
        "max_bin": settings.max_bin,
        "num_threads": config.run.threads,
        "seed": config.run.seed,
        "verbosity": -1,
        "force_col_wise": True,
    }


def train_lgbm(
    config: Config,
    spec: FeatureSpec,
    train: WindowBatch,
    validation: WindowBatch,
) -> tuple[LgbmForecaster, list[int]]:
    names = spec.names()
    parameters = _parameters(config)
    training_set = lgb.Dataset(
        train.features,
        label=np.ascontiguousarray(train.scaled_targets[:, 0]),
        feature_name=names,
        free_raw_data=False,
        params=parameters,
    )
    validation_set = lgb.Dataset(
        validation.features,
        label=np.ascontiguousarray(validation.scaled_targets[:, 0]),
        feature_name=names,
        reference=training_set,
        free_raw_data=False,
        params=parameters,
    )
    boosters = []
    rounds = []
    for step in range(spec.horizon):
        training_set.set_label(np.ascontiguousarray(train.scaled_targets[:, step]))
        validation_set.set_label(np.ascontiguousarray(validation.scaled_targets[:, step]))
        booster = lgb.train(
            parameters,
            training_set,
            num_boost_round=config.lgbm.num_boost_round,
            valid_sets=[validation_set],
            callbacks=[lgb.early_stopping(config.lgbm.early_stopping_rounds, verbose=False)],
        )
        boosters.append(booster)
        rounds.append(int(booster.best_iteration or config.lgbm.num_boost_round))
    return LgbmForecaster(boosters=tuple(boosters), feature_names=tuple(names)), rounds
