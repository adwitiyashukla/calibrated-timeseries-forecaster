from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    pass


def _table(root: dict[str, Any], name: str) -> dict[str, Any]:
    value = root.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing table [{name}]")
    return value


def _reject_unknown(table: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = sorted(set(table) - allowed)
    if extra:
        raise ConfigError(f"unknown keys in [{where}]: {', '.join(extra)}")


def _string(table: dict[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"[{where}].{key} must be a non-empty string")
    return value


def _integer(table: dict[str, Any], key: str, where: str, low: int, high: int) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"[{where}].{key} must be an integer")
    if value < low or value > high:
        raise ConfigError(f"[{where}].{key} must be between {low} and {high}, got {value}")
    return value


def _number(table: dict[str, Any], key: str, where: str, low: float, high: float) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"[{where}].{key} must be a number")
    number = float(value)
    if number < low or number > high:
        raise ConfigError(f"[{where}].{key} must be between {low} and {high}, got {number}")
    return number


def _boolean(table: dict[str, Any], key: str, where: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"[{where}].{key} must be true or false")
    return value


def _integer_list(table: dict[str, Any], key: str, where: str, low: int, high: int) -> tuple[int, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"[{where}].{key} must be a non-empty list of integers")
    items = []
    for entry in value:
        if not isinstance(entry, int) or isinstance(entry, bool) or entry < low or entry > high:
            raise ConfigError(f"[{where}].{key} entries must be integers in [{low}, {high}]")
        items.append(entry)
    return tuple(items)


def _number_list(table: dict[str, Any], key: str, where: str, low: float, high: float) -> tuple[float, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"[{where}].{key} must be a non-empty list of numbers")
    items = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, int | float):
            raise ConfigError(f"[{where}].{key} entries must be numbers")
        number = float(entry)
        if number < low or number > high:
            raise ConfigError(f"[{where}].{key} entries must be in [{low}, {high}]")
        items.append(number)
    return tuple(items)


def _string_list(table: dict[str, Any], key: str, where: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"[{where}].{key} must be a non-empty list of strings")
    items = []
    for entry in value:
        if entry not in allowed:
            raise ConfigError(f"[{where}].{key} entries must be one of {', '.join(allowed)}")
        if entry in items:
            raise ConfigError(f"[{where}].{key} contains a duplicate entry {entry}")
        items.append(entry)
    return tuple(items)


FREQUENCY_NAMES: tuple[str, ...] = ("Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly")


@dataclass(frozen=True)
class DataConfig:
    source_base_url: str
    raw_dir: Path
    panel_dir: Path
    download_timeout_seconds: int


@dataclass(frozen=True)
class RunConfig:
    seed: int
    artifact_dir: Path
    frequencies: tuple[str, ...]
    window_cap_scale: float
    holdout_series_fraction: float
    threads: int


@dataclass(frozen=True)
class FrequencyConfig:
    name: str
    horizon: int
    period: int
    lookback: int
    max_windows_per_series: int


@dataclass(frozen=True)
class FeatureConfig:
    scale_floor: float
    include_window_statistics: bool


@dataclass(frozen=True)
class LgbmConfig:
    objective: str
    num_leaves: int
    learning_rate: float
    num_boost_round: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    lambda_l2: float
    max_bin: int
    early_stopping_rounds: int


@dataclass(frozen=True)
class NhitsConfig:
    stacks: int
    mlp_width: int
    mlp_layers: int
    pooling_sizes: tuple[int, ...]
    interpolation_factors: tuple[int, ...]
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    max_train_samples: int


@dataclass(frozen=True)
class BlendConfig:
    weight_grid: int


@dataclass(frozen=True)
class ConformalConfig:
    levels: tuple[float, ...]
    calibration_origins: int
    shrinkage_weight: float
    score_floor: float


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int
    bootstrap_seed: int
    bootstrap_series_fraction: float


@dataclass(frozen=True)
class ServingConfig:
    host: str
    port: int
    onnx_opset: int
    latency_iterations: int
    latency_warmup: int


@dataclass(frozen=True)
class Config:
    data: DataConfig
    run: RunConfig
    frequency: dict[str, FrequencyConfig]
    features: FeatureConfig
    lgbm: LgbmConfig
    nhits: NhitsConfig
    blend: BlendConfig
    conformal: ConformalConfig
    evaluation: EvaluationConfig
    serving: ServingConfig

    def frequencies(self) -> tuple[FrequencyConfig, ...]:
        return tuple(self.frequency[name] for name in self.run.frequencies)

    def window_cap(self, name: str) -> int:
        scaled = self.frequency[name].max_windows_per_series * self.run.window_cap_scale
        return max(1, int(round(scaled)))


def _data(root: dict[str, Any]) -> DataConfig:
    table = _table(root, "data")
    _reject_unknown(
        table,
        {"source_base_url", "raw_dir", "panel_dir", "download_timeout_seconds"},
        "data",
    )
    return DataConfig(
        source_base_url=_string(table, "source_base_url", "data"),
        raw_dir=Path(_string(table, "raw_dir", "data")),
        panel_dir=Path(_string(table, "panel_dir", "data")),
        download_timeout_seconds=_integer(table, "download_timeout_seconds", "data", 1, 100_000),
    )


def _run(root: dict[str, Any]) -> RunConfig:
    table = _table(root, "run")
    _reject_unknown(
        table,
        {
            "seed",
            "artifact_dir",
            "frequencies",
            "window_cap_scale",
            "holdout_series_fraction",
            "threads",
        },
        "run",
    )
    return RunConfig(
        seed=_integer(table, "seed", "run", 0, 2**31 - 1),
        artifact_dir=Path(_string(table, "artifact_dir", "run")),
        frequencies=_string_list(table, "frequencies", "run", FREQUENCY_NAMES),
        window_cap_scale=_number(table, "window_cap_scale", "run", 0.001, 100.0),
        holdout_series_fraction=_number(table, "holdout_series_fraction", "run", 0.001, 0.5),
        threads=_integer(table, "threads", "run", 0, 256),
    )


def _frequency(root: dict[str, Any], names: tuple[str, ...]) -> dict[str, FrequencyConfig]:
    table = _table(root, "frequency")
    _reject_unknown(table, set(FREQUENCY_NAMES), "frequency")
    parsed: dict[str, FrequencyConfig] = {}
    for name in names:
        entry = table.get(name)
        if not isinstance(entry, dict):
            raise ConfigError(f"missing table [frequency.{name}]")
        where = f"frequency.{name}"
        _reject_unknown(
            entry,
            {"horizon", "period", "lookback", "max_windows_per_series"},
            where,
        )
        horizon = _integer(entry, "horizon", where, 1, 1000)
        period = _integer(entry, "period", where, 1, 1000)
        lookback = _integer(entry, "lookback", where, 1, 5000)
        if lookback < period:
            raise ConfigError(f"[{where}].lookback must be at least the seasonal period {period}")
        parsed[name] = FrequencyConfig(
            name=name,
            horizon=horizon,
            period=period,
            lookback=lookback,
            max_windows_per_series=_integer(entry, "max_windows_per_series", where, 1, 100_000),
        )
    return parsed


def _features(root: dict[str, Any]) -> FeatureConfig:
    table = _table(root, "features")
    _reject_unknown(table, {"scale_floor", "include_window_statistics"}, "features")
    return FeatureConfig(
        scale_floor=_number(table, "scale_floor", "features", 1e-12, 1.0),
        include_window_statistics=_boolean(table, "include_window_statistics", "features"),
    )


def _lgbm(root: dict[str, Any]) -> LgbmConfig:
    table = _table(root, "lgbm")
    where = "lgbm"
    _reject_unknown(
        table,
        {
            "objective",
            "num_leaves",
            "learning_rate",
            "num_boost_round",
            "min_data_in_leaf",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l2",
            "max_bin",
            "early_stopping_rounds",
        },
        where,
    )
    objective = _string(table, "objective", where)
    if objective not in {"l1", "l2", "huber"}:
        raise ConfigError("[lgbm].objective must be one of l1, l2, huber")
    return LgbmConfig(
        objective=objective,
        num_leaves=_integer(table, "num_leaves", where, 2, 4096),
        learning_rate=_number(table, "learning_rate", where, 1e-4, 1.0),
        num_boost_round=_integer(table, "num_boost_round", where, 1, 100_000),
        min_data_in_leaf=_integer(table, "min_data_in_leaf", where, 1, 1_000_000),
        feature_fraction=_number(table, "feature_fraction", where, 0.01, 1.0),
        bagging_fraction=_number(table, "bagging_fraction", where, 0.01, 1.0),
        bagging_freq=_integer(table, "bagging_freq", where, 0, 1000),
        lambda_l2=_number(table, "lambda_l2", where, 0.0, 1000.0),
        max_bin=_integer(table, "max_bin", where, 8, 255),
        early_stopping_rounds=_integer(table, "early_stopping_rounds", where, 1, 100_000),
    )


def _nhits(root: dict[str, Any]) -> NhitsConfig:
    table = _table(root, "nhits")
    where = "nhits"
    _reject_unknown(
        table,
        {
            "stacks",
            "mlp_width",
            "mlp_layers",
            "pooling_sizes",
            "interpolation_factors",
            "dropout",
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "early_stopping_patience",
            "max_train_samples",
        },
        where,
    )
    stacks = _integer(table, "stacks", where, 1, 16)
    pooling_sizes = _integer_list(table, "pooling_sizes", where, 1, 512)
    interpolation_factors = _integer_list(table, "interpolation_factors", where, 1, 512)
    if len(pooling_sizes) != stacks:
        raise ConfigError("[nhits].pooling_sizes must have one entry per stack")
    if len(interpolation_factors) != stacks:
        raise ConfigError("[nhits].interpolation_factors must have one entry per stack")
    return NhitsConfig(
        stacks=stacks,
        mlp_width=_integer(table, "mlp_width", where, 8, 4096),
        mlp_layers=_integer(table, "mlp_layers", where, 1, 8),
        pooling_sizes=pooling_sizes,
        interpolation_factors=interpolation_factors,
        dropout=_number(table, "dropout", where, 0.0, 0.9),
        batch_size=_integer(table, "batch_size", where, 1, 100_000),
        epochs=_integer(table, "epochs", where, 1, 10_000),
        learning_rate=_number(table, "learning_rate", where, 1e-6, 1.0),
        weight_decay=_number(table, "weight_decay", where, 0.0, 1.0),
        early_stopping_patience=_integer(table, "early_stopping_patience", where, 1, 10_000),
        max_train_samples=_integer(table, "max_train_samples", where, 100, 100_000_000),
    )


def _blend(root: dict[str, Any]) -> BlendConfig:
    table = _table(root, "blend")
    _reject_unknown(table, {"weight_grid"}, "blend")
    return BlendConfig(weight_grid=_integer(table, "weight_grid", "blend", 2, 1001))


def _conformal(root: dict[str, Any]) -> ConformalConfig:
    table = _table(root, "conformal")
    where = "conformal"
    _reject_unknown(
        table,
        {"levels", "calibration_origins", "shrinkage_weight", "score_floor"},
        where,
    )
    return ConformalConfig(
        levels=_number_list(table, "levels", where, 0.5, 0.999),
        calibration_origins=_integer(table, "calibration_origins", where, 1, 100),
        shrinkage_weight=_number(table, "shrinkage_weight", where, 0.0, 1.0),
        score_floor=_number(table, "score_floor", where, 1e-12, 1.0),
    )


def _evaluation(root: dict[str, Any]) -> EvaluationConfig:
    table = _table(root, "evaluation")
    where = "evaluation"
    _reject_unknown(
        table,
        {"bootstrap_samples", "bootstrap_seed", "bootstrap_series_fraction"},
        where,
    )
    return EvaluationConfig(
        bootstrap_samples=_integer(table, "bootstrap_samples", where, 10, 1_000_000),
        bootstrap_seed=_integer(table, "bootstrap_seed", where, 0, 2**31 - 1),
        bootstrap_series_fraction=_number(table, "bootstrap_series_fraction", where, 0.01, 1.0),
    )


def _serving(root: dict[str, Any]) -> ServingConfig:
    table = _table(root, "serving")
    where = "serving"
    _reject_unknown(
        table,
        {"host", "port", "onnx_opset", "latency_iterations", "latency_warmup"},
        where,
    )
    return ServingConfig(
        host=_string(table, "host", where),
        port=_integer(table, "port", where, 1, 65535),
        onnx_opset=_integer(table, "onnx_opset", where, 11, 25),
        latency_iterations=_integer(table, "latency_iterations", where, 1, 1_000_000),
        latency_warmup=_integer(table, "latency_warmup", where, 0, 1_000_000),
    )


def parse_config(root: dict[str, Any]) -> Config:
    _reject_unknown(
        root,
        {
            "data",
            "run",
            "frequency",
            "features",
            "lgbm",
            "nhits",
            "blend",
            "conformal",
            "evaluation",
            "serving",
        },
        "root",
    )
    run = _run(root)
    return Config(
        data=_data(root),
        run=run,
        frequency=_frequency(root, run.frequencies),
        features=_features(root),
        lgbm=_lgbm(root),
        nhits=_nhits(root),
        blend=_blend(root),
        conformal=_conformal(root),
        evaluation=_evaluation(root),
        serving=_serving(root),
    )


def load_config(path: str | Path = "config.toml") -> Config:
    location = Path(path)
    if not location.is_file():
        raise ConfigError(f"config file not found: {location}")
    with location.open("rb") as handle:
        root = tomllib.load(handle)
    return parse_config(root)
