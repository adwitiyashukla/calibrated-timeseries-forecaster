from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from forecaster.config import Config
from forecaster.features.spec import FeatureSpec
from forecaster.models.matrices import WindowBatch


def _interpolate(values: torch.Tensor, size: int) -> torch.Tensor:
    if values.shape[1] == size:
        return values
    if values.shape[1] == 1:
        return values.expand(-1, size)
    return nn.functional.interpolate(
        values.unsqueeze(1), size=size, mode="linear", align_corners=False
    ).squeeze(1)


class NhitsBlock(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        width: int,
        layers: int,
        pooling: int,
        interpolation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.pooling = pooling
        self.backcast_size = max(1, math.ceil(lookback / interpolation))
        self.forecast_size = max(1, math.ceil(horizon / interpolation))
        pooled = math.ceil(lookback / pooling)
        self.pool = nn.MaxPool1d(pooling, stride=pooling, ceil_mode=True)
        blocks: list[nn.Module] = []
        size = pooled
        for _ in range(layers):
            blocks.append(nn.Linear(size, width))
            blocks.append(nn.ReLU())
            if dropout > 0.0:
                blocks.append(nn.Dropout(dropout))
            size = width
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(width, self.backcast_size + self.forecast_size)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(residual.unsqueeze(1)).squeeze(1)
        hidden = self.body(pooled)
        theta = self.head(hidden)
        backcast = _interpolate(theta[:, : self.backcast_size], self.lookback)
        forecast = _interpolate(theta[:, self.backcast_size :], self.horizon)
        return backcast, forecast


class Nhits(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        width: int,
        layers: int,
        pooling_sizes: tuple[int, ...],
        interpolation_factors: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.blocks = nn.ModuleList(
            [
                NhitsBlock(lookback, horizon, width, layers, pooling, interpolation, dropout)
                for pooling, interpolation in zip(pooling_sizes, interpolation_factors, strict=True)
            ]
        )

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        residual = window
        forecast = torch.zeros(window.shape[0], self.horizon, dtype=window.dtype)
        for block in self.blocks:
            backcast, partial = block(residual)
            residual = residual - backcast
            forecast = forecast + partial
        return forecast


@dataclass(frozen=True)
class NhitsForecaster:
    model: Nhits
    lookback: int
    horizon: int

    def predict(self, batch: WindowBatch, batch_size: int = 4096) -> np.ndarray:
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, batch.scaled_windows.shape[0], batch_size):
                chunk = torch.from_numpy(batch.scaled_windows[start : start + batch_size])
                outputs.append(self.model(chunk).numpy())
        scaled = np.concatenate(outputs, axis=0)
        return (scaled * batch.scale[:, None]).astype(np.float32)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "weights.pt")
        shape = {"lookback": self.lookback, "horizon": self.horizon}
        (directory / "shape.json").write_text(json.dumps(shape))

    @classmethod
    def load(cls, config: Config, directory: Path) -> NhitsForecaster:
        shape = json.loads((directory / "shape.json").read_text())
        model = build_model(config, shape["lookback"], shape["horizon"])
        model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu"))
        return cls(model=model, lookback=shape["lookback"], horizon=shape["horizon"])


def build_model(config: Config, lookback: int, horizon: int) -> Nhits:
    settings = config.nhits
    return Nhits(
        lookback=lookback,
        horizon=horizon,
        width=settings.mlp_width,
        layers=settings.mlp_layers,
        pooling_sizes=settings.pooling_sizes,
        interpolation_factors=settings.interpolation_factors,
        dropout=settings.dropout,
    )


def train_nhits(
    config: Config,
    spec: FeatureSpec,
    train: WindowBatch,
    validation: WindowBatch,
    progress: bool = False,
) -> tuple[NhitsForecaster, dict[str, float]]:
    torch.manual_seed(config.run.seed)
    if config.run.threads > 0:
        torch.set_num_threads(config.run.threads)
    settings = config.nhits
    generator = np.random.default_rng(config.run.seed)

    inputs = train.scaled_windows
    targets = train.scaled_targets
    if inputs.shape[0] > settings.max_train_samples:
        selected = generator.choice(inputs.shape[0], settings.max_train_samples, replace=False)
        selected.sort()
        inputs = inputs[selected]
        targets = targets[selected]

    model = build_model(config, spec.lookback, spec.horizon)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    validation_inputs = torch.from_numpy(validation.scaled_windows)
    validation_targets = torch.from_numpy(validation.scaled_targets)

    best_loss = float("inf")
    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    patience = 0
    history: dict[str, float] = {}
    sample_count = inputs.shape[0]

    for epoch in range(settings.epochs):
        model.train()
        order = generator.permutation(sample_count)
        total = 0.0
        batches = 0
        for start in range(0, sample_count, settings.batch_size):
            index = order[start : start + settings.batch_size]
            window = torch.from_numpy(inputs[index])
            target = torch.from_numpy(targets[index])
            prediction = model(window)
            loss = torch.mean(torch.abs(prediction - target))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.item())
            batches += 1
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                torch.mean(torch.abs(model(validation_inputs) - validation_targets)).item()
            )
        history[f"epoch_{epoch:03d}_train"] = total / max(batches, 1)
        history[f"epoch_{epoch:03d}_validation"] = validation_loss
        if progress:
            print(
                f"    epoch {epoch:3d} train {total / max(batches, 1):.5f} validation {validation_loss:.5f}",
                flush=True,
            )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= settings.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    history["best_validation"] = best_loss
    history["train_samples"] = float(sample_count)
    return NhitsForecaster(model=model, lookback=spec.lookback, horizon=spec.horizon), history
