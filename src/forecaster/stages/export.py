from __future__ import annotations

import numpy as np
import onnxruntime
import torch

from forecaster.config import Config
from forecaster.evaluation.paths import Artifacts
from forecaster.models.nhits import NhitsForecaster

TOLERANCE = 1e-4


def run(config: Config) -> dict[str, object]:
    artifacts = Artifacts(config)
    summary: dict[str, object] = {}
    for frequency in config.run.frequencies:
        settings = config.frequency[frequency]
        forecaster = NhitsForecaster.load(config, artifacts.model_directory(frequency, "nhits"))
        forecaster.model.eval()
        example = torch.randn(4, settings.lookback)
        destination = artifacts.ensure(artifacts.onnx(frequency))
        torch.onnx.export(
            forecaster.model,
            (example,),
            str(destination),
            input_names=["window"],
            output_names=["forecast"],
            dynamic_shapes={"window": {0: torch.export.Dim.DYNAMIC}},
            opset_version=config.serving.onnx_opset,
        )
        session = onnxruntime.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
        probe = np.random.default_rng(config.run.seed).standard_normal(
            (7, settings.lookback)
        ).astype(np.float32)
        with torch.no_grad():
            reference = forecaster.model(torch.from_numpy(probe)).numpy()
        produced = session.run(["forecast"], {"window": probe})[0]
        deviation = float(np.max(np.abs(reference - produced)))
        if deviation > TOLERANCE:
            raise ValueError(f"{frequency} onnx output deviates by {deviation}")
        summary[frequency] = {
            "bytes": destination.stat().st_size,
            "max_absolute_deviation": deviation,
        }
    return summary
