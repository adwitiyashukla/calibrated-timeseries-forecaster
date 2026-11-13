from __future__ import annotations

import json
import time

import numpy as np

from forecaster.config import Config
from forecaster.data.panel import load_panel
from forecaster.evaluation.paths import Artifacts
from forecaster.serving.predictor import FrequencyPredictor


def run(config: Config) -> dict[str, object]:
    artifacts = Artifacts(config)
    summary: dict[str, object] = {}
    for frequency in config.run.frequencies:
        predictor = FrequencyPredictor(config, frequency)
        panel = load_panel(config, frequency)
        history = panel.series(0).astype(np.float32)
        for _ in range(config.serving.latency_warmup):
            predictor.predict(history)
        durations = np.empty(config.serving.latency_iterations, dtype=np.float64)
        for index in range(config.serving.latency_iterations):
            started = time.perf_counter()
            predictor.predict(history)
            durations[index] = (time.perf_counter() - started) * 1000.0
        summary[frequency] = {
            "iterations": int(durations.size),
            "p50_ms": round(float(np.percentile(durations, 50)), 3),
            "p90_ms": round(float(np.percentile(durations, 90)), 3),
            "p99_ms": round(float(np.percentile(durations, 99)), 3),
            "mean_ms": round(float(durations.mean()), 3),
        }
    artifacts.ensure(artifacts.results("latency")).write_text(json.dumps(summary, indent=2))
    return summary
