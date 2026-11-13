from __future__ import annotations

import json
import time

import numpy as np

from forecaster.config import Config
from forecaster.data.panel import load_panel
from forecaster.evaluation.paths import Artifacts
from forecaster.features.spec import FeatureSpec
from forecaster.models.lgbm import train_lgbm
from forecaster.models.matrices import build_batch
from forecaster.models.nhits import train_nhits
from forecaster.models.windows import OriginSet, build_window_plan


def run(config: Config, progress: bool = True) -> dict[str, object]:
    artifacts = Artifacts(config)
    summary: dict[str, object] = {}
    for frequency in config.run.frequencies:
        settings = config.frequency[frequency]
        panel = load_panel(config, frequency)
        spec = FeatureSpec.from_config(config, settings)
        plan = build_window_plan(config, settings, panel)
        if progress:
            print(
                f"  {frequency}: train {plan.train.size} validation {plan.validation.size}",
                flush=True,
            )
        train_batch = build_batch(panel, spec, plan.train, True)
        validation_batch = build_batch(panel, spec, plan.validation, True)

        started = time.time()
        lgbm, rounds = train_lgbm(config, spec, train_batch, validation_batch)
        lgbm_seconds = time.time() - started
        lgbm.save(artifacts.model_directory(frequency, "lgbm"))
        if progress:
            print(f"  {frequency}: lightgbm {lgbm_seconds:.0f}s", flush=True)

        started = time.time()
        nhits, history = train_nhits(config, spec, train_batch, validation_batch, progress=progress)
        nhits_seconds = time.time() - started
        nhits.save(artifacts.model_directory(frequency, "nhits"))
        if progress:
            print(f"  {frequency}: nhits {nhits_seconds:.0f}s", flush=True)

        test_set = OriginSet(np.arange(panel.size, dtype=np.int32), plan.test_origins)
        test_batch = build_batch(panel, spec, test_set, False)
        np.savez_compressed(
            artifacts.ensure(artifacts.forecasts(frequency)),
            lgbm=lgbm.predict(test_batch),
            nhits=nhits.predict(test_batch),
            blend=np.zeros_like(test_batch.scale)[:, None],
            scale=test_batch.scale,
        )
        artifacts.ensure(artifacts.training_report(frequency)).write_text(
            json.dumps(
                {
                    "train_windows": int(plan.train.size),
                    "validation_windows": int(plan.validation.size),
                    "calibration_windows": int(plan.calibration.size),
                    "feature_width": spec.width(),
                    "lgbm_seconds": lgbm_seconds,
                    "lgbm_rounds": rounds,
                    "nhits_seconds": nhits_seconds,
                    "nhits_history": history,
                },
                indent=2,
            )
        )
        summary[frequency] = {
            "lgbm_seconds": round(lgbm_seconds, 1),
            "nhits_seconds": round(nhits_seconds, 1),
        }
    return summary
