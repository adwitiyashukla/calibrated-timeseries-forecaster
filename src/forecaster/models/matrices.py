from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecaster.data.panel import SeriesPanel
from forecaster.features.spec import FeatureSpec, assemble_targets, assemble_windows
from forecaster.models.windows import OriginSet


@dataclass(frozen=True)
class WindowBatch:
    features: np.ndarray
    scaled_windows: np.ndarray
    scaled_targets: np.ndarray
    scale: np.ndarray
    series_index: np.ndarray
    origins: np.ndarray


def build_batch(
    panel: SeriesPanel, spec: FeatureSpec, origin_set: OriginSet, with_targets: bool
) -> WindowBatch:
    windows, context_lengths = assemble_windows(
        panel, origin_set.series_index, origin_set.origins, spec.lookback
    )
    features = spec.transform(windows, context_lengths, origin_set.origins)
    scaled, scale = spec.scaled_windows(windows)
    if with_targets:
        targets = assemble_targets(panel, origin_set.series_index, origin_set.origins, spec.horizon)
        scaled_targets = targets / scale[:, None]
    else:
        scaled_targets = np.empty((origin_set.size, 0), dtype=np.float32)
    return WindowBatch(
        features=features,
        scaled_windows=scaled.astype(np.float32),
        scaled_targets=scaled_targets.astype(np.float32),
        scale=scale,
        series_index=origin_set.series_index,
        origins=origin_set.origins,
    )
