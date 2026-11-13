from __future__ import annotations

from pathlib import Path

from forecaster.config import Config


class Artifacts:
    def __init__(self, config: Config) -> None:
        self.root = config.run.artifact_dir

    def baselines(self, frequency: str) -> Path:
        return self.root / "baselines" / f"{frequency}.npz"

    def model_directory(self, frequency: str, name: str) -> Path:
        return self.root / "models" / frequency / name

    def blend(self, frequency: str) -> Path:
        return self.root / "models" / frequency / "blend.json"

    def training_report(self, frequency: str) -> Path:
        return self.root / "models" / frequency / "training.json"

    def onnx(self, frequency: str) -> Path:
        return self.root / "models" / frequency / "nhits.onnx"

    def forecasts(self, frequency: str) -> Path:
        return self.root / "forecasts" / f"{frequency}.npz"

    def conformal(self, frequency: str) -> Path:
        return self.root / "conformal" / frequency

    def results(self, name: str) -> Path:
        return self.root / "results" / f"{name}.json"

    def ensure(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
