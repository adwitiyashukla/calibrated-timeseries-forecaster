from __future__ import annotations

from forecaster.config import Config
from forecaster.data.download import download, missing_files
from forecaster.data.panel import build_panel, panel_exists


def run(config: Config, force: bool = False) -> dict[str, object]:
    fetched = download(config, force=force)
    outstanding = missing_files(config)
    if outstanding:
        raise FileNotFoundError(f"missing raw files: {', '.join(outstanding)}")
    summary: dict[str, object] = {"downloaded": fetched}
    panels: dict[str, object] = {}
    for frequency in config.run.frequencies:
        if panel_exists(config, frequency) and not force:
            panels[frequency] = "cached"
            continue
        panels[frequency] = build_panel(config, frequency)
    summary["panels"] = panels
    return summary
