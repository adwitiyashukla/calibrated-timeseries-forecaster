from __future__ import annotations

import urllib.request
from pathlib import Path

from forecaster.config import Config

TRAIN_SUFFIX = "-train.csv"
TEST_SUFFIX = "-test.csv"
INFO_FILE = "M4-info.csv"


def _fetch(url: str, destination: Path, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "calibrated-timeseries-forecaster"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    destination.write_bytes(payload)


def required_files(config: Config) -> list[str]:
    names = [INFO_FILE]
    for frequency in config.run.frequencies:
        names.append(frequency + TRAIN_SUFFIX)
        names.append(frequency + TEST_SUFFIX)
    return names


def missing_files(config: Config) -> list[str]:
    return [name for name in required_files(config) if not (config.data.raw_dir / name).is_file()]


def download(config: Config, force: bool = False) -> list[str]:
    fetched = []
    timeout = config.data.download_timeout_seconds
    base = config.data.source_base_url.rstrip("/")
    for name in required_files(config):
        destination = config.data.raw_dir / name
        if destination.is_file() and not force:
            continue
        if name == INFO_FILE:
            url = f"{base}/{name}"
        elif name.endswith(TRAIN_SUFFIX):
            url = f"{base}/Train/{name}"
        else:
            url = f"{base}/Test/{name}"
        _fetch(url, destination, timeout)
        fetched.append(name)
    return fetched
