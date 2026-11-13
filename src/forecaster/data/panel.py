from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from forecaster.config import Config


@dataclass(frozen=True)
class SeriesPanel:
    frequency: str
    series_ids: tuple[str, ...]
    values: np.ndarray
    offsets: np.ndarray

    @property
    def size(self) -> int:
        return len(self.series_ids)

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def series(self, index: int) -> np.ndarray:
        return self.values[self.offsets[index] : self.offsets[index + 1]]


def _read_wide(path: Path) -> tuple[list[str], list[np.ndarray]]:
    identifiers: list[str] = []
    rows: list[np.ndarray] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            values = []
            for cell in row[1:]:
                text = cell.strip()
                if not text:
                    break
                values.append(float(text))
            identifiers.append(row[0])
            rows.append(np.asarray(values, dtype=np.float32))
    return identifiers, rows


def _panel_directory(config: Config, frequency: str) -> Path:
    return config.data.panel_dir / frequency


def build_panel(config: Config, frequency: str) -> dict[str, int]:
    raw = config.data.raw_dir
    train_ids, train_rows = _read_wide(raw / f"{frequency}-train.csv")
    test_ids, test_rows = _read_wide(raw / f"{frequency}-test.csv")
    if train_ids != test_ids:
        raise ValueError(f"{frequency} train and test series identifiers do not match")

    horizon = config.frequency[frequency].horizon
    for identifier, row in zip(test_ids, test_rows, strict=True):
        if row.size != horizon:
            raise ValueError(f"{identifier} test length {row.size} does not match horizon {horizon}")

    directory = _panel_directory(config, frequency)
    directory.mkdir(parents=True, exist_ok=True)

    index_frame = pl.DataFrame(
        {
            "series_index": np.arange(len(train_ids), dtype=np.int32),
            "series_id": train_ids,
            "train_length": np.asarray([row.size for row in train_rows], dtype=np.int32),
        }
    )
    index_frame.write_parquet(directory / "index.parquet")

    for label, rows in (("train", train_rows), ("test", test_rows)):
        counts = np.asarray([row.size for row in rows], dtype=np.int64)
        series_index = np.repeat(np.arange(len(rows), dtype=np.int32), counts)
        step = np.concatenate([np.arange(count, dtype=np.int32) for count in counts])
        value = np.concatenate(rows).astype(np.float32)
        pl.DataFrame(
            {"series_index": series_index, "step": step, "value": value}
        ).write_parquet(directory / f"{label}.parquet")

    return {
        "series": len(train_ids),
        "train_observations": int(sum(row.size for row in train_rows)),
        "test_observations": int(sum(row.size for row in test_rows)),
    }


def build_panels(config: Config) -> dict[str, dict[str, int]]:
    return {name: build_panel(config, name) for name in config.run.frequencies}


def panel_exists(config: Config, frequency: str) -> bool:
    directory = _panel_directory(config, frequency)
    return all((directory / name).is_file() for name in ("index.parquet", "train.parquet", "test.parquet"))


def load_panel(config: Config, frequency: str) -> SeriesPanel:
    directory = _panel_directory(config, frequency)
    index_frame = pl.read_parquet(directory / "index.parquet").sort("series_index")
    frame = pl.read_parquet(directory / "train.parquet").sort(["series_index", "step"])
    lengths = index_frame["train_length"].to_numpy().astype(np.int64)
    offsets = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    values = frame["value"].to_numpy().astype(np.float32, copy=False)
    if values.size != offsets[-1]:
        raise ValueError(f"{frequency} panel is inconsistent with its index")
    return SeriesPanel(
        frequency=frequency,
        series_ids=tuple(index_frame["series_id"].to_list()),
        values=values,
        offsets=offsets,
    )


def load_test(config: Config, frequency: str) -> np.ndarray:
    directory = _panel_directory(config, frequency)
    frame = pl.read_parquet(directory / "test.parquet").sort(["series_index", "step"])
    horizon = config.frequency[frequency].horizon
    values = frame["value"].to_numpy().astype(np.float32, copy=False)
    if values.size % horizon != 0:
        raise ValueError(f"{frequency} test panel is not a multiple of the horizon")
    return values.reshape(-1, horizon)
