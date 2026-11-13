from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence

from forecaster.config import Config, ConfigError, load_config
from forecaster.stages import (
    backtest,
    baselines,
    benchmark,
    blend,
    calibrate,
    evaluate,
    export,
    prepare,
    train,
)

Stage = Callable[[Config], dict[str, object]]

STAGES: dict[str, Stage] = {
    "prepare": lambda config: prepare.run(config),
    "baselines": lambda config: baselines.run(config),
    "train": lambda config: train.run(config),
    "blend": lambda config: blend.run(config),
    "calibrate": lambda config: calibrate.run(config),
    "backtest": lambda config: backtest.run(config),
    "evaluate": lambda config: evaluate.run(config),
    "export": lambda config: export.run(config),
    "benchmark": lambda config: benchmark.run(config),
}

PIPELINE: tuple[str, ...] = (
    "prepare",
    "baselines",
    "train",
    "blend",
    "calibrate",
    "backtest",
    "export",
    "evaluate",
    "benchmark",
)


def _run_stage(name: str, config: Config) -> dict[str, object]:
    started = time.time()
    print(f"[{name}] started", flush=True)
    result = STAGES[name](config)
    print(f"[{name}] finished in {time.time() - started:.1f}s", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _serve(config: Config) -> None:
    import uvicorn

    from forecaster.serving.app import create_app

    uvicorn.run(create_app(config), host=config.serving.host, port=config.serving.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecaster")
    parser.add_argument("--config", default="config.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in STAGES:
        subparsers.add_parser(name)
    subparsers.add_parser("all")
    subparsers.add_parser("serve")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    if arguments.command == "serve":
        _serve(config)
        return 0
    names = PIPELINE if arguments.command == "all" else (arguments.command,)
    for name in names:
        _run_stage(name, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
