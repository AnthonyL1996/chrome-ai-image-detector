from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from collections.abc import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poidh_detector.inference import run_inference


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parse_arguments(arguments)
    destination = parsed.output.absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"prediction output already exists: {destination}")
    result = run_inference(
        parsed.dataset_root,
        parsed.run,
        partition=parsed.partition,
        batch_size=parsed.batch_size,
        workers=parsed.workers,
        device=parsed.device,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        output.write(result.predictions_json_bytes())
    sys.stdout.buffer.write(result.summary_json_bytes())
    return 0


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a selected immutable detector checkpoint on a frozen split."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--partition", choices=("calibration", "validation"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
