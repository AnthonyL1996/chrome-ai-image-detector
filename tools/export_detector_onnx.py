#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poidh_detector.onnx_export import export_detector_onnx  # noqa: E402


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parse_arguments(arguments)
    export_detector_onnx(
        checkpoint=parsed.checkpoint,
        calibrator=parsed.calibrator,
        training_config=parsed.training_config,
        dataset_manifest=parsed.dataset_manifest,
        code_provenance=parsed.code_provenance,
        license_policy=parsed.license_policy,
        output=parsed.output,
    )
    return 0


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the frozen ConvNeXtV2-Nano detector to ONNX opset 18"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--code-provenance", type=Path, required=True)
    parser.add_argument("--license-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
