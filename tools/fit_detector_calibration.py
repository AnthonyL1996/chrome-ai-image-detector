from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poidh_detector.calibration_fit import (
    CalibrationPredictions,
    fit_platt_calibration,
)
from poidh_detector.data import SplitManifest
from poidh_detector.training import TrainingConfig


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parse_arguments(arguments)
    destination = parsed.output.absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"calibration output already exists: {destination}")
    predictions = CalibrationPredictions.from_json_bytes(
        parsed.predictions.read_bytes()
    )
    training_config = _load_training_config(parsed.training_config)
    split_manifest = _load_split_manifest(parsed.split_manifest)
    checkpoint_sha256 = _sha256_file(parsed.checkpoint)
    result = fit_platt_calibration(
        predictions,
        training_config=training_config,
        checkpoint_sha256=checkpoint_sha256,
        split_manifest=split_manifest,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        output.write(result.to_json_bytes())
    return 0


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit frozen Platt calibration from checkpoint logits"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def _load_training_config(path: Path) -> TrainingConfig:
    payload = path.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid training config JSON") from error
    if not isinstance(document, dict):
        raise ValueError("training config must be a JSON object")
    expected_fields = {
        "architecture",
        "calibration_split_sha256",
        "dataset_manifest_sha256",
        "exposed_holdout_sha256",
        "pretrained",
        "schema_version",
        "seed",
        "selection_metric",
        "selection_minimize",
        "split_manifest_sha256",
        "weights_origin",
    }
    if set(document) != expected_fields:
        raise ValueError("training config fields do not match schema")
    holdouts = document["exposed_holdout_sha256"]
    if not isinstance(holdouts, list):
        raise ValueError("exposed_holdout_sha256 must be a JSON array")
    config = TrainingConfig(
        dataset_manifest_sha256=document["dataset_manifest_sha256"],
        split_manifest_sha256=document["split_manifest_sha256"],
        calibration_split_sha256=document["calibration_split_sha256"],
        exposed_holdout_sha256=tuple(holdouts),
        seed=document["seed"],
        selection_metric=document["selection_metric"],
    )
    if config.to_json_bytes() != payload:
        raise ValueError("training config JSON must be canonical and frozen")
    return config


def _load_split_manifest(path: Path) -> SplitManifest:
    payload = path.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid split manifest JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "assignments",
        "dataset_manifest_sha256",
        "ratios",
        "schema_version",
        "seed",
    }:
        raise ValueError("split manifest fields do not match schema")
    split_manifest = SplitManifest(**document)
    if split_manifest.to_json_bytes() != payload:
        raise ValueError("split manifest JSON must be canonical")
    return split_manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
