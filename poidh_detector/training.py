from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from collections.abc import Iterable


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_SEED = 2**32 - 1


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    calibration_split_sha256: str
    exposed_holdout_sha256: tuple[str, ...]
    seed: int
    architecture: str = field(default="convnextv2_nano", init=False)
    weights_origin: str = field(default="random_initialization", init=False)
    pretrained: bool = field(default=False, init=False)
    selection_metric: str = field(default="validation_bce", init=False)
    selection_minimize: bool = field(default=True, init=False)
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        _require_sha256(self.split_manifest_sha256, "split_manifest_sha256")
        _require_sha256(self.calibration_split_sha256, "calibration_split_sha256")
        holdout_hashes = tuple(self.exposed_holdout_sha256)
        if not holdout_hashes:
            raise ValueError("at least one exposed holdout digest is required")
        for digest in holdout_hashes:
            _require_sha256(digest, "exposed_holdout_sha256")
        if len(holdout_hashes) != len(set(holdout_hashes)):
            raise ValueError("exposed holdout digests must be unique")
        object.__setattr__(
            self, "exposed_holdout_sha256", tuple(sorted(holdout_hashes))
        )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= _MAX_SEED
        ):
            raise ValueError(f"seed must be an integer from 0 to {_MAX_SEED}")

    def to_json_bytes(self) -> bytes:
        document = {
            "architecture": self.architecture,
            "calibration_split_sha256": self.calibration_split_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "exposed_holdout_sha256": list(self.exposed_holdout_sha256),
            "pretrained": self.pretrained,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "selection_metric": self.selection_metric,
            "selection_minimize": self.selection_minimize,
            "split_manifest_sha256": self.split_manifest_sha256,
            "weights_origin": self.weights_origin,
        }
        return (
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointCandidate:
    checkpoint_id: str
    epoch: int
    global_step: int
    validation_bce: float
    training_bce: float | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint_id or self.checkpoint_id != self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must be non-empty and trimmed")
        for field_name, value in (
            ("epoch", self.epoch),
            ("global_step", self.global_step),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        _require_loss(self.validation_bce, "validation_bce")
        if self.training_bce is not None:
            _require_loss(self.training_bce, "training_bce")


def select_best_checkpoint(
    candidates: Iterable[CheckpointCandidate],
) -> CheckpointCandidate:
    rows = list(candidates)
    if not rows:
        raise ValueError("checkpoint candidates must not be empty")
    identifiers = [row.checkpoint_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate checkpoint ID")
    return min(
        rows,
        key=lambda row: (
            row.validation_bce,
            row.epoch,
            row.global_step,
            row.checkpoint_id,
        ),
    )


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_loss(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
