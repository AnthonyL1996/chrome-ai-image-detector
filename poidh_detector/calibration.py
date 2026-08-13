from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import ClassVar

from poidh_detector.training import TrainingConfig


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CalibrationClassCounts:
    real: int
    ai: int

    def __post_init__(self) -> None:
        for field_name, value in (("real", self.real), ("ai", self.ai)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} count must be a positive integer")

    @property
    def total(self) -> int:
        return self.real + self.ai


@dataclass(frozen=True, slots=True)
class PlattCalibrationArtifact:
    scale: float
    bias: float
    checkpoint_sha256: str
    calibration_split_sha256: str
    training_config: TrainingConfig
    input_identifier: str
    sample_count: int
    class_counts: CalibrationClassCounts

    schema_version: ClassVar[int] = 1
    method: ClassVar[str] = "platt_on_raw_logit"
    threshold: ClassVar[float] = 0.65

    def __post_init__(self) -> None:
        scale = _require_finite_number(self.scale, "scale")
        if scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        bias = _require_finite_number(self.bias, "bias")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "bias", bias)

        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _require_sha256(self.calibration_split_sha256, "calibration_split_sha256")
        if not isinstance(self.training_config, TrainingConfig):
            raise TypeError("training_config must be TrainingConfig")
        if (
            self.calibration_split_sha256
            != self.training_config.calibration_split_sha256
        ):
            raise ValueError("calibration split digest mismatch")
        if self.calibration_split_sha256 in self.training_config.exposed_holdout_sha256:
            raise ValueError("calibration split overlaps an exposed holdout")
        if self.input_identifier != "calibration":
            raise ValueError("input_identifier must identify the calibration split")
        if not isinstance(self.class_counts, CalibrationClassCounts):
            raise TypeError("class_counts must be CalibrationClassCounts")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count <= 0
            or self.sample_count != self.class_counts.total
        ):
            raise ValueError("sample_count must equal the sum of real and AI counts")

    def probability_ai(self, raw_logit: float) -> float:
        logit = _require_finite_number(raw_logit, "raw_logit")
        calibrated_logit = self.scale * logit + self.bias
        if calibrated_logit >= 0.0:
            return 1.0 / (1.0 + math.exp(-calibrated_logit))
        exponential = math.exp(calibrated_logit)
        return exponential / (1.0 + exponential)

    def to_json_bytes(self) -> bytes:
        document = {
            "schema_version": self.schema_version,
            "method": self.method,
            "scale": self.scale,
            "bias": self.bias,
            "checkpoint_sha256": self.checkpoint_sha256,
            "calibration_split_sha256": self.calibration_split_sha256,
            "training_config_sha256": self.training_config.sha256,
            "input_identifier": self.input_identifier,
            "threshold": self.threshold,
            "sample_count": self.sample_count,
            "class_counts": {
                "real": self.class_counts.real,
                "ai": self.class_counts.ai,
            },
        }
        return (
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()


def _require_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
