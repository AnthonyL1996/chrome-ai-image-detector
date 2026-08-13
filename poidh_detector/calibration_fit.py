from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import ClassVar

from poidh_detector.calibration import (
    CalibrationClassCounts,
    PlattCalibrationArtifact,
)
from poidh_detector.data import SplitManifest
from poidh_detector.training import TrainingConfig

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_THRESHOLD = 0.65
_ECE_BINS = 15
_REGULARIZATION = 1e-8
_MAX_ITERATIONS = 100
_MIN_SCALE = 1e-12
_GRADIENT_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class CalibrationPrediction:
    sample_id: str
    raw_logit: float
    label: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sample_id, str)
            or not self.sample_id
            or self.sample_id != self.sample_id.strip()
        ):
            raise ValueError("sample_id must be non-empty and trimmed")
        logit = _finite_number(self.raw_logit, "raw_logit")
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError("label must be 0 (real) or 1 (AI)")
        object.__setattr__(self, "raw_logit", logit)


@dataclass(frozen=True, slots=True)
class CalibrationPredictions:
    schema_version: int
    input_identifier: str
    checkpoint_sha256: str
    calibration_split_sha256: str
    training_config_sha256: str
    predictions: tuple[CalibrationPrediction, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported calibration predictions schema version")
        if self.input_identifier != "calibration":
            raise ValueError("input_identifier must identify the calibration split")
        for field_name in (
            "checkpoint_sha256",
            "calibration_split_sha256",
            "training_config_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if type(self.predictions) is not tuple or not self.predictions:
            raise ValueError("predictions must be a non-empty tuple")
        if any(type(row) is not CalibrationPrediction for row in self.predictions):
            raise TypeError("predictions must contain CalibrationPrediction rows")
        sorted_rows = tuple(sorted(self.predictions, key=lambda row: row.sample_id))
        identifiers = [row.sample_id for row in sorted_rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate sample ID in calibration predictions")
        object.__setattr__(self, "predictions", sorted_rows)

    def to_json_bytes(self) -> bytes:
        document = {
            "calibration_split_sha256": self.calibration_split_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "input_identifier": self.input_identifier,
            "predictions": [
                {
                    "label": row.label,
                    "raw_logit": row.raw_logit,
                    "sample_id": row.sample_id,
                }
                for row in self.predictions
            ],
            "schema_version": self.schema_version,
            "training_config_sha256": self.training_config_sha256,
        }
        return _canonical_json(document)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> CalibrationPredictions:
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid calibration predictions JSON") from error
        if not isinstance(document, dict) or set(document) != {
            "calibration_split_sha256",
            "checkpoint_sha256",
            "input_identifier",
            "predictions",
            "schema_version",
            "training_config_sha256",
        }:
            raise ValueError("calibration predictions fields do not match schema")
        raw_rows = document["predictions"]
        if not isinstance(raw_rows, list):
            raise ValueError("predictions must be a JSON array")
        rows: list[CalibrationPrediction] = []
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict) or set(raw_row) != {
                "label",
                "raw_logit",
                "sample_id",
            }:
                raise ValueError("calibration prediction fields do not match schema")
            rows.append(
                CalibrationPrediction(
                    sample_id=raw_row["sample_id"],
                    raw_logit=raw_row["raw_logit"],
                    label=raw_row["label"],
                )
            )
        result = cls(
            schema_version=document["schema_version"],
            input_identifier=document["input_identifier"],
            checkpoint_sha256=document["checkpoint_sha256"],
            calibration_split_sha256=document["calibration_split_sha256"],
            training_config_sha256=document["training_config_sha256"],
            predictions=tuple(rows),
        )
        if result.to_json_bytes() != payload:
            raise ValueError("calibration predictions JSON must be canonical")
        return result


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    bce: float
    ece: float
    accuracy_at_threshold: float

    def __post_init__(self) -> None:
        bce = _finite_number(self.bce, "bce")
        ece = _finite_number(self.ece, "ece")
        accuracy = _finite_number(self.accuracy_at_threshold, "accuracy_at_threshold")
        if bce < 0.0:
            raise ValueError("bce must be non-negative")
        if not 0.0 <= ece <= 1.0:
            raise ValueError("ece must be between 0 and 1")
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("accuracy_at_threshold must be between 0 and 1")
        object.__setattr__(self, "bce", bce)
        object.__setattr__(self, "ece", ece)
        object.__setattr__(self, "accuracy_at_threshold", accuracy)


@dataclass(frozen=True, slots=True)
class CalibrationFitResult:
    artifact: PlattCalibrationArtifact
    predictions_sha256: str
    uncalibrated: CalibrationMetrics
    calibrated: CalibrationMetrics

    schema_version: ClassVar[int] = 1
    threshold: ClassVar[float] = _THRESHOLD
    ece_bins: ClassVar[int] = _ECE_BINS

    def __post_init__(self) -> None:
        if type(self.artifact) is not PlattCalibrationArtifact:
            raise TypeError("artifact must be a PlattCalibrationArtifact")
        _require_sha256(self.predictions_sha256, "predictions_sha256")
        if (
            type(self.uncalibrated) is not CalibrationMetrics
            or type(self.calibrated) is not CalibrationMetrics
        ):
            raise TypeError("fit metrics must be CalibrationMetrics")

    def to_json_bytes(self) -> bytes:
        return _canonical_json(
            {
                "artifact": json.loads(self.artifact.to_json_bytes()),
                "calibrated": _metrics_document(self.calibrated),
                "ece_bins": self.ece_bins,
                "predictions_sha256": self.predictions_sha256,
                "schema_version": self.schema_version,
                "threshold": self.threshold,
                "uncalibrated": _metrics_document(self.uncalibrated),
            }
        )


def fit_platt_calibration(
    predictions: CalibrationPredictions,
    *,
    training_config: TrainingConfig,
    checkpoint_sha256: str,
    split_manifest: SplitManifest,
) -> CalibrationFitResult:
    if type(predictions) is not CalibrationPredictions:
        raise TypeError("predictions must be CalibrationPredictions")
    if type(training_config) is not TrainingConfig:
        raise TypeError("training_config must be TrainingConfig")
    if type(split_manifest) is not SplitManifest:
        raise TypeError("split_manifest must be SplitManifest")
    _require_sha256(checkpoint_sha256, "checkpoint_sha256")
    if predictions.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("checkpoint digest mismatch")
    if predictions.training_config_sha256 != training_config.sha256:
        raise ValueError("training config digest mismatch")
    if predictions.calibration_split_sha256 != training_config.calibration_split_sha256:
        raise ValueError("calibration split digest mismatch")
    if predictions.calibration_split_sha256 in training_config.exposed_holdout_sha256:
        raise ValueError("calibration split overlaps an exposed holdout")
    _validate_calibration_assignments(predictions, training_config, split_manifest)

    labels = [row.label for row in predictions.predictions]
    if set(labels) != {0, 1}:
        raise ValueError("calibration fitting requires both real and AI samples")
    counts = CalibrationClassCounts(real=labels.count(0), ai=labels.count(1))
    logits = [row.raw_logit for row in predictions.predictions]
    scale, bias = _fit_parameters(logits, labels)
    artifact = PlattCalibrationArtifact(
        scale=scale,
        bias=bias,
        checkpoint_sha256=checkpoint_sha256,
        calibration_split_sha256=predictions.calibration_split_sha256,
        training_config=training_config,
        input_identifier=predictions.input_identifier,
        sample_count=len(labels),
        class_counts=counts,
    )
    uncalibrated_probabilities = [_sigmoid(logit) for logit in logits]
    calibrated_logits = [scale * logit + bias for logit in logits]
    calibrated_probabilities = [_sigmoid(logit) for logit in calibrated_logits]
    return CalibrationFitResult(
        artifact=artifact,
        predictions_sha256=predictions.sha256,
        uncalibrated=_metrics(uncalibrated_probabilities, logits, labels),
        calibrated=_metrics(calibrated_probabilities, calibrated_logits, labels),
    )


def _validate_calibration_assignments(
    predictions: CalibrationPredictions,
    training_config: TrainingConfig,
    split_manifest: SplitManifest,
) -> None:
    if split_manifest.sha256 != training_config.split_manifest_sha256:
        raise ValueError("split manifest digest mismatch")
    if (
        split_manifest.dataset_manifest_sha256
        != training_config.dataset_manifest_sha256
    ):
        raise ValueError("split manifest dataset digest mismatch")
    for row in predictions.predictions:
        assignment = split_manifest.assignments.get(row.sample_id)
        if assignment is None:
            raise ValueError(
                f"prediction sample ID not present in split manifest: {row.sample_id}"
            )
        if assignment != "calibration":
            raise ValueError(
                f"prediction sample is assigned to {assignment}, not calibration: "
                f"{row.sample_id}"
            )


def _fit_parameters(logits: list[float], labels: list[int]) -> tuple[float, float]:
    if set(labels) != {0, 1}:
        raise ValueError("calibration fitting requires both real and AI samples")
    normalization = max(1.0, *(abs(logit) for logit in logits))
    normalized = [logit / normalization for logit in logits]
    scale = 1.0
    bias = 0.0
    objective = _objective(normalized, labels, scale, bias)

    for _ in range(_MAX_ITERATIONS):
        gradient_scale = _REGULARIZATION * scale
        gradient_bias = _REGULARIZATION * bias
        hessian_scale = _REGULARIZATION
        hessian_cross = 0.0
        hessian_bias = _REGULARIZATION
        sample_weight = 1.0 / len(labels)
        for logit, label in zip(normalized, labels, strict=True):
            probability = _sigmoid(scale * logit + bias)
            error = probability - label
            curvature = probability * (1.0 - probability)
            gradient_scale += sample_weight * error * logit
            gradient_bias += sample_weight * error
            hessian_scale += sample_weight * curvature * logit * logit
            hessian_cross += sample_weight * curvature * logit
            hessian_bias += sample_weight * curvature
        if max(abs(gradient_scale), abs(gradient_bias)) <= _GRADIENT_TOLERANCE:
            break
        determinant = hessian_scale * hessian_bias - hessian_cross**2
        if not math.isfinite(determinant) or determinant <= 0.0:
            raise RuntimeError("calibration optimizer Hessian is not positive definite")
        step_scale = (
            hessian_bias * gradient_scale - hessian_cross * gradient_bias
        ) / determinant
        step_bias = (
            hessian_scale * gradient_bias - hessian_cross * gradient_scale
        ) / determinant

        step_fraction = 1.0
        accepted = False
        for _ in range(60):
            candidate_scale = scale - step_fraction * step_scale
            candidate_bias = bias - step_fraction * step_bias
            if candidate_scale >= _MIN_SCALE:
                candidate_objective = _objective(
                    normalized, labels, candidate_scale, candidate_bias
                )
                if candidate_objective < objective:
                    scale = candidate_scale
                    bias = candidate_bias
                    objective = candidate_objective
                    accepted = True
                    break
            step_fraction *= 0.5
        if not accepted:
            break

    raw_scale = scale / normalization
    if not math.isfinite(raw_scale) or raw_scale <= 0.0 or not math.isfinite(bias):
        raise RuntimeError("calibration optimizer produced invalid parameters")
    return raw_scale, bias


def _objective(
    logits: list[float], labels: list[int], scale: float, bias: float
) -> float:
    loss = sum(
        _binary_cross_entropy_from_logit(scale * logit + bias, label)
        for logit, label in zip(logits, labels, strict=True)
    ) / len(labels)
    return loss + 0.5 * _REGULARIZATION * (scale * scale + bias * bias)


def _metrics(
    probabilities: list[float], logits: list[float], labels: list[int]
) -> CalibrationMetrics:
    bce = sum(
        _binary_cross_entropy_from_logit(logit, label)
        for logit, label in zip(logits, labels, strict=True)
    ) / len(labels)
    accuracy = sum(
        (probability >= _THRESHOLD) == bool(label)
        for probability, label in zip(probabilities, labels, strict=True)
    ) / len(labels)
    ece = 0.0
    for bin_index in range(_ECE_BINS):
        lower = bin_index / _ECE_BINS
        upper = (bin_index + 1) / _ECE_BINS
        members = [
            (probability, label)
            for probability, label in zip(probabilities, labels, strict=True)
            if lower <= probability < upper
            or (bin_index == _ECE_BINS - 1 and probability == 1.0)
        ]
        if members:
            confidence = sum(row[0] for row in members) / len(members)
            prevalence = sum(row[1] for row in members) / len(members)
            ece += len(members) / len(labels) * abs(confidence - prevalence)
    return CalibrationMetrics(bce=bce, ece=ece, accuracy_at_threshold=accuracy)


def _binary_cross_entropy_from_logit(logit: float, label: int) -> float:
    return max(logit, 0.0) - logit * label + math.log1p(math.exp(-abs(logit)))


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _metrics_document(metrics: CalibrationMetrics) -> dict[str, float]:
    return {
        "accuracy_at_threshold": metrics.accuracy_at_threshold,
        "bce": metrics.bce,
        "ece": metrics.ece,
    }


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _canonical_json(document: object) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
