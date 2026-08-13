from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable


_EPSILON = 1e-6
BOUNTY_THRESHOLD = 0.65


@dataclass(frozen=True, slots=True)
class Prediction:
    sample_id: str
    label: int
    probability_ai: float
    source: str
    content_sha256: str | None = None
    provenance_group: str | None = None
    generator_family: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 (real) or 1 (AI)")
        if not math.isfinite(self.probability_ai) or not 0.0 <= self.probability_ai <= 1.0:
            raise ValueError("probability_ai must be finite and between 0 and 1")
        if not self.source:
            raise ValueError("source must not be empty")
        if self.content_sha256 is not None:
            digest = self.content_sha256.lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("content_sha256 must be a 64-character hexadecimal digest")
        for field_name, value in (
            ("provenance_group", self.provenance_group),
            ("generator_family", self.generator_family),
        ):
            if value is not None and not value:
                raise ValueError(f"{field_name} must not be empty when provided")


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    count: int
    tn: int
    fp: int
    fn: int
    tp: int
    true_negative_rate: float | None
    true_positive_rate: float | None
    balanced_accuracy: float | None


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    scale: float
    bias: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("calibrator scale must be finite and positive")
        if not math.isfinite(self.bias):
            raise ValueError("calibrator bias must be finite")

    def transform(self, probability_ai: float) -> float:
        if not math.isfinite(probability_ai) or not 0.0 <= probability_ai <= 1.0:
            raise ValueError("probability_ai must be finite and between 0 and 1")
        raw_logit = _logit(probability_ai)
        return _sigmoid(self.scale * raw_logit + self.bias)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    threshold: float
    calibrator: PlattCalibrator
    overall: ClassificationMetrics
    by_source: dict[str, ClassificationMetrics]


def balanced_accuracy(
    predictions: Iterable[Prediction], *, threshold: float = 0.65
) -> ClassificationMetrics:
    metrics = _classification_metrics(predictions, threshold=threshold)
    if metrics.true_negative_rate is None or metrics.true_positive_rate is None:
        raise ValueError("balanced accuracy requires both real and AI samples")
    return metrics


def fit_platt_calibrator(
    predictions: Iterable[Prediction],
    *,
    learning_rate: float = 0.03,
    iterations: int = 2_000,
) -> PlattCalibrator:
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    rows = list(predictions)
    _require_unique_sample_ids(rows, split_name="calibration")
    if not rows or {row.label for row in rows} != {0, 1}:
        raise ValueError("calibration requires both real and AI samples")

    features = [_logit(row.probability_ai) for row in rows]
    labels = [float(row.label) for row in rows]
    log_scale = 0.0
    bias = 0.0
    class_counts = {
        0: sum(row.label == 0 for row in rows),
        1: sum(row.label == 1 for row in rows),
    }

    # Optimizing log(scale) guarantees a monotonically increasing calibration
    # curve, so higher raw AI scores cannot become lower displayed confidence.
    for _ in range(iterations):
        scale = math.exp(log_scale)
        gradient_scale = 0.0
        gradient_bias = 0.0
        for feature, label in zip(features, labels, strict=True):
            error = _sigmoid(scale * feature + bias) - label
            class_weight = 0.5 / class_counts[int(label)]
            gradient_scale += class_weight * error * feature * scale
            gradient_bias += class_weight * error
        log_scale -= learning_rate * gradient_scale
        bias -= learning_rate * gradient_bias
        log_scale = min(max(log_scale, -6.0), 6.0)
        bias = min(max(bias, -20.0), 20.0)

    return PlattCalibrator(scale=math.exp(log_scale), bias=bias)


def evaluate_holdout(
    calibration: Iterable[Prediction],
    holdout: Iterable[Prediction],
    *,
    threshold: float = BOUNTY_THRESHOLD,
    require_generator_disjoint: bool = False,
) -> EvaluationReport:
    calibration_rows = list(calibration)
    holdout_rows = list(holdout)
    if threshold != BOUNTY_THRESHOLD:
        raise ValueError("bounty holdout threshold is fixed at 0.65")
    _require_unique_sample_ids(calibration_rows, split_name="calibration")
    _require_unique_sample_ids(holdout_rows, split_name="holdout")

    leaked_ids = {row.sample_id for row in calibration_rows} & {
        row.sample_id for row in holdout_rows
    }
    if leaked_ids:
        preview = ", ".join(sorted(leaked_ids)[:3])
        raise ValueError(f"sample leakage between calibration and holdout: {preview}")

    _reject_metadata_leakage(
        calibration_rows,
        holdout_rows,
        field_name="content_sha256",
        description="content leakage",
    )
    _reject_metadata_leakage(
        calibration_rows,
        holdout_rows,
        field_name="provenance_group",
        description="provenance leakage",
    )
    if require_generator_disjoint:
        _reject_metadata_leakage(
            calibration_rows,
            holdout_rows,
            field_name="generator_family",
            description="generator-family leakage",
        )

    calibrator = fit_platt_calibrator(calibration_rows)
    calibrated = [
        Prediction(
            sample_id=row.sample_id,
            label=row.label,
            probability_ai=calibrator.transform(row.probability_ai),
            source=row.source,
        )
        for row in holdout_rows
    ]
    overall = balanced_accuracy(calibrated, threshold=threshold)

    grouped: dict[str, list[Prediction]] = {}
    for row in calibrated:
        grouped.setdefault(row.source, []).append(row)
    by_source: dict[str, ClassificationMetrics] = {}
    for source, rows in sorted(grouped.items()):
        source_metrics = _classification_metrics(rows, threshold=threshold)
        if source_metrics.balanced_accuracy is None:
            if source_metrics.true_positive_rate is not None:
                paired_rate = overall.true_negative_rate
                score = (source_metrics.true_positive_rate + paired_rate) / 2.0
            else:
                paired_rate = overall.true_positive_rate
                score = (source_metrics.true_negative_rate + paired_rate) / 2.0
            source_metrics = replace(source_metrics, balanced_accuracy=score)
        by_source[source] = source_metrics
    return EvaluationReport(
        threshold=threshold,
        calibrator=calibrator,
        overall=overall,
        by_source=by_source,
    )


def _classification_metrics(
    predictions: Iterable[Prediction], *, threshold: float
) -> ClassificationMetrics:
    _validate_threshold(threshold)
    rows = list(predictions)
    if not rows:
        raise ValueError("predictions must not be empty")

    tn = fp = fn = tp = 0
    for row in rows:
        predicted_ai = row.probability_ai >= threshold
        if row.label == 1 and predicted_ai:
            tp += 1
        elif row.label == 1:
            fn += 1
        elif predicted_ai:
            fp += 1
        else:
            tn += 1

    real_count = tn + fp
    ai_count = tp + fn
    true_negative_rate = tn / real_count if real_count else None
    true_positive_rate = tp / ai_count if ai_count else None
    score = (
        (true_negative_rate + true_positive_rate) / 2.0
        if true_negative_rate is not None and true_positive_rate is not None
        else None
    )
    return ClassificationMetrics(
        count=len(rows),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        true_negative_rate=true_negative_rate,
        true_positive_rate=true_positive_rate,
        balanced_accuracy=score,
    )


def _require_unique_sample_ids(rows: list[Prediction], *, split_name: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        if row.sample_id in seen:
            duplicates.add(row.sample_id)
        seen.add(row.sample_id)
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:3])
        raise ValueError(f"duplicate sample IDs in {split_name}: {preview}")


def _reject_metadata_leakage(
    calibration_rows: list[Prediction],
    holdout_rows: list[Prediction],
    *,
    field_name: str,
    description: str,
) -> None:
    calibration_values = {
        value
        for row in calibration_rows
        if (value := getattr(row, field_name)) is not None
    }
    holdout_values = {
        value
        for row in holdout_rows
        if (value := getattr(row, field_name)) is not None
    }
    overlap = calibration_values & holdout_values
    if overlap:
        preview = ", ".join(sorted(overlap)[:3])
        raise ValueError(f"{description} between calibration and holdout: {preview}")


def _validate_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")


def _logit(probability: float) -> float:
    clipped = min(max(probability, _EPSILON), 1.0 - _EPSILON)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        denominator = 1.0 + math.exp(-value)
        return 1.0 / denominator
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
