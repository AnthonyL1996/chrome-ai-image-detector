from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class Prediction:
    sample_id: str
    label: int
    probability_ai: float
    source: str

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 (real) or 1 (AI)")
        if not math.isfinite(self.probability_ai) or not 0.0 <= self.probability_ai <= 1.0:
            raise ValueError("probability_ai must be finite and between 0 and 1")
        if not self.source:
            raise ValueError("source must not be empty")


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
    rows = list(predictions)
    _require_unique_sample_ids(rows, split_name="calibration")
    if not rows or {row.label for row in rows} != {0, 1}:
        raise ValueError("calibration requires both real and AI samples")

    features = [_logit(row.probability_ai) for row in rows]
    labels = [float(row.label) for row in rows]
    log_scale = 0.0
    bias = 0.0
    count = float(len(rows))

    # Optimizing log(scale) guarantees a monotonically increasing calibration
    # curve, so higher raw AI scores cannot become lower displayed confidence.
    for _ in range(iterations):
        scale = math.exp(log_scale)
        gradient_scale = 0.0
        gradient_bias = 0.0
        for feature, label in zip(features, labels, strict=True):
            error = _sigmoid(scale * feature + bias) - label
            gradient_scale += error * feature * scale
            gradient_bias += error
        log_scale -= learning_rate * gradient_scale / count
        bias -= learning_rate * gradient_bias / count
        log_scale = min(max(log_scale, -6.0), 6.0)
        bias = min(max(bias, -20.0), 20.0)

    return PlattCalibrator(scale=math.exp(log_scale), bias=bias)


def evaluate_holdout(
    calibration: Iterable[Prediction],
    holdout: Iterable[Prediction],
    *,
    threshold: float = 0.65,
) -> EvaluationReport:
    calibration_rows = list(calibration)
    holdout_rows = list(holdout)
    _validate_threshold(threshold)
    _require_unique_sample_ids(calibration_rows, split_name="calibration")
    _require_unique_sample_ids(holdout_rows, split_name="holdout")

    leaked_ids = {row.sample_id for row in calibration_rows} & {
        row.sample_id for row in holdout_rows
    }
    if leaked_ids:
        preview = ", ".join(sorted(leaked_ids)[:3])
        raise ValueError(f"sample leakage between calibration and holdout: {preview}")

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
    by_source = {
        source: _classification_metrics(rows, threshold=threshold)
        for source, rows in sorted(grouped.items())
    }
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
