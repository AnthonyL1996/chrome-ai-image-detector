from __future__ import annotations

import hashlib
import json
import math
import unittest

from poidh_detector.calibration import (
    CalibrationClassCounts,
    PlattCalibrationArtifact,
)
from poidh_detector.training import TrainingConfig


_MIRAGE_V1_SHA256 = "f92cf3e694b14cb96c64aae0b36fb44b47bd81f0189dc18c23b652e242e4f94a"
_MIRAGE_V2_SHA256 = "857b7410280011a5b98a55d22c1155d49212519331b4a63bf344a43b5836eadc"


def _training_config(
    *,
    calibration_split_sha256: str = "b" * 64,
    exposed_holdout_sha256: tuple[str, ...] = (
        _MIRAGE_V1_SHA256,
        _MIRAGE_V2_SHA256,
    ),
) -> TrainingConfig:
    return TrainingConfig(
        dataset_manifest_sha256="d" * 64,
        split_manifest_sha256="e" * 64,
        calibration_split_sha256=calibration_split_sha256,
        exposed_holdout_sha256=exposed_holdout_sha256,
        seed=323,
    )


def _artifact(**overrides: object) -> PlattCalibrationArtifact:
    values: dict[str, object] = {
        "scale": 2.0,
        "bias": -0.5,
        "checkpoint_sha256": "a" * 64,
        "calibration_split_sha256": "b" * 64,
        "training_config": _training_config(),
        "input_identifier": "calibration",
        "sample_count": 10,
        "class_counts": CalibrationClassCounts(real=6, ai=4),
    }
    values.update(overrides)
    return PlattCalibrationArtifact(**values)  # type: ignore[arg-type]


class PlattCalibrationArtifactTests(unittest.TestCase):
    def test_serializes_canonical_frozen_contract_and_digest(self) -> None:
        artifact = _artifact()
        expected_document = {
            "bias": -0.5,
            "calibration_split_sha256": "b" * 64,
            "checkpoint_sha256": "a" * 64,
            "class_counts": {"ai": 4, "real": 6},
            "input_identifier": "calibration",
            "method": "platt_on_raw_logit",
            "training_config_sha256": _training_config().sha256,
            "sample_count": 10,
            "scale": 2.0,
            "schema_version": 1,
            "threshold": 0.65,
        }
        expected = (
            json.dumps(expected_document, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")

        self.assertEqual(artifact.schema_version, 1)
        self.assertEqual(artifact.method, "platt_on_raw_logit")
        self.assertEqual(artifact.threshold, 0.65)
        self.assertEqual(artifact.to_json_bytes(), expected)
        self.assertEqual(artifact.sha256, hashlib.sha256(expected).hexdigest())
        self.assertEqual(_artifact().to_json_bytes(), artifact.to_json_bytes())

    def test_fixed_threshold_cannot_be_overridden(self) -> None:
        with self.assertRaises(TypeError):
            _artifact(threshold=0.66)

    def test_requires_finite_positive_scale_and_finite_bias(self) -> None:
        for scale in (0.0, -1.0, math.inf, -math.inf, math.nan, True):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "scale"):
                    _artifact(scale=scale)

        for bias in (math.inf, -math.inf, math.nan, True):
            with self.subTest(bias=bias):
                with self.assertRaisesRegex(ValueError, "bias"):
                    _artifact(bias=bias)

    def test_requires_canonical_checkpoint_and_split_digests(self) -> None:
        for field in ("checkpoint_sha256", "calibration_split_sha256"):
            for digest in ("a" * 63, "A" * 64, "z" * 64, True):
                with self.subTest(field=field, digest=digest):
                    with self.assertRaisesRegex(ValueError, field):
                        _artifact(**{field: digest})

    def test_requires_consistent_nonempty_class_counts(self) -> None:
        for counts in (
            CalibrationClassCounts(real=1, ai=1),
            CalibrationClassCounts(real=6, ai=4),
        ):
            with self.subTest(counts=counts):
                artifact = _artifact(
                    class_counts=counts,
                    sample_count=counts.real + counts.ai,
                )
                self.assertEqual(artifact.sample_count, counts.real + counts.ai)

        for field, value in (
            ("real", 0),
            ("ai", 0),
            ("real", -1),
            ("ai", True),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    CalibrationClassCounts(
                        **{field: value, "ai" if field == "real" else "real": 1}
                    )

        for sample_count in (9, 11, 0, True):
            with self.subTest(sample_count=sample_count):
                with self.assertRaisesRegex(ValueError, "sample_count"):
                    _artifact(sample_count=sample_count)

    def test_rejects_non_calibration_input_identifiers(self) -> None:
        for identifier in (
            "holdout",
            "development_exposed",
            "mirage-v1",
            "validation",
            "",
        ):
            with self.subTest(identifier=identifier):
                with self.assertRaisesRegex(ValueError, "input_identifier"):
                    _artifact(input_identifier=identifier)

    def test_requires_exact_expected_calibration_split_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "calibration split digest mismatch"):
            _artifact(calibration_split_sha256="c" * 64)

    def test_rejects_registered_exposed_holdout_as_calibration_split(self) -> None:
        for digest in (_MIRAGE_V1_SHA256, _MIRAGE_V2_SHA256):
            with self.subTest(digest=digest):
                with self.assertRaisesRegex(ValueError, "exposed holdout"):
                    _artifact(
                        calibration_split_sha256=digest,
                        training_config=_training_config(
                            calibration_split_sha256=digest
                        ),
                    )

    def test_caller_cannot_substitute_deny_set_independently_of_training_config(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            _artifact(exposed_holdout_sha256=("c" * 64,))

        with self.assertRaisesRegex(ValueError, "exposed holdout"):
            _artifact(
                calibration_split_sha256=_MIRAGE_V1_SHA256,
                training_config=_training_config(
                    calibration_split_sha256=_MIRAGE_V1_SHA256,
                ),
            )

    def test_requires_training_config_contract(self) -> None:
        with self.assertRaisesRegex(TypeError, "training_config"):
            _artifact(training_config=object())

    def test_applies_monotonic_stable_sigmoid_to_extreme_logits(self) -> None:
        artifact = _artifact(scale=2.0, bias=0.0)
        probabilities = [
            artifact.probability_ai(logit) for logit in (-1e308, 0.0, 1e308)
        ]

        self.assertEqual(probabilities[0], 0.0)
        self.assertEqual(probabilities[1], 0.5)
        self.assertEqual(probabilities[2], 1.0)
        self.assertEqual(probabilities, sorted(probabilities))

        for logit in (math.inf, -math.inf, math.nan, True):
            with self.subTest(logit=logit):
                with self.assertRaisesRegex(ValueError, "logit"):
                    artifact.probability_ai(logit)


if __name__ == "__main__":
    unittest.main()
