from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from poidh_detector.calibration_fit import (
    CalibrationPrediction,
    CalibrationPredictions,
    fit_platt_calibration,
)
from poidh_detector.data import SplitManifest
from poidh_detector.training import TrainingConfig
from tools.fit_detector_calibration import main


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _split_manifest(**assignment_overrides: str) -> SplitManifest:
    assignments = {
        "real-hard": "calibration",
        "real-easy": "calibration",
        "ai-hard": "calibration",
        "ai-easy": "calibration",
        "real": "calibration",
        "ai": "calibration",
        "real-1": "calibration",
        "real-2": "calibration",
        "validation-sample": "validation",
        "train-sample": "train",
    }
    assignments.update(assignment_overrides)
    return SplitManifest(
        dataset_manifest_sha256=_digest("dataset"),
        seed=323,
        ratios={"train": 0.8, "validation": 0.1, "calibration": 0.1},
        assignments=assignments,
    )


def _config(
    *, split_manifest: SplitManifest | None = None, **overrides: object
) -> TrainingConfig:
    frozen_split = split_manifest or _split_manifest()
    values: dict[str, object] = {
        "dataset_manifest_sha256": _digest("dataset"),
        "split_manifest_sha256": frozen_split.sha256,
        "calibration_split_sha256": _digest("calibration"),
        "exposed_holdout_sha256": (_digest("holdout-v1"), _digest("holdout-v2")),
        "seed": 323,
    }
    values.update(overrides)
    return TrainingConfig(**values)  # type: ignore[arg-type]


def _predictions(
    *,
    rows: tuple[CalibrationPrediction, ...] | None = None,
    config: TrainingConfig | None = None,
    checkpoint_sha256: str = "a" * 64,
    input_identifier: str = "calibration",
    calibration_split_sha256: str | None = None,
) -> CalibrationPredictions:
    training_config = config or _config()
    return CalibrationPredictions(
        schema_version=1,
        input_identifier=input_identifier,
        checkpoint_sha256=checkpoint_sha256,
        calibration_split_sha256=(
            calibration_split_sha256 or training_config.calibration_split_sha256
        ),
        training_config_sha256=training_config.sha256,
        predictions=rows
        or (
            CalibrationPrediction("real-hard", 1.0, 0),
            CalibrationPrediction("real-easy", -0.2, 0),
            CalibrationPrediction("ai-hard", -1.0, 1),
            CalibrationPrediction("ai-easy", 0.2, 1),
        ),
    )


class CalibrationPredictionsTests(unittest.TestCase):
    def test_serialization_is_canonical_and_order_independent(self) -> None:
        rows = (
            CalibrationPrediction("b", 0.25, 1),
            CalibrationPrediction("a", -0.5, 0),
        )
        forward = _predictions(rows=rows)
        reverse = _predictions(rows=tuple(reversed(rows)))

        self.assertEqual(forward, reverse)
        self.assertEqual(forward.to_json_bytes(), reverse.to_json_bytes())
        self.assertEqual(
            forward.sha256, hashlib.sha256(forward.to_json_bytes()).hexdigest()
        )
        self.assertEqual(
            CalibrationPredictions.from_json_bytes(forward.to_json_bytes()), forward
        )
        document = json.loads(forward.to_json_bytes())
        self.assertEqual(
            [row["sample_id"] for row in document["predictions"]], ["a", "b"]
        )

    def test_rejects_ambiguous_or_invalid_prediction_rows(self) -> None:
        valid = CalibrationPrediction("sample", 0.0, 0)
        with self.assertRaisesRegex(ValueError, "duplicate sample"):
            _predictions(rows=(valid, valid))
        for raw_logit in (math.inf, -math.inf, math.nan, True):
            with self.subTest(raw_logit=raw_logit):
                with self.assertRaisesRegex(ValueError, "raw_logit"):
                    CalibrationPrediction("sample", raw_logit, 0)
        for label in (-1, 2, True, 0.0):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "label"):
                    CalibrationPrediction("sample", 0.0, label)

        with self.assertRaisesRegex(ValueError, "schema version"):
            CalibrationPredictions(
                schema_version=1.0,  # type: ignore[arg-type]
                input_identifier="calibration",
                checkpoint_sha256="a" * 64,
                calibration_split_sha256="b" * 64,
                training_config_sha256="c" * 64,
                predictions=(valid,),
            )

    def test_rejects_non_calibration_identifiers_at_input_boundary(self) -> None:
        for identifier in ("validation", "holdout", "development_exposed", ""):
            with self.subTest(identifier=identifier):
                with self.assertRaisesRegex(ValueError, "input_identifier"):
                    _predictions(input_identifier=identifier)


class CalibrationFittingTests(unittest.TestCase):
    def test_deterministically_fits_and_improves_bce_without_threshold_search(
        self,
    ) -> None:
        config = _config()
        split_manifest = _split_manifest()
        predictions = _predictions(config=config)

        forward = fit_platt_calibration(
            predictions,
            training_config=config,
            checkpoint_sha256="a" * 64,
            split_manifest=split_manifest,
        )
        reverse = fit_platt_calibration(
            _predictions(rows=tuple(reversed(predictions.predictions)), config=config),
            training_config=config,
            checkpoint_sha256="a" * 64,
            split_manifest=split_manifest,
        )

        self.assertEqual(forward.to_json_bytes(), reverse.to_json_bytes())
        self.assertEqual(forward.artifact.checkpoint_sha256, "a" * 64)
        self.assertEqual(forward.artifact.training_config, config)
        self.assertLess(forward.calibrated.bce, forward.uncalibrated.bce)
        self.assertGreaterEqual(forward.calibrated.ece, 0.0)
        self.assertLessEqual(forward.calibrated.ece, 1.0)
        self.assertEqual(forward.threshold, 0.65)
        self.assertNotIn("optimized_threshold", json.loads(forward.to_json_bytes()))

    def test_fit_is_stable_for_extreme_finite_logits(self) -> None:
        config = _config()
        split_manifest = _split_manifest()
        predictions = _predictions(
            config=config,
            rows=(
                CalibrationPrediction("real", -1e308, 0),
                CalibrationPrediction("ai", 1e308, 1),
            ),
        )

        result = fit_platt_calibration(
            predictions,
            training_config=config,
            checkpoint_sha256="a" * 64,
            split_manifest=split_manifest,
        )

        self.assertTrue(math.isfinite(result.artifact.scale))
        self.assertGreater(result.artifact.scale, 0.0)
        self.assertTrue(math.isfinite(result.artifact.bias))
        probabilities = (
            result.artifact.probability_ai(-1e308),
            result.artifact.probability_ai(1e308),
        )
        self.assertTrue(all(math.isfinite(value) for value in probabilities))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities))
        self.assertLess(probabilities[0], probabilities[1])

    def test_requires_both_classes_and_exact_frozen_bindings(self) -> None:
        config = _config()
        split_manifest = _split_manifest()
        one_class = _predictions(
            config=config,
            rows=(
                CalibrationPrediction("real-1", -1.0, 0),
                CalibrationPrediction("real-2", 1.0, 0),
            ),
        )
        with self.assertRaisesRegex(ValueError, "both real and AI"):
            fit_platt_calibration(
                one_class,
                training_config=config,
                checkpoint_sha256="a" * 64,
                split_manifest=split_manifest,
            )

        cases = (
            (
                _predictions(config=config, checkpoint_sha256="b" * 64),
                config,
                "a" * 64,
                "checkpoint digest mismatch",
            ),
            (
                _predictions(config=config, calibration_split_sha256="c" * 64),
                config,
                "a" * 64,
                "calibration split digest mismatch",
            ),
            (
                _predictions(config=config),
                _config(seed=324),
                "a" * 64,
                "training config digest mismatch",
            ),
        )
        for artifact, training_config, checkpoint, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    fit_platt_calibration(
                        artifact,
                        training_config=training_config,
                        checkpoint_sha256=checkpoint,
                        split_manifest=split_manifest,
                    )

    def test_refuses_ids_not_assigned_to_frozen_calibration_split(self) -> None:
        split_manifest = _split_manifest()
        config = _config(split_manifest=split_manifest)
        cases = (
            ("validation-sample", "validation"),
            ("missing-sample", "not present"),
        )
        for sample_id, message in cases:
            with self.subTest(sample_id=sample_id):
                predictions = _predictions(
                    config=config,
                    rows=(
                        CalibrationPrediction(sample_id, -1.0, 0),
                        CalibrationPrediction("ai", 1.0, 1),
                    ),
                )
                with self.assertRaisesRegex(ValueError, message):
                    fit_platt_calibration(
                        predictions,
                        training_config=config,
                        checkpoint_sha256="a" * 64,
                        split_manifest=split_manifest,
                    )

        wrong_manifest = _split_manifest(extra="calibration")
        with self.assertRaisesRegex(ValueError, "split manifest digest mismatch"):
            fit_platt_calibration(
                _predictions(config=config),
                training_config=config,
                checkpoint_sha256="a" * 64,
                split_manifest=wrong_manifest,
            )

    def test_refuses_exposed_holdout_digest_even_if_config_is_misbound(self) -> None:
        holdout = _digest("holdout-v1")
        config = _config(calibration_split_sha256=holdout)
        split_manifest = _split_manifest()
        predictions = _predictions(config=config, calibration_split_sha256=holdout)

        with self.assertRaisesRegex(ValueError, "exposed holdout"):
            fit_platt_calibration(
                predictions,
                training_config=config,
                checkpoint_sha256="a" * 64,
                split_manifest=split_manifest,
            )

    def test_cli_hashes_checkpoint_and_refuses_overwrite(self) -> None:
        config = _config()
        split_manifest = _split_manifest()
        checkpoint = b"frozen checkpoint"
        checkpoint_sha256 = hashlib.sha256(checkpoint).hexdigest()
        predictions = _predictions(
            config=config,
            checkpoint_sha256=checkpoint_sha256,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions_path = root / "predictions.json"
            config_path = root / "training-config.json"
            checkpoint_path = root / "model.bin"
            split_manifest_path = root / "splits.json"
            output_path = root / "calibration-fit.json"
            predictions_path.write_bytes(predictions.to_json_bytes())
            config_path.write_bytes(config.to_json_bytes())
            checkpoint_path.write_bytes(checkpoint)
            split_manifest_path.write_bytes(split_manifest.to_json_bytes())

            self.assertEqual(
                main(
                    [
                        "--predictions",
                        str(predictions_path),
                        "--training-config",
                        str(config_path),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--split-manifest",
                        str(split_manifest_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            document = json.loads(output_path.read_bytes())
            self.assertEqual(
                document["artifact"]["checkpoint_sha256"], checkpoint_sha256
            )
            self.assertEqual(document["threshold"], 0.65)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                main(
                    [
                        "--predictions",
                        str(predictions_path),
                        "--training-config",
                        str(config_path),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--split-manifest",
                        str(split_manifest_path),
                        "--output",
                        str(output_path),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
