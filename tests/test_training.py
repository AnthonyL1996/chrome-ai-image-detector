import hashlib
import math
import unittest

from poidh_detector.training import (
    CheckpointCandidate,
    TrainingConfig,
    select_best_checkpoint,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class TrainingConfigTests(unittest.TestCase):
    def test_is_canonical_and_freezes_random_initialization_contract(self) -> None:
        config = TrainingConfig(
            dataset_manifest_sha256=_digest("dataset"),
            split_manifest_sha256=_digest("splits"),
            calibration_split_sha256=_digest("calibration"),
            exposed_holdout_sha256=(_digest("v2"), _digest("v1")),
            seed=323,
        )
        reordered = TrainingConfig(
            dataset_manifest_sha256=_digest("dataset"),
            split_manifest_sha256=_digest("splits"),
            calibration_split_sha256=_digest("calibration"),
            exposed_holdout_sha256=(_digest("v1"), _digest("v2")),
            seed=323,
        )

        self.assertEqual(config, reordered)
        self.assertEqual(config.sha256, reordered.sha256)
        self.assertEqual(config.architecture, "convnextv2_nano")
        self.assertEqual(config.weights_origin, "random_initialization")
        self.assertFalse(config.pretrained)
        self.assertEqual(config.selection_metric, "validation_bce")
        self.assertTrue(config.selection_minimize)
        self.assertTrue(config.to_json_bytes().endswith(b"\n"))

        with self.assertRaisesRegex(AttributeError, "cannot assign"):
            config.seed = 7  # type: ignore[misc]

    def test_rejects_invalid_or_duplicate_contract_values(self) -> None:
        valid = {
            "dataset_manifest_sha256": _digest("dataset"),
            "split_manifest_sha256": _digest("splits"),
            "calibration_split_sha256": _digest("calibration"),
            "exposed_holdout_sha256": (_digest("v1"), _digest("v2")),
            "seed": 323,
        }
        invalid_cases = (
            {"dataset_manifest_sha256": "A" * 64},
            {"split_manifest_sha256": "main"},
            {"calibration_split_sha256": "main"},
            {"exposed_holdout_sha256": ()},
            {
                "exposed_holdout_sha256": (
                    _digest("v1"),
                    _digest("v1"),
                )
            },
            {"seed": -1},
            {"seed": True},
        )

        for changes in invalid_cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    TrainingConfig(**(valid | changes))


class CheckpointSelectionTests(unittest.TestCase):
    def test_selects_only_by_validation_bce(self) -> None:
        candidates = [
            CheckpointCandidate("epoch-1", 1, 100, 0.25, training_bce=0.05),
            CheckpointCandidate("epoch-2", 2, 200, 0.20, training_bce=0.80),
        ]

        selected = select_best_checkpoint(candidates)

        self.assertEqual(selected.checkpoint_id, "epoch-2")

    def test_selection_is_order_independent_with_deterministic_ties(self) -> None:
        earlier = CheckpointCandidate("z", 2, 200, 0.2)
        later = CheckpointCandidate("a", 3, 300, 0.2)

        self.assertEqual(
            select_best_checkpoint([later, earlier]),
            select_best_checkpoint([earlier, later]),
        )
        self.assertEqual(select_best_checkpoint([later, earlier]), earlier)

    def test_rejects_invalid_or_ambiguous_candidates(self) -> None:
        valid = CheckpointCandidate("epoch-1", 1, 100, 0.2)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            select_best_checkpoint([])
        with self.assertRaisesRegex(ValueError, "duplicate checkpoint"):
            select_best_checkpoint([valid, valid])
        for value in (-0.1, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CheckpointCandidate("bad", 1, 1, value)


if __name__ == "__main__":
    unittest.main()
