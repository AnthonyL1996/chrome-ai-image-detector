import math
import unittest

from poidh_benchmark.evaluation import (
    BOUNTY_THRESHOLD,
    PlattCalibrator,
    Prediction,
    balanced_accuracy,
    evaluate_holdout,
    fit_platt_calibrator,
)


class BalancedAccuracyTests(unittest.TestCase):
    def test_uses_fixed_inclusive_65_percent_ai_threshold(self) -> None:
        rows = [
            Prediction("real-ok", 0, 0.10, "camera"),
            Prediction("real-fp", 0, 0.80, "camera"),
            Prediction("ai-boundary", 1, 0.65, "flux"),
            Prediction("ai-fn", 1, 0.64, "flux"),
        ]

        result = balanced_accuracy(rows, threshold=0.65)

        self.assertEqual(result.true_negative_rate, 0.5)
        self.assertEqual(result.true_positive_rate, 0.5)
        self.assertEqual(result.balanced_accuracy, 0.5)
        self.assertEqual((result.tn, result.fp, result.fn, result.tp), (1, 1, 1, 1))

    def test_rejects_missing_class_instead_of_reporting_a_misleading_score(
        self,
    ) -> None:
        rows = [Prediction("only-real", 0, 0.1, "camera")]

        with self.assertRaisesRegex(ValueError, "both real and AI"):
            balanced_accuracy(rows, threshold=0.65)

    def test_rejects_scores_outside_probability_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            Prediction("bad", 1, 1.01, "flux")


class CalibrationContractTests(unittest.TestCase):
    def test_bounty_holdout_threshold_cannot_be_overridden(self) -> None:
        calibration = [
            Prediction("cr", 0, 0.2, "cal-real"),
            Prediction("ca", 1, 0.8, "cal-ai"),
        ]
        holdout = [
            Prediction("hr", 0, 0.2, "camera"),
            Prediction("ha", 1, 0.8, "flux"),
        ]

        for invalid_threshold in (0.64, 0.66):
            with self.subTest(threshold=invalid_threshold):
                with self.assertRaisesRegex(ValueError, "fixed at 0.65"):
                    evaluate_holdout(
                        calibration,
                        holdout,
                        threshold=invalid_threshold,
                    )

    def test_platt_calibrator_returns_bounded_monotonic_probabilities(self) -> None:
        calibration = [
            Prediction("r1", 0, 0.10, "camera"),
            Prediction("r2", 0, 0.25, "camera"),
            Prediction("a1", 1, 0.70, "sdxl"),
            Prediction("a2", 1, 0.90, "sdxl"),
        ]

        calibrator = fit_platt_calibrator(calibration)
        probabilities = [calibrator.transform(value) for value in (0.1, 0.5, 0.9)]

        self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities))
        self.assertLess(probabilities[0], probabilities[1])
        self.assertLess(probabilities[1], probabilities[2])
        self.assertTrue(all(math.isfinite(value) for value in probabilities))

    def test_holdout_evaluation_rejects_sample_leakage(self) -> None:
        calibration = [
            Prediction("duplicate", 0, 0.2, "camera"),
            Prediction("cal-ai", 1, 0.8, "sdxl"),
        ]
        holdout = [
            Prediction("duplicate", 0, 0.3, "camera-2"),
            Prediction("test-ai", 1, 0.9, "flux"),
        ]

        with self.assertRaisesRegex(ValueError, "sample leakage"):
            evaluate_holdout(calibration, holdout, threshold=0.65)

    def test_holdout_rejects_content_and_provenance_leakage(self) -> None:
        calibration = [
            Prediction(
                "cal-real",
                0,
                0.2,
                "camera",
                content_sha256="a" * 64,
                provenance_group="photo-1",
            ),
            Prediction("cal-ai", 1, 0.8, "sdxl"),
        ]
        same_content = [
            Prediction("other-id", 0, 0.3, "web", content_sha256="a" * 64),
            Prediction("test-ai", 1, 0.9, "flux"),
        ]
        same_parent = [
            Prediction("crop-id", 0, 0.3, "web", provenance_group="photo-1"),
            Prediction("test-ai", 1, 0.9, "flux"),
        ]

        with self.assertRaisesRegex(ValueError, "content leakage"):
            evaluate_holdout(calibration, same_content)
        with self.assertRaisesRegex(ValueError, "provenance leakage"):
            evaluate_holdout(calibration, same_parent)

    def test_generator_disjoint_mode_rejects_family_overlap(self) -> None:
        calibration = [
            Prediction("cal-real", 0, 0.2, "camera"),
            Prediction("cal-ai", 1, 0.8, "sdxl-v1", generator_family="sdxl"),
        ]
        holdout = [
            Prediction("test-real", 0, 0.2, "web"),
            Prediction("test-ai", 1, 0.8, "sdxl-turbo", generator_family="sdxl"),
        ]

        with self.assertRaisesRegex(ValueError, "generator-family leakage"):
            evaluate_holdout(calibration, holdout, require_generator_disjoint=True)

    def test_calibration_corrects_shifted_scores_at_fixed_bounty_threshold(
        self,
    ) -> None:
        calibration = [
            Prediction("r1", 0, 0.25, "camera"),
            Prediction("r2", 0, 0.35, "camera"),
            Prediction("a1", 1, 0.50, "sdxl"),
            Prediction("a2", 1, 0.60, "sdxl"),
        ]

        calibrator = fit_platt_calibrator(calibration)

        self.assertLess(0.55, BOUNTY_THRESHOLD)
        self.assertGreaterEqual(calibrator.transform(0.55), BOUNTY_THRESHOLD)

    def test_calibration_is_class_balanced(self) -> None:
        balanced = [
            Prediction("r1", 0, 0.2, "camera"),
            Prediction("r2", 0, 0.3, "camera"),
            Prediction("a1", 1, 0.7, "sdxl"),
            Prediction("a2", 1, 0.8, "sdxl"),
        ]
        duplicated_real = balanced + [
            Prediction("r3", 0, 0.2, "camera"),
            Prediction("r4", 0, 0.3, "camera"),
        ]

        base = fit_platt_calibrator(balanced)
        duplicated = fit_platt_calibrator(duplicated_real)

        self.assertAlmostEqual(base.transform(0.5), duplicated.transform(0.5), places=6)

    def test_calibrator_and_optimizer_reject_invalid_invariants(self) -> None:
        for scale in (-1.0, 0.0, math.nan, math.inf):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    PlattCalibrator(scale=scale, bias=0.0)
        with self.assertRaises(ValueError):
            PlattCalibrator(scale=1.0, bias=math.nan)

        rows = [
            Prediction("r", 0, 0.2, "camera"),
            Prediction("a", 1, 0.8, "sdxl"),
        ]
        for learning_rate in (0.0, -0.1, math.nan, math.inf):
            with self.subTest(learning_rate=learning_rate):
                with self.assertRaises(ValueError):
                    fit_platt_calibrator(rows, learning_rate=learning_rate)
        for iterations in (0, -1, 1.5, True):
            with self.subTest(iterations=iterations):
                with self.assertRaises(ValueError):
                    fit_platt_calibrator(rows, iterations=iterations)

    def test_holdout_reports_per_generator_metrics_and_fixed_threshold(self) -> None:
        calibration = [
            Prediction("cr1", 0, 0.05, "cal-real"),
            Prediction("cr2", 0, 0.20, "cal-real"),
            Prediction("ca1", 1, 0.80, "cal-ai"),
            Prediction("ca2", 1, 0.95, "cal-ai"),
        ]
        holdout = [
            Prediction("hr1", 0, 0.05, "camera"),
            Prediction("hr2", 0, 0.15, "camera"),
            Prediction("ha1", 1, 0.90, "flux"),
            Prediction("ha2", 1, 0.85, "midjourney"),
        ]

        report = evaluate_holdout(calibration, holdout, threshold=0.65)

        self.assertEqual(report.threshold, BOUNTY_THRESHOLD)
        self.assertEqual(report.overall.balanced_accuracy, 1.0)
        self.assertEqual(set(report.by_source), {"camera", "flux", "midjourney"})
        self.assertEqual(report.by_source["flux"].balanced_accuracy, 1.0)
        self.assertEqual(report.by_source["midjourney"].balanced_accuracy, 1.0)
        self.assertEqual(report.by_source["camera"].balanced_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
