import math
import unittest

from poidh_benchmark.evaluation import (
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

    def test_rejects_missing_class_instead_of_reporting_a_misleading_score(self) -> None:
        rows = [Prediction("only-real", 0, 0.1, "camera")]

        with self.assertRaisesRegex(ValueError, "both real and AI"):
            balanced_accuracy(rows, threshold=0.65)

    def test_rejects_scores_outside_probability_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            Prediction("bad", 1, 1.01, "flux")


class CalibrationContractTests(unittest.TestCase):
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

        self.assertEqual(report.threshold, 0.65)
        self.assertEqual(report.overall.balanced_accuracy, 1.0)
        self.assertEqual(set(report.by_source), {"camera", "flux", "midjourney"})


if __name__ == "__main__":
    unittest.main()
