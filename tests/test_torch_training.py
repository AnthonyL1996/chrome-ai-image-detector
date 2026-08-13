from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from poidh_detector.reproducibility import EnvironmentFingerprint
import poidh_detector.torch_training as torch_training
from poidh_detector.torch_training import (
    OptimizationConfig,
    ResumeContract,
    ValidationMetrics,
    binary_auc,
    candidate_from_validation,
    cosine_warmup_multiplier,
    load_current_generation,
    publish_generation,
    reserve_run_directory,
    validate_generation_resume_contract,
    validate_resume_contract,
)
from poidh_detector.training import CheckpointCandidate, TrainingConfig
from tools import train_detector


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        dataset_manifest_sha256=_digest("dataset"),
        split_manifest_sha256=_digest("split"),
        calibration_split_sha256=_digest("calibration"),
        exposed_holdout_sha256=(_digest("holdout"),),
        seed=323,
    )


def _environment() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        python_version="3.12.0",
        platform="test",
        machine="x86_64",
        torch_version="2.7.0",
        numpy_version="2.2.0",
        timm_version="1.0.0",
        cuda_available=True,
        cuda_version="12.8",
        cudnn_version="9",
        cuda_device="RTX 3090",
    )


class OptimizationTests(unittest.TestCase):
    def test_profiles_freeze_stage_sizes_and_optimizer_contract(self) -> None:
        smoke = OptimizationConfig.for_profile("smoke")
        full = OptimizationConfig.for_profile("full")

        self.assertEqual(smoke.optimizer, "adamw")
        self.assertEqual(smoke.loss, "bce_with_logits")
        self.assertEqual(smoke.schedule, "linear_warmup_cosine")
        self.assertEqual(smoke.image_size, 224)
        self.assertIsNotNone(smoke.train_per_class_cap)
        self.assertIsNone(full.train_per_class_cap)

    def test_cosine_schedule_has_linear_warmup_and_zero_endpoint(self) -> None:
        self.assertEqual(cosine_warmup_multiplier(0, 100, 10), 0.0)
        self.assertEqual(cosine_warmup_multiplier(5, 100, 10), 0.5)
        self.assertEqual(cosine_warmup_multiplier(10, 100, 10), 1.0)
        self.assertAlmostEqual(cosine_warmup_multiplier(100, 100, 10), 0.0)
        self.assertTrue(0 < cosine_warmup_multiplier(55, 100, 10) < 1)

        with self.assertRaises(ValueError):
            cosine_warmup_multiplier(0, 0, 0)


class MetricAndSelectionTests(unittest.TestCase):
    def test_auc_handles_ranking_and_ties(self) -> None:
        self.assertEqual(binary_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(binary_auc([0, 1], [0.5, 0.5]), 0.5)
        with self.assertRaisesRegex(ValueError, "both classes"):
            binary_auc([0, 0], [0.1, 0.2])

    def test_candidate_is_derived_from_validation_bce(self) -> None:
        candidate = candidate_from_validation(
            epoch=2,
            global_step=20,
            training_bce=0.01,
            validation=ValidationMetrics(bce=0.4, auc=0.99, count=10),
        )
        self.assertEqual(candidate.validation_bce, 0.4)
        self.assertEqual(candidate.training_bce, 0.01)


class ResumeTests(unittest.TestCase):
    def test_resume_contract_captures_config_environment_and_progress(self) -> None:
        config = _training_config()
        optimization = OptimizationConfig.for_profile("pilot")
        contract = ResumeContract.create(
            config=config,
            optimization=optimization,
            environment=_environment(),
            completed_epoch=3,
            global_step=150,
        )

        self.assertEqual(contract.training_config_sha256, config.sha256)
        self.assertEqual(
            contract.dataset_manifest_sha256, config.dataset_manifest_sha256
        )
        self.assertEqual(contract.environment["cuda_device"], "RTX 3090")
        self.assertEqual(contract.completed_epoch, 3)
        self.assertEqual(contract.global_step, 150)
        self.assertTrue(contract.to_json_bytes().endswith(b"\n"))
        validate_resume_contract(contract, config, optimization, _environment())

    def test_resume_rejects_changed_config_or_environment(self) -> None:
        config = _training_config()
        optimization = OptimizationConfig.for_profile("pilot")
        contract = ResumeContract.create(
            config=config,
            optimization=optimization,
            environment=_environment(),
            completed_epoch=1,
            global_step=1,
        )
        changed = EnvironmentFingerprint(
            **(_environment().to_dict() | {"torch_version": "2.8.0"})
        )

        with self.assertRaisesRegex(ValueError, "environment"):
            validate_resume_contract(contract, config, optimization, changed)
        with self.assertRaisesRegex(ValueError, "optimization"):
            validate_resume_contract(
                contract, config, OptimizationConfig.for_profile("full"), _environment()
            )

    def test_metric_validation_rejects_non_finite_values(self) -> None:
        with self.assertRaises(ValueError):
            ValidationMetrics(bce=math.nan, auc=0.5, count=1)


class TransactionalRunTests(unittest.TestCase):
    def test_file_fsync_uses_writable_descriptor_without_changing_bytes(self) -> None:
        events: list[tuple[object, ...]] = []
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.bin"
            path.write_bytes(b"immutable-checkpoint")

            def open_file(opened_path: Path, mode: str):
                events.append(("open", opened_path, mode))
                return opened_path.open(mode)

            torch_training._fsync_file(
                path,
                open_file=open_file,
                fsync_descriptor=lambda descriptor: events.append(
                    ("fsync", descriptor)
                ),
            )

            self.assertEqual(path.read_bytes(), b"immutable-checkpoint")
            self.assertEqual(events[0], ("open", path, "r+b"))
            self.assertEqual(events[1][0], "fsync")

    def test_file_fsync_refuses_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.bin"

            with self.assertRaises(FileNotFoundError):
                torch_training._fsync_file(missing)

    def test_windows_skips_directory_fsync_but_publishes_with_file_fsync(self) -> None:
        config = _training_config()
        optimization = OptimizationConfig.for_profile("pilot")
        environment = _environment()
        candidate = CheckpointCandidate("epoch-0001", 1, 10, 0.4, 0.3)
        contract = ResumeContract.create(
            config=config,
            optimization=optimization,
            environment=environment,
            completed_epoch=1,
            global_step=10,
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            reserve_run_directory(output, os_name="nt")
            with patch.object(
                torch_training,
                "_fsync_file",
                wraps=torch_training._fsync_file,
            ) as file_fsync:
                published = publish_generation(
                    output,
                    contract=contract,
                    candidates=[candidate],
                    selected=candidate,
                    write_resume=lambda path: path.write_bytes(b"resume"),
                    write_best_weights=lambda path: path.write_bytes(b"weights"),
                    os_name="nt",
                )

            self.assertTrue((output / "CURRENT").is_file())
            self.assertTrue(published.path.is_dir())
            self.assertEqual(
                (published.path / "READY").read_bytes(),
                (published.manifest.sha256 + "\n").encode("ascii"),
            )
            self.assertEqual(file_fsync.call_count, 5)

    def test_posix_directory_fsync_opens_syncs_and_closes_directory(self) -> None:
        events: list[tuple[object, ...]] = []

        torch_training._fsync_directory(
            Path("/run"),
            os_name="posix",
            open_directory=lambda path, flags: events.append(("open", path, flags))
            or 17,
            fsync_descriptor=lambda descriptor: events.append(("fsync", descriptor)),
            close_descriptor=lambda descriptor: events.append(("close", descriptor)),
        )

        self.assertEqual(events[0][0:2], ("open", Path("/run")))
        self.assertEqual(events[1:], [("fsync", 17), ("close", 17)])

    def test_fresh_run_atomically_reserves_and_never_reuses_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            reserve_run_directory(output)

            self.assertTrue(output.is_dir())
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                reserve_run_directory(output)

            empty = Path(temporary) / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                reserve_run_directory(empty)

    def test_cli_output_preparation_never_clobbers_or_repairs_existing_path(
        self,
    ) -> None:
        config = _training_config()
        optimization = OptimizationConfig.for_profile("pilot")
        environment = _environment()
        plan = {"profile": "pilot"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                train_detector._prepare_run_output(
                    output,
                    resume=False,
                    config=config,
                    optimization=optimization,
                    environment=environment,
                    plan=plan,
                )
            self.assertEqual(list(output.iterdir()), [])

            with self.assertRaisesRegex(ValueError, "provenance does not match"):
                train_detector._prepare_run_output(
                    output,
                    resume=True,
                    config=config,
                    optimization=optimization,
                    environment=environment,
                    plan=plan,
                )
            self.assertEqual(list(output.iterdir()), [])

    def test_resume_ignores_generation_interrupted_before_pointer_publish(self) -> None:
        config = _training_config()
        optimization = OptimizationConfig.for_profile("pilot")
        environment = _environment()
        first = CheckpointCandidate("epoch-0001", 1, 10, 0.4, 0.3)
        second = CheckpointCandidate("epoch-0002", 2, 20, 0.3, 0.2)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            reserve_run_directory(output)
            first_contract = ResumeContract.create(
                config=config,
                optimization=optimization,
                environment=environment,
                completed_epoch=1,
                global_step=10,
            )
            publish_generation(
                output,
                contract=first_contract,
                candidates=[first],
                selected=first,
                write_resume=lambda path: path.write_bytes(b"resume-one"),
                write_best_weights=lambda path: path.write_bytes(b"weights-one"),
            )

            second_contract = ResumeContract.create(
                config=config,
                optimization=optimization,
                environment=environment,
                completed_epoch=2,
                global_step=20,
            )

            for interruption_point in (
                "after_artifacts",
                "after_generation_rename",
                "before_pointer_replace",
            ):
                with self.subTest(interruption_point=interruption_point):

                    def interrupt(stage: str) -> None:
                        if stage == interruption_point:
                            raise RuntimeError("interrupted")

                    with self.assertRaisesRegex(RuntimeError, "interrupted"):
                        publish_generation(
                            output,
                            contract=second_contract,
                            candidates=[first, second],
                            selected=second,
                            write_resume=lambda path: path.write_bytes(b"resume-two"),
                            write_best_weights=lambda path: path.write_bytes(
                                b"weights-two"
                            ),
                            interruption_hook=interrupt,
                        )

                    loaded = load_current_generation(
                        output,
                        config=config,
                        optimization=optimization,
                        environment=environment,
                    )
                    self.assertEqual(loaded.manifest.epoch, 1)
                    self.assertEqual(loaded.resume_path.read_bytes(), b"resume-one")
                    self.assertEqual(
                        loaded.best_weights_path.read_bytes(), b"weights-one"
                    )
                    self.assertEqual(loaded.candidates, (first,))
                    validate_generation_resume_contract(loaded, first_contract)

    def test_resume_rejects_tampered_generation_artifact(self) -> None:
        config = _training_config()
        optimization = OptimizationConfig.for_profile("pilot")
        environment = _environment()
        candidate = CheckpointCandidate("epoch-0001", 1, 10, 0.4, 0.3)
        contract = ResumeContract.create(
            config=config,
            optimization=optimization,
            environment=environment,
            completed_epoch=1,
            global_step=10,
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            reserve_run_directory(output)
            published = publish_generation(
                output,
                contract=contract,
                candidates=[candidate],
                selected=candidate,
                write_resume=lambda path: path.write_bytes(b"resume"),
                write_best_weights=lambda path: path.write_bytes(b"weights"),
            )
            published.history_path.write_bytes(b"tampered")

            with self.assertRaisesRegex(ValueError, "history artifact digest"):
                load_current_generation(
                    output,
                    config=config,
                    optimization=optimization,
                    environment=environment,
                )

    def test_first_epoch_interruption_resumes_from_clean_epoch_zero(self) -> None:
        config = _training_config()
        optimization = OptimizationConfig.for_profile("pilot")
        environment = _environment()
        candidate = CheckpointCandidate("epoch-0001", 1, 10, 0.4, 0.3)
        contract = ResumeContract.create(
            config=config,
            optimization=optimization,
            environment=environment,
            completed_epoch=1,
            global_step=10,
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            reserve_run_directory(output)

            def interrupt(stage: str) -> None:
                if stage == "after_generation_rename":
                    raise RuntimeError("interrupted")

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                publish_generation(
                    output,
                    contract=contract,
                    candidates=[candidate],
                    selected=candidate,
                    write_resume=lambda path: path.write_bytes(b"resume"),
                    write_best_weights=lambda path: path.write_bytes(b"weights"),
                    interruption_hook=interrupt,
                )

            self.assertIsNone(
                load_current_generation(
                    output,
                    config=config,
                    optimization=optimization,
                    environment=environment,
                )
            )


if __name__ == "__main__":
    unittest.main()
