from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, Mock, patch

from poidh_detector.calibration_fit import (
    CalibrationPrediction,
    CalibrationPredictions,
)
from poidh_detector.contracts import SampleRecord
from poidh_detector.data import SplitManifest
from poidh_detector.inference import (
    FIXED_THRESHOLD,
    build_inference_result,
    load_selected_checkpoint,
    partition_sha256,
    predict_logits,
    run_inference,
)
from poidh_detector.reproducibility import EnvironmentFingerprint
from poidh_detector.torch_training import (
    OptimizationConfig,
    ResumeContract,
    publish_generation,
    reserve_run_directory,
)
from poidh_detector.training import CheckpointCandidate, TrainingConfig
from tools.predict_detector import main


_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _split_manifest() -> SplitManifest:
    return SplitManifest(
        dataset_manifest_sha256=_digest("dataset"),
        seed=323,
        ratios={"train": 0.5, "validation": 0.25, "calibration": 0.25},
        assignments={
            "train-ai": "train",
            "train-real": "train",
            "validation-ai": "validation",
            "validation-real": "validation",
            "calibration-ai": "calibration",
            "calibration-real": "calibration",
        },
    )


def _training_config(split: SplitManifest | None = None) -> TrainingConfig:
    frozen_split = split or _split_manifest()
    return TrainingConfig(
        dataset_manifest_sha256=frozen_split.dataset_manifest_sha256,
        split_manifest_sha256=frozen_split.sha256,
        calibration_split_sha256=partition_sha256(
            frozen_split.assignments, "calibration"
        ),
        exposed_holdout_sha256=(_digest("mirage-v1"), _digest("mirage-v2")),
        seed=323,
    )


def _environment(*, torch_version: str = "2.9.1+cu128") -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        python_version="3.13.5",
        platform="linux-test",
        machine="x86_64",
        torch_version=torch_version,
        numpy_version="2.2.6",
        timm_version="1.0.28",
        cuda_available=True,
        cuda_version="12.8",
        cudnn_version="9.10.2",
        cuda_device="NVIDIA H100",
    )


def _rows(partition: str) -> tuple[CalibrationPrediction, ...]:
    return (
        CalibrationPrediction(f"{partition}-ai", 2.0, 1),
        CalibrationPrediction(f"{partition}-real", -2.0, 0),
    )


def _sample_records() -> tuple[SampleRecord, ...]:
    rows = []
    for sample_id, label in (
        ("train-ai", 1),
        ("train-real", 0),
        ("validation-ai", 1),
        ("validation-real", 0),
        ("calibration-ai", 1),
        ("calibration-real", 0),
    ):
        rows.append(
            SampleRecord(
                sample_id=sample_id,
                source_id="source",
                upstream_path=f"upstream/{sample_id}.png",
                local_path=f"images/{sample_id}.png",
                label=label,
                content_sha256=_digest(f"content:{sample_id}"),
                provenance_group=f"group:{sample_id}",
                generator_family="test-generator" if label else None,
                content_type="image/png",
            )
        )
    return tuple(rows)


def _publish_run(root: Path) -> tuple[Path, TrainingConfig, EnvironmentFingerprint]:
    import torch

    run = root / "run"
    config = _training_config()
    optimization = OptimizationConfig.for_profile("pilot")
    environment = _environment()
    reserve_run_directory(run)
    (run / "training-config.json").write_bytes(config.to_json_bytes())
    (run / "optimization-config.json").write_bytes(optimization.to_json_bytes())
    (run / "environment.json").write_text(
        json.dumps(environment.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidate = CheckpointCandidate("epoch-0001", 1, 10, 0.25, 0.3)
    contract = ResumeContract.create(
        config=config,
        optimization=optimization,
        environment=environment,
        completed_epoch=1,
        global_step=10,
    )
    source_model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        source_model.weight.fill_(2.0)
        source_model.bias.fill_(-0.5)
    publish_generation(
        run,
        contract=contract,
        candidates=(candidate,),
        selected=candidate,
        write_resume=lambda path: path.write_bytes(b"unused-resume"),
        write_best_weights=lambda path: torch.save(source_model.state_dict(), path),
    )
    return run, config, environment


class InferenceResultTests(unittest.TestCase):
    def test_calibration_output_is_canonical_and_accepted_by_fitter(self) -> None:
        split = _split_manifest()
        config = _training_config(split)
        checkpoint_sha256 = _digest("checkpoint")
        forward = build_inference_result(
            partition="calibration",
            predictions=_rows("calibration"),
            training_config=config,
            split_manifest=split,
            checkpoint_sha256=checkpoint_sha256,
            environment=_environment(),
        )
        reverse = build_inference_result(
            partition="calibration",
            predictions=tuple(reversed(_rows("calibration"))),
            training_config=config,
            split_manifest=split,
            checkpoint_sha256=checkpoint_sha256,
            environment=_environment(),
        )

        parsed = CalibrationPredictions.from_json_bytes(
            forward.predictions_json_bytes()
        )
        self.assertEqual(parsed.checkpoint_sha256, checkpoint_sha256)
        self.assertEqual(parsed.training_config_sha256, config.sha256)
        self.assertEqual(
            parsed.calibration_split_sha256, config.calibration_split_sha256
        )
        self.assertEqual(
            forward.predictions_json_bytes(), reverse.predictions_json_bytes()
        )
        self.assertEqual(forward.metrics.auc, 1.0)
        self.assertEqual(forward.metrics.balanced_accuracy, 1.0)
        self.assertEqual(forward.metrics.threshold, FIXED_THRESHOLD)

    def test_validation_report_preserves_all_hash_provenance(self) -> None:
        split = _split_manifest()
        config = _training_config(split)
        result = build_inference_result(
            partition="validation",
            predictions=_rows("validation"),
            training_config=config,
            split_manifest=split,
            checkpoint_sha256=_digest("checkpoint"),
            environment=_environment(),
        )

        document = json.loads(result.predictions_json_bytes())
        self.assertEqual(document["partition"], "validation")
        self.assertEqual(
            document["dataset_manifest_sha256"], config.dataset_manifest_sha256
        )
        self.assertEqual(document["split_manifest_sha256"], split.sha256)
        self.assertEqual(document["training_config_sha256"], config.sha256)
        self.assertEqual(
            document["partition_sha256"],
            partition_sha256(split.assignments, "validation"),
        )
        self.assertEqual(document["metrics"]["threshold"], 0.65)
        self.assertRegex(document["environment_sha256"], r"^[0-9a-f]{64}$")

    def test_refuses_train_inference_or_incomplete_and_mismatched_partitions(
        self,
    ) -> None:
        split = _split_manifest()
        config = _training_config(split)
        common = {
            "training_config": config,
            "split_manifest": split,
            "checkpoint_sha256": _digest("checkpoint"),
            "environment": _environment(),
        }
        with self.assertRaisesRegex(ValueError, "calibration or validation"):
            build_inference_result(
                partition="train", predictions=_rows("train"), **common
            )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            build_inference_result(
                partition="validation",
                predictions=(_rows("validation")[0],),
                **common,
            )

        changed_config = replace(config, split_manifest_sha256=_digest("other"))
        with self.assertRaisesRegex(ValueError, "split manifest digest mismatch"):
            build_inference_result(
                partition="validation",
                predictions=_rows("validation"),
                **(common | {"training_config": changed_config}),
            )

    def test_does_not_claim_partition_digest_proves_holdout_overlap(self) -> None:
        split = _split_manifest()
        validation_digest = partition_sha256(split.assignments, "validation")
        config = replace(
            _training_config(split),
            exposed_holdout_sha256=(validation_digest, _digest("mirage-v1")),
        )

        result = build_inference_result(
            partition="validation",
            predictions=_rows("validation"),
            training_config=config,
            split_manifest=split,
            checkpoint_sha256=_digest("checkpoint"),
            environment=_environment(),
        )

        self.assertEqual(result.partition_sha256, validation_digest)

    def test_metrics_require_both_classes_and_use_fixed_threshold(self) -> None:
        split = _split_manifest()
        config = _training_config(split)
        rows = (
            CalibrationPrediction("validation-ai", -2.0, 1),
            CalibrationPrediction("validation-real", 2.0, 0),
        )
        result = build_inference_result(
            partition="validation",
            predictions=rows,
            training_config=config,
            split_manifest=split,
            checkpoint_sha256=_digest("checkpoint"),
            environment=_environment(),
        )
        self.assertEqual(result.metrics.auc, 0.0)
        self.assertEqual(result.metrics.balanced_accuracy, 0.0)

        with self.assertRaisesRegex(ValueError, "both classes"):
            build_inference_result(
                partition="validation",
                predictions=(
                    CalibrationPrediction("validation-ai", 2.0, 1),
                    CalibrationPrediction("validation-real", 1.0, 1),
                ),
                training_config=config,
                split_manifest=split,
                checkpoint_sha256=_digest("checkpoint"),
                environment=_environment(),
            )


class SelectedCheckpointTests(unittest.TestCase):
    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch integration test")
    def test_loads_only_verified_current_generation_with_exact_environment(
        self,
    ) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            run, config, environment = _publish_run(Path(temporary))
            loaded = load_selected_checkpoint(
                run,
                device="cpu",
                observed_environment=environment,
                torch_module=torch,
                model_factory=lambda: torch.nn.Linear(1, 1),
            )

            self.assertEqual(loaded.training_config, config)
            self.assertEqual(
                loaded.checkpoint_sha256, loaded.generation.manifest.best_weights_sha256
            )
            self.assertFalse(loaded.model.training)
            self.assertEqual(loaded.model(torch.tensor([[2.0]])).item(), 3.5)

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch integration test")
    def test_rejects_changed_environment_or_checkpoint_bytes(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            run, _, environment = _publish_run(Path(temporary))
            with self.assertRaisesRegex(ValueError, "environment provenance mismatch"):
                load_selected_checkpoint(
                    run,
                    device="cpu",
                    observed_environment=replace(environment, timm_version="1.0.29"),
                    torch_module=torch,
                    model_factory=lambda: torch.nn.Linear(1, 1),
                )

            pointer = json.loads((run / "CURRENT").read_bytes())
            weights = run / "generations" / pointer["generation"] / "best-model.pt"
            weights.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "best weights artifact digest"):
                load_selected_checkpoint(
                    run,
                    device="cpu",
                    observed_environment=environment,
                    torch_module=torch,
                    model_factory=lambda: torch.nn.Linear(1, 1),
                )

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch integration test")
    def test_predict_logits_preserves_dataset_order(self) -> None:
        import torch

        class Samples(torch.utils.data.Dataset):
            rows = (
                (torch.tensor([2.0]), 1.0, "z"),
                (torch.tensor([-2.0]), 0.0, "a"),
            )

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int):
                return self.rows[index]

        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.zero_()
        predictions = predict_logits(
            model,
            Samples(),
            batch_size=2,
            workers=0,
            device="cpu",
            torch_module=torch,
        )

        self.assertEqual([row.sample_id for row in predictions], ["z", "a"])
        self.assertEqual([row.raw_logit for row in predictions], [2.0, -2.0])
        self.assertEqual([row.label for row in predictions], [1, 0])

    def test_run_inference_verifies_data_and_uses_only_named_partition(self) -> None:
        split = _split_manifest()
        config = _training_config(split)
        manifest = Mock(sha256=config.dataset_manifest_sha256)
        selected = SimpleNamespace(
            model=object(),
            training_config=config,
            environment=_environment(),
            checkpoint_sha256=_digest("checkpoint"),
        )
        samples = (object(), object())
        dataset = object()
        with (
            patch(
                "poidh_detector.inference.load_selected_checkpoint",
                return_value=selected,
            ) as load_checkpoint,
            patch(
                "poidh_detector.inference.load_dataset_manifest",
                return_value=manifest,
            ),
            patch(
                "poidh_detector.inference.load_split_manifest", return_value=split
            ) as load_split,
            patch(
                "poidh_detector.inference.samples_for_split", return_value=samples
            ) as select_samples,
            patch(
                "poidh_detector.inference.DatasetImageSamples", return_value=dataset
            ) as build_dataset,
            patch(
                "poidh_detector.inference.predict_logits",
                return_value=_rows("validation"),
            ) as predict,
            patch("poidh_detector.inference.configure_determinism") as deterministic,
        ):
            result = run_inference(
                Path("/dataset"),
                Path("/run"),
                partition="validation",
                batch_size=8,
                workers=0,
                device="cpu",
                torch_module=object(),
            )

        self.assertEqual(result.partition, "validation")
        load_checkpoint.assert_called_once()
        load_split.assert_called_once_with(
            Path("/dataset/splits.json"),
            manifest,
            expected_sha256=config.split_manifest_sha256,
        )
        manifest.verify_materialized_files.assert_called_once_with(Path("/dataset"))
        deterministic.assert_called_once_with(config.seed, torch_module=ANY)
        select_samples.assert_called_once_with(manifest, split, "validation")
        build_dataset.assert_called_once_with(samples, Path("/dataset"))
        predict.assert_called_once()

        with self.assertRaisesRegex(ValueError, "calibration or validation"):
            run_inference(
                Path("/dataset"),
                Path("/run"),
                partition="train",
                batch_size=8,
                workers=0,
                device="cpu",
                torch_module=object(),
            )

    def test_run_inference_rejects_real_sample_id_overlap_with_registered_holdout(
        self,
    ) -> None:
        split = _split_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            holdout_path = root / "mirage-v1.json"
            holdout_payload = json.dumps(
                {
                    "schema_version": 1,
                    "entries": [
                        {
                            "file_name": "validation-ai",
                            "content_sha256": _digest("different-content"),
                        }
                    ],
                },
                sort_keys=True,
            ).encode()
            holdout_path.write_bytes(holdout_payload)
            holdout_digest = hashlib.sha256(holdout_payload).hexdigest()
            config = replace(
                _training_config(split),
                exposed_holdout_sha256=(holdout_digest,),
            )
            (root / "preparation.json").write_text(
                json.dumps(
                    {
                        "manifest_sha256": config.dataset_manifest_sha256,
                        "splits_sha256": config.split_manifest_sha256,
                        "exposed_holdout_manifest_sha256": [holdout_digest],
                    }
                ),
                encoding="utf-8",
            )
            manifest = SimpleNamespace(
                sha256=config.dataset_manifest_sha256,
                samples=_sample_records(),
                verify_materialized_files=Mock(),
            )
            selected = SimpleNamespace(
                model=object(),
                training_config=config,
                environment=_environment(),
                checkpoint_sha256=_digest("checkpoint"),
            )
            with (
                patch(
                    "poidh_detector.inference.load_selected_checkpoint",
                    return_value=selected,
                ),
                patch(
                    "poidh_detector.inference.load_dataset_manifest",
                    return_value=manifest,
                ),
                patch(
                    "poidh_detector.inference.load_split_manifest",
                    return_value=split,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "sample ID overlap"):
                    run_inference(
                        root,
                        Path("/run"),
                        partition="validation",
                        holdout_manifests=(holdout_path,),
                        batch_size=8,
                        workers=0,
                        device="cpu",
                        torch_module=object(),
                    )


class PredictionCliTests(unittest.TestCase):
    def test_cli_requires_registered_holdout_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch("tools.predict_detector.run_inference"):
                with self.assertRaises(SystemExit):
                    main(
                        [
                            "/dataset",
                            "/run",
                            "--partition",
                            "validation",
                            "--output",
                            str(Path(temporary) / "predictions.json"),
                        ]
                    )

    def test_cli_writes_fitter_compatible_output_without_overwrite(self) -> None:
        split = _split_manifest()
        result = build_inference_result(
            partition="calibration",
            predictions=_rows("calibration"),
            training_config=_training_config(split),
            split_manifest=split,
            checkpoint_sha256=_digest("checkpoint"),
            environment=_environment(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.json"
            from unittest.mock import patch

            with patch("tools.predict_detector.run_inference", return_value=result):
                exit_code = main(
                    [
                        "/dataset",
                        "/run",
                        "--partition",
                        "calibration",
                        "--output",
                        str(output),
                        "--holdout-manifest",
                        "/holdouts/mirage-v1.json",
                        "--device",
                        "cpu",
                    ]
                )
            self.assertEqual(exit_code, 0)
            CalibrationPredictions.from_json_bytes(output.read_bytes())

            with patch("tools.predict_detector.run_inference", return_value=result):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    main(
                        [
                            "/dataset",
                            "/run",
                            "--partition",
                            "calibration",
                            "--output",
                            str(output),
                            "--holdout-manifest",
                            "/holdouts/mirage-v1.json",
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
